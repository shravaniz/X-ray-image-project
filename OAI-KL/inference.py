"""
Inference for the KL-grade UI.

Loads the *best 4-model ensemble* from the paper's reproduction:
    EfficientNet-V2-s (456) + RegNet-Y-8GF (384) + ResNet-101 (384) + ResNeXt-50-32x4d (512)
For each model we load one checkpoint per fold (the best epoch by test accuracy),
run horizontal-flip TTA, average the softmax probabilities across the 5 folds,
then soft-vote across the 4 models.  This reproduces the 76.87% full-test
ensemble reported in ENSEMBLE_ACCURACY_REPORT.txt.

Models are loaded lazily / cached on first use, so the UI can start fast.
"""
import os
import numpy as np

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

from model import model_return

# ---- configuration ----------------------------------------------------------
MODELS = {
    'efficientnet_v2_s': 456,
    'regnet_y_8gf':      384,
    'resnet_101':        384,
    'resnext_50_32x4d':  512,
}

# best epoch per fold for each model (from the reproduction run)
BEST_EPOCHS = {
    'efficientnet_v2_s': {1: 13, 2: 17, 3: 12, 4: 12, 5: 17},
    'regnet_y_8gf':      {1: 14, 2: 12, 3: 15, 4: 6,  5: 12},
    'resnet_101':        {1: 13, 2: 18, 3: 20, 4: 17, 5: 19},
    'resnext_50_32x4d':  {1: 11, 2: 11, 3: 12, 4: 16, 5: 13},
}

MODELS_DIR = os.environ.get('KL_MODELS_DIR', './models')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GRADE_DESC = {
    0: "KL 0 - Normal (no osteoarthritis)",
    1: "KL 1 - Doubtful narrowing, possible osteophytes",
    2: "KL 2 - Mild (definite osteophytes, possible narrowing)",
    3: "KL 3 - Moderate (multiple osteophytes, definite narrowing)",
    4: "KL 4 - Severe (large osteophytes, marked narrowing, sclerosis)",
}

# cache: {model_type: [list of loaded nets]}
_CACHE = {}


def _build_transform(size):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_CUBIC, p=1),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _load_model(model_type, image_size, fold, epoch):
    args = type('Args', (), {'model_type': model_type, 'image_size': image_size})()
    net = model_return(args)
    path = os.path.join(MODELS_DIR, model_type, f'({image_size}, {image_size})',
                        f'{fold}fold_epoch{epoch}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Set KL_MODELS_DIR to the folder containing the trained model tree, "
            f"or retrain with main.py.")
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval().to(DEVICE)
    return net


def _get_nets(model_type, image_size):
    """Load (and cache) the 5 per-fold nets for one architecture."""
    if model_type in _CACHE:
        return _CACHE[model_type]
    nets = []
    for fold, epoch in BEST_EPOCHS[model_type].items():
        nets.append(_load_model(model_type, image_size, fold, epoch))
    _CACHE[model_type] = nets
    return nets


def available_models():
    """Return the list of model types whose checkpoints are actually present."""
    ok = []
    for mt, sz in MODELS.items():
        d = os.path.join(MODELS_DIR, mt, f'({sz}, {sz})')
        if os.path.isdir(d) and any(
                os.path.exists(os.path.join(d, f'{f}fold_epoch{e}.pt'))
                for f, e in BEST_EPOCHS[mt].items()):
            ok.append(mt)
    return ok


@torch.no_grad()
def predict(image_bgr_or_path):
    """
    Run the ensemble on one knee X-ray.

    Accepts a BGR numpy image (as from cv2.imread / gr. Image) or a file path.
    Returns (grade:int, probs:np.ndarray[5], detail:dict).
    """
    if isinstance(image_bgr_or_path, str):
        img = cv2.imread(image_bgr_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_bgr_or_path}")
    else:
        img = image_bgr_or_path
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detail = {}
    model_probs = []
    for mt, size in MODELS.items():
        if mt not in _CACHE and mt not in available_models():
            continue  # skip missing models
        nets = _get_nets(mt, size)
        t = _build_transform(size)
        x = t(image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))['image'].float().unsqueeze(0).to(DEVICE)

        fold_probs = []
        for net in nets:
            # TTA: original + horizontal flip
            out = net(x)
            out_flip = net(torch.flip(x, dims=[3]))
            p = (nn.Softmax(dim=1)(out) + nn.Softmax(dim=1)(out_flip)) / 2.0
            fold_probs.append(p.cpu().numpy()[0])
        mp = np.mean(fold_probs, axis=0)          # average over folds
        model_probs.append(mp)
        detail[mt] = mp

    if not model_probs:
        raise RuntimeError("No trained models found. Set KL_MODELS_DIR or train with main.py.")

    ensemble = np.mean(model_probs, axis=0)       # soft-vote across models
    grade = int(np.argmax(ensemble))
    return grade, ensemble, detail


def format_result(grade, probs):
    lines = [f"### Predicted KL Grade: {grade}", "", GRADE_DESC[grade], "", "**Class probabilities:**", ""]
    for g in range(5):
        bar = '█' * int(round(probs[g] * 30))
        lines.append(f"- KL {g}: {probs[g]*100:5.1f}%  {bar}")
    lines.append("")
    lines.append("_Ensemble: EfficientNet-V2-s + RegNet-Y-8GF + ResNet-101 + ResNeXt-50-32x4d "
                 "(5-fold, flip-TTA, soft voting). Research use only — not a medical device._")
    return "\n".join(lines)
