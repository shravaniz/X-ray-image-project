# Running the KL-Grade UI on another system

No training needed - the 20 required checkpoints ship with the repo.

## 1. Clone (Git LFS required for the model files)
```bash
# one-time install if missing:
#   sudo apt-get install git-lfs && git lfs install
git lfs install
git clone https://github.com/shravaniz/X-ray-image-project.git
cd X-ray-image-project/OAI-KL
```
If the `.pt` files are tiny text stubs instead of ~80-170 MB each, run
`git lfs pull`.

## 2. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu
pip install albumentations opencv-python pandas numpy tqdm natsort scikit-learn ttach
pip install gradio
```
(CPU-only works too; GPU is optional and just makes inference faster.)

## 3. Run
```bash
python app.py
```
Open http://127.0.0.1:7860  (or http://<machine-ip>:7860 from another device on the LAN).

## What ships in the repo
- `app.py`, `inference.py`         - the UI and prediction code
- `model.py`                       - architecture definitions
- `dataset.py`                     - simple image loader used by inference
- `models_inference/`              - the 20 checkpoints the app loads (~2.4 GB, Git LFS)
- `requirements-ui.txt`            - the UI extra dependency

## Env vars
| Var | Default | Meaning |
|---|---|---|
| `KL_MODELS_DIR` | `./models_inference` | folder with the `<model>/(<size>, <size>)/<fold>fold_epoch<e>.pt` tree |
| `KL_SHARE` | unset | `1` -> create a temporary public `*.gradio.live` link |

## Notes
- The app only needs these 4 architectures: efficientnet_v2_s (456), regnet_y_8gf (384),
  resnet_101 (384), resnext_50_32x4d (512).
- The example images on the left come from the held-out test set and are optional;
  if `KneeXray/test/test_correct.csv` is absent the Examples panel is empty and
  everything else still works.
- Research/educational use only - not a medical device.
