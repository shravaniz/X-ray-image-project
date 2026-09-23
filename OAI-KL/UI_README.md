# Knee KL-Grade Web UI

Upload a knee X-ray (AP view, single knee) and the app predicts the
**Kellgren-Lawrence grade (0-4)**, with per-class probabilities.

## What it uses
A soft-voting ensemble of the four strongest reproduced models
(see `ENSEMBLE_ACCURACY_REPORT.txt`):

| Model | Input |
|---|---|
| EfficientNet-V2-s | 456 |
| RegNet-Y-8GF      | 384 |
| ResNet-101        | 384 |
| ResNeXt-50-32x4d  | 512 |

Each model contributes its best checkpoint per fold (5 folds), with
horizontal-flip TTA; fold probabilities are averaged, then the four models
are soft-voted. This reproduces the **76.87%** full-test accuracy.

## Run it
```bash
cd OAI-KL
source /path/to/venv/bin/activate
pip install gradio
python app.py
```
Open http://127.0.0.1:7860 (or the machine's IP at port 7860).

### Configuration (env vars)
| Var | Default | Meaning |
|---|---|---|
| `KL_MODELS_DIR` | `./models` | Folder containing `<model>/(<size>, <size>)/<fold>fold_epoch<e>.pt` |
| `KL_SHARE`      | unset | Set to `1` to create a public `*.gradio.live` link |

`app.py` starts even if models are missing (it shows which are available) so
you can confirm the setup before loading the heavy checkpoints.

## Files
- `inference.py` - model loading + ensemble prediction (importable, no UI deps)
- `app.py` - the Gradio interface
- `requirements-ui.txt` - the extra dependency

## Disclaimer
Research / educational use only. **Not a medical device**; not a substitute
for a radiologist's diagnosis.
