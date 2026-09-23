"""
KL-Grade UI  -  upload a knee X-ray, get a Kellgren-Lawrence grade.

Backed by the reproduced 4-model ensemble
(EfficientNet-V2-s + RegNet-Y-8GF + ResNet-101 + ResNeXt-50-32x4d).

Run:
    pip install gradio
    python app.py
then open the printed local URL (default http://127.0.0.1:7860).

Env:
    KL_MODELS_DIR  path to the trained-model tree (default ./models)
    KL_SHARE=1     create a public *.gradio.live link
"""
import os

import gradio as gr
import numpy as np
import pandas as pd

import inference

CSS = """
.kl-header {text-align:center;}
.kl-footer {font-size:0.85em; color:#666; text-align:center; margin-top:8px;}
"""

GRADE_COLORS = {0: "#2e7d32", 1: "#9e9d24", 2: "#ef6c00", 3: "#d84315", 4: "#b71c1c"}


def _grade_summary_md(grade, probs):
    color = GRADE_COLORS[grade]
    desc = inference.GRADE_DESC[grade]
    return f"<h2 style='color:{color}; margin-bottom:0'>Predicted KL Grade: {grade}</h2>" \
           f"<p style='font-size:1.05em; margin-top:4px'>{desc}</p>"


def run(image):
    if image is None:
        return "<p>Please upload a knee X-ray image.</p>", None
    try:
        grade, probs, _ = inference.predict(image)
    except Exception as e:
        return f"<p style='color:#b71c1c'>Error: {e}</p>", None

    html = _grade_summary_md(grade, probs)
    conf = float(probs[grade]) * 100
    html += f"<p>Model confidence for this grade: <b>{conf:.1f}%</b></p>"

    # BarPlot needs the categories as a real column (it ignores the DataFrame
    # index), plus an explicit x/y mapping.
    chart = pd.DataFrame({
        "grade": [f"KL {i}" for i in range(5)],
        "probability": [float(p) for p in probs],
    })
    return html, chart


def main():
    models = inference.available_models()
    status = ("Models available: " + ", ".join(models)) if models else \
             "No trained models found - set KL_MODELS_DIR."

    with gr.Blocks(title="Knee KL-Grade Classifier") as demo:
        gr.Markdown(
            "# Knee Osteoarthritis (KL-Grade) Classifier\n"
            "Upload a **knee X-ray** (AP view, single knee) and the ensemble predicts the "
            "**Kellgren-Lawrence grade (0-4)**.\n\n"
            f"`{status}`")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Knee X-ray image")
                btn = gr.Button("Predict KL Grade", variant="primary")
                gr.Examples(
                    examples=[[p] for p in _example_images()],
                    inputs=inp, label="Example images (from the held-out test set)")
            with gr.Column(scale=1):
                out_html = gr.HTML("<p>Prediction will appear here.</p>")
                out_chart = gr.BarPlot(
                    x="grade", y="probability", title="Class probabilities",
                    x_title="KL grade", y_title="probability",
                    y_lim=[0, 1], height=280, tooltip="probability",
                    color="grade", sort="grade", x_label_angle=0)

        btn.click(run, inputs=inp, outputs=[out_html, out_chart])
        inp.change(run, inputs=inp, outputs=[out_html, out_chart])

        gr.Markdown(
            "**Ensemble:** EfficientNet-V2-s + RegNet-Y-8GF + ResNet-101 + ResNeXt-50-32x4d "
            "(5-fold, horizontal-flip TTA, soft voting). "
            "Reported full-test accuracy 76.87% (KL 0-4).\n\n"
            "⚠️ Research/educational use only - **not a medical device** and not a substitute "
            "for a radiologist's diagnosis.",
            elem_classes=["kl-footer"])

    demo.launch(server_name="0.0.0.0",
                css=CSS,
                share=os.environ.get("KL_SHARE") == "1")


def _example_images():
    """Grab a few held-out test images to show as examples (if present)."""
    csv = "./KneeXray/test/test_correct.csv"
    if not os.path.exists(csv):
        return []
    df = pd.read_csv(csv)
    picks = []
    for g in range(5):
        sub = df[df["label"] == g]
        if len(sub):
            p = sub.iloc[0]["data"]
            if os.path.exists(p):
                picks.append(p)
    return picks


if __name__ == "__main__":
    main()
