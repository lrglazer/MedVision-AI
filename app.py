from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/best_model.pt")
ROC_CURVE_PATH = Path("models/roc_curve.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

TEST_AUC = 0.7744
TEST_SENSITIVITY = 0.7633
TEST_SPECIFICITY = 0.6508
TEST_PRECISION = 0.0481

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = None


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #030817;
            --surface: #08172d;
            --surface-2: #0b203e;
            --surface-3: #0d294f;
            --line: rgba(66, 170, 255, 0.22);
            --line-strong: rgba(66, 190, 255, 0.48);
            --blue: #168cff;
            --cyan: #35d8ff;
            --text: #f7fbff;
            --muted: #a9bdd6;
        }

        .stApp {
            background:
                radial-gradient(circle at 78% 4%, rgba(0, 126, 255, 0.16), transparent 24%),
                radial-gradient(circle at 18% 18%, rgba(0, 80, 190, 0.11), transparent 28%),
                linear-gradient(160deg, #020714 0%, #061226 52%, #020714 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020a18 0%, #06152a 100%);
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1460px;
            padding-top: 1.55rem;
            padding-bottom: 4rem;
        }

        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
        }

        p, label, .stCaption {
            color: var(--muted);
        }

        div[data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: linear-gradient(
                145deg,
                rgba(11, 37, 72, 0.95),
                rgba(4, 18, 39, 0.96)
            );
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stMetricLabel"] {
            color: #7acbff !important;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 800;
            font-size: 1.85rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        [data-testid="stFileUploader"] {
            padding: 1rem;
            border: 1px dashed var(--line-strong);
            border-radius: 18px;
            background: linear-gradient(
                145deg,
                rgba(9, 32, 63, 0.94),
                rgba(4, 18, 40, 0.94)
            );
        }

        [data-testid="stFileUploader"]:hover {
            border-color: var(--cyan);
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 120px;
            border: none;
            background: transparent;
        }

        [data-testid="stTabs"] {
            padding: 0.35rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(4, 18, 40, 0.72);
        }

        [data-testid="stTabs"] button {
            border-radius: 10px;
            color: #9fb4ce;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: white;
            background: rgba(17, 109, 210, 0.36);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid rgba(53, 178, 255, 0.44);
            border-radius: 10px;
            background: linear-gradient(115deg, #0d72df, #084a9e);
            color: white;
            font-weight: 700;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cyan);
            color: white;
            box-shadow: 0 8px 24px rgba(22, 139, 255, 0.25);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        [data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, var(--blue), var(--cyan));
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 17px !important;
            background: linear-gradient(
                145deg,
                rgba(10, 34, 67, 0.88),
                rgba(4, 17, 38, 0.90)
            );
        }

        hr {
            border-color: rgba(56, 171, 255, 0.14);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL + PREPROCESSING
# ============================================================

@st.cache_resource
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. "
            "Place best_model.pt inside the models folder."
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def prepare_image(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image, np.ndarray, torch.Tensor]:
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = (
        np.asarray(resized_image)
        .astype(np.float32)
        / 255.0
    )

    input_tensor = (
        image_transform(rgb_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return rgb_image, resized_image, image_array, input_tensor


def predict(model: nn.Module, input_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        logit = model(input_tensor)
        return float(torch.sigmoid(logit).item())


def create_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    with GradCAM(
        model=model,
        target_layers=[model.layer4[-1]],
    ) as cam:
        return cam(
            input_tensor=input_tensor,
            targets=None,
        )[0]


def create_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    intensity: float,
) -> np.ndarray:
    overlay = show_cam_on_image(
        image_array,
        heatmap,
        use_rgb=True,
    ).astype(np.float32) / 255.0

    blended = (
        (1.0 - intensity) * image_array
        + intensity * overlay
    )

    return np.clip(
        blended * 255,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# HELPERS
# ============================================================

def score_band(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "Elevated"
    return "High"


def short_classification(is_positive: bool) -> str:
    return "Pneumonia" if is_positive else "No Finding"


def add_history_record(
    *,
    filename: str,
    probability: float,
    threshold: float,
    classification: str,
) -> None:
    record_key = (
        filename,
        round(probability, 6),
        round(threshold, 2),
    )

    existing_keys = {
        (
            item["Filename"],
            round(float(item["Pneumonia Score"]), 6),
            round(float(item["Threshold"]), 2),
        )
        for item in st.session_state.analysis_history
    }

    if record_key in existing_keys:
        return

    st.session_state.analysis_history.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": filename,
            "Pneumonia Score": round(probability, 6),
            "Displayed Score": f"{probability:.1%}",
            "Threshold": round(threshold, 2),
            "Classification": classification,
            "Score Band": score_band(probability),
        }
    )


def build_report(
    *,
    filename: str,
    dimensions: tuple[int, int],
    probability: float,
    threshold: float,
    classification: str,
) -> str:
    return f"""
MEDVISION AI RESEARCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE
Filename: {filename}
Original dimensions: {dimensions[0]} x {dimensions[1]} pixels
Model input dimensions: {IMAGE_SIZE} x {IMAGE_SIZE} pixels

MODEL OUTPUT
Classification: {classification}
Pneumonia-class score: {probability:.6f}
Displayed score: {probability:.1%}
Decision threshold: {threshold:.2f}
Score band: {score_band(probability)}

MODEL
Architecture: ResNet-18
Framework: PyTorch
Runtime device: {DEVICE}

HELD-OUT TEST PERFORMANCE
ROC-AUC: {TEST_AUC:.4f}
Sensitivity: {TEST_SENSITIVITY:.4f}
Specificity: {TEST_SPECIFICITY:.4f}
Pneumonia precision: {TEST_PRECISION:.4f}

LIMITATIONS
This is an educational research prototype, not a medical device.
It must not be used for diagnosis, screening, or treatment decisions.
The model score is not a calibrated clinical probability.
Grad-CAM does not identify lesions or establish valid medical reasoning.
""".strip()


def render_processing_sequence() -> None:
    progress = st.progress(0)
    status = st.empty()

    steps = [
        ("Preprocessing image", 20),
        ("Running neural network", 58),
        ("Computing Grad-CAM", 84),
        ("Preparing dashboard", 100),
    ]

    for text, value in steps:
        status.caption(f"✓ {text}")
        progress.progress(value)
        time.sleep(0.10)

    status.empty()
    progress.empty()


def render_summary_card(
    *,
    probability: float,
    threshold: float,
    is_positive: bool,
) -> None:
    classification = short_classification(is_positive)

    with st.container(border=True):
        st.subheader("AI research summary")

        summary_left, summary_right = st.columns(2)

        with summary_left:
            st.write(f"**Classification:** {classification}")
            st.write(f"**Pneumonia score:** {probability:.1%}")
            st.write(f"**Score band:** {score_band(probability)}")

        with summary_right:
            st.write(f"**Decision threshold:** {threshold:.0%}")
            st.write(
                "**Threshold relationship:** "
                + (
                    "Score is at or above threshold"
                    if is_positive
                    else "Score is below threshold"
                )
            )
            st.write("**Use:** Educational research only")

        st.info(
            "This summary explains the application output. "
            "It is not medical advice and does not interpret the image clinically."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🫁 MedVision AI")
    st.caption("Chest X-ray research dashboard")

    st.divider()

    st.subheader("Analysis settings")

    decision_threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Scores at or above this value display as Pneumonia."
        ),
    )

    heatmap_intensity = st.slider(
        "Heatmap intensity",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
    )

    show_details = st.toggle(
        "Show technical details",
        value=True,
    )

    st.divider()

    st.subheader("Model performance")

    st.metric("ROC-AUC", f"{TEST_AUC:.3f}")
    st.metric("Sensitivity", f"{TEST_SENSITIVITY:.2%}")
    st.metric("Specificity", f"{TEST_SPECIFICITY:.2%}")
    st.metric("Precision", f"{TEST_PRECISION:.2%}")

    with st.expander("Why is precision low?"):
        st.write(
            "Pneumonia was rare in the training set. "
            "Class weighting improved sensitivity but increased "
            "false-positive predictions."
        )

    st.divider()

    st.subheader("Runtime")
    st.caption(f"Device: {DEVICE}")
    st.caption("Architecture: ResNet-18")
    st.caption(f"Input: {IMAGE_SIZE} × {IMAGE_SIZE}")

    st.warning("Educational research use only. Not for diagnosis.")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([1.35, 1], gap="large")

with header_left:
    st.caption("🔵 LIVE RESEARCH PROTOTYPE")
    st.title("🫁 MedVision AI")
    st.caption("BIOMEDICAL COMPUTER-VISION RESEARCH DASHBOARD")
    st.subheader("See what an AI model notices in a chest X-ray.")

    st.write(
        "Explore a ResNet-18 chest X-ray classifier with adjustable "
        "decision thresholds, Grad-CAM attention visualization, "
        "model-performance reporting, and session analysis history."
    )

    badge_one, badge_two, badge_three, badge_four = st.columns(4)
    badge_one.info("ResNet-18")
    badge_two.info("PyTorch")
    badge_three.info("Grad-CAM")
    badge_four.info("Patient split")

with header_right:
    with st.container(border=True):
        st.subheader("Held-out test performance")

        performance_one, performance_two = st.columns(2)
        performance_three, performance_four = st.columns(2)

        performance_one.metric("ROC-AUC", f"{TEST_AUC:.3f}")
        performance_two.metric(
            "Sensitivity",
            f"{TEST_SENSITIVITY:.2%}",
        )
        performance_three.metric(
            "Specificity",
            f"{TEST_SPECIFICITY:.2%}",
        )
        performance_four.metric(
            "Precision",
            f"{TEST_PRECISION:.2%}",
        )


st.divider()


# ============================================================
# UPLOAD + WORKFLOW
# ============================================================

st.header("Analyze a chest X-ray")

uploaded_file = st.file_uploader(
    "Drag and drop a PNG, JPG, or JPEG chest X-ray",
    type=["png", "jpg", "jpeg"],
    width="stretch",
)

workflow_one, workflow_two, workflow_three = st.columns(3)

with workflow_one:
    with st.container(border=True):
        st.caption("01 · UPLOAD")
        st.subheader("⬆️ Add an X-ray")
        st.write("Select a frontal chest X-ray from your computer.")

with workflow_two:
    with st.container(border=True):
        st.caption("02 · ANALYZE")
        st.subheader("🧠 Run inference")
        st.write("The model generates a pneumonia-class score.")

with workflow_three:
    with st.container(border=True):
        st.caption("03 · INTERPRET")
        st.subheader("👁️ Explore attention")
        st.write("Grad-CAM visualizes influential image regions.")


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:
    st.info("Upload a chest X-ray above to begin.")

    overview_tab, performance_tab, limitations_tab = st.tabs(
        [
            "Project overview",
            "Model performance",
            "Limitations",
        ]
    )

    with overview_tab:
        st.subheader("Project overview")
        st.write(
            "This application uses transfer learning with ResNet-18 "
            "to distinguish No Finding images from images whose NIH "
            "labels contain Pneumonia."
        )
        st.write(
            "The model was evaluated using a patient-level held-out "
            "test split to reduce leakage between training and testing."
        )

    with performance_tab:
        if ROC_CURVE_PATH.exists():
            st.image(
                str(ROC_CURVE_PATH),
                caption="Held-out test-set ROC curve",
                width="stretch",
            )
        else:
            st.info("ROC curve image not found.")

    with limitations_tab:
        st.warning(
            "The dataset labels are imperfect and strongly imbalanced. "
            "This project is not appropriate for clinical use."
        )


# ============================================================
# ANALYSIS
# ============================================================

else:
    try:
        uploaded_image = Image.open(uploaded_file)
        original_dimensions = uploaded_image.size

        (
            original_rgb,
            resized_image,
            image_array,
            input_tensor,
        ) = prepare_image(uploaded_image)

        current_file_key = (
            uploaded_file.name,
            uploaded_file.size,
            decision_threshold,
            heatmap_intensity,
        )

        if current_file_key != st.session_state.last_file_key:
            render_processing_sequence()
            st.session_state.last_file_key = current_file_key

        probability = predict(model, input_tensor)
        heatmap = create_gradcam(model, input_tensor)
        overlay = create_overlay(
            image_array,
            heatmap,
            heatmap_intensity,
        )

        is_positive = probability >= decision_threshold
        classification = (
            "Pneumonia class"
            if is_positive
            else "No Finding class"
        )

        add_history_record(
            filename=uploaded_file.name,
            probability=probability,
            threshold=decision_threshold,
            classification=classification,
        )

        st.success("Analysis complete")
        st.header("Model result")

        result_left, result_right = st.columns(
            [0.85, 1.45],
            gap="large",
        )

        with result_left:
            with st.container(border=True):
                st.caption("PNEUMONIA SCORE")
                st.title(f"{probability:.1%}")
                st.subheader(score_band(probability))
                st.progress(probability)
                st.caption(
                    "Model output only—not a calibrated "
                    "clinical probability."
                )

        with result_right:
            metric_one, metric_two, metric_three = st.columns(3)

            with metric_one:
                with st.container(border=True):
                    st.caption("CLASSIFICATION")
                    st.subheader(short_classification(is_positive))

            with metric_two:
                with st.container(border=True):
                    st.caption("THRESHOLD")
                    st.subheader(f"{decision_threshold:.0%}")

            with metric_three:
                with st.container(border=True):
                    st.caption("SCORE BAND")
                    st.subheader(score_band(probability))

            if is_positive:
                st.error(
                    f"The score of {probability:.1%} is at or above "
                    f"the selected threshold of {decision_threshold:.0%}. "
                    "This is a research-model output, not a diagnosis."
                )
            else:
                st.success(
                    f"The score of {probability:.1%} is below "
                    f"the selected threshold of {decision_threshold:.0%}. "
                    "This does not establish that the X-ray is medically normal."
                )

        (
            comparison_tab,
            attention_tab,
            summary_tab,
            explanation_tab,
            performance_tab,
        ) = st.tabs(
            [
                "Image comparison",
                "Attention map",
                "AI summary",
                "Explanation",
                "Performance",
            ]
        )

        with comparison_tab:
            original_column, heatmap_column = st.columns(2)

            with original_column:
                with st.container(border=True):
                    st.subheader("Original X-ray")
                    st.image(
                        resized_image,
                        caption=uploaded_file.name,
                        width="stretch",
                    )

            with heatmap_column:
                with st.container(border=True):
                    st.subheader("Grad-CAM overlay")
                    st.image(
                        overlay,
                        caption="Model-attention visualization",
                        width="stretch",
                    )

        with attention_tab:
            st.image(
                overlay,
                caption=(
                    "Warmer regions contributed more strongly "
                    "to this model output."
                ),
                width="stretch",
            )

            st.warning(
                "Grad-CAM is not a lesion outline. The model may respond "
                "to borders, text markers, equipment, positioning, or "
                "other dataset artifacts."
            )

        with summary_tab:
            render_summary_card(
                probability=probability,
                threshold=decision_threshold,
                is_positive=is_positive,
            )

        with explanation_tab:
            st.subheader("How to read the output")

            st.write(
                f"The model produced a pneumonia-class score of "
                f"**{probability:.1%}**."
            )

            st.write(
                f"At a threshold of **{decision_threshold:.0%}**, "
                f"the displayed classification is **{classification}**."
            )

            st.subheader("Why the threshold matters")

            st.write(
                "Lower thresholds usually increase sensitivity but also "
                "increase false positives. Higher thresholds generally "
                "reduce false positives but may miss more positive cases."
            )

            st.subheader("What this result does not mean")

            st.write(
                "It does not confirm pneumonia, rule out disease, "
                "or provide a patient's true clinical probability."
            )

        with performance_tab:
            performance_left, performance_right = st.columns([1.4, 1])

            with performance_left:
                if ROC_CURVE_PATH.exists():
                    st.image(
                        str(ROC_CURVE_PATH),
                        caption="Held-out test-set ROC curve",
                        width="stretch",
                    )
                else:
                    st.info("ROC curve image not found.")

            with performance_right:
                st.metric("ROC-AUC", f"{TEST_AUC:.4f}")
                st.metric(
                    "Sensitivity",
                    f"{TEST_SENSITIVITY:.2%}",
                )
                st.metric(
                    "Specificity",
                    f"{TEST_SPECIFICITY:.2%}",
                )
                st.metric(
                    "Precision",
                    f"{TEST_PRECISION:.2%}",
                )

        if show_details:
            with st.expander("Technical details"):
                detail_left, detail_right = st.columns(2)

                with detail_left:
                    st.write(f"**Filename:** `{uploaded_file.name}`")
                    st.write(
                        f"**Original dimensions:** "
                        f"`{original_dimensions[0]} × "
                        f"{original_dimensions[1]}`"
                    )
                    st.write(
                        f"**Model input:** `{IMAGE_SIZE} × {IMAGE_SIZE}`"
                    )
                    st.write(f"**Image mode:** `{original_rgb.mode}`")

                with detail_right:
                    st.write(f"**Runtime device:** `{DEVICE}`")
                    st.write(f"**Raw score:** `{probability:.6f}`")
                    st.write(
                        f"**Threshold:** `{decision_threshold:.2f}`"
                    )
                    st.write(
                        f"**Classification:** `{classification}`"
                    )

        report = build_report(
            filename=uploaded_file.name,
            dimensions=original_dimensions,
            probability=probability,
            threshold=decision_threshold,
            classification=classification,
        )

        st.download_button(
            "Download research report",
            data=report,
            file_name="medvision_research_report.txt",
            mime="text/plain",
            width="stretch",
        )

    except UnidentifiedImageError:
        st.error(
            "The selected file could not be read as an image. "
            "Please upload a valid PNG, JPG, or JPEG file."
        )
    except Exception as error:
        st.error(f"Could not process the uploaded image: {error}")


# ============================================================
# ANALYSIS HISTORY
# ============================================================

st.divider()
st.header("Analysis history")

if not st.session_state.analysis_history:
    st.info(
        "Analyzed X-rays will appear here during this browser session."
    )

else:
    history_df = pd.DataFrame(st.session_state.analysis_history)
    history_scores = history_df["Pneumonia Score"]

    history_one, history_two, history_three, history_four = st.columns(4)

    history_one.metric("Total analyses", len(history_df))
    history_two.metric(
        "Average score",
        f"{history_scores.mean():.1%}",
    )

    positive_count = (
        history_df["Classification"]
        .eq("Pneumonia class")
        .sum()
    )

    history_three.metric("Positive cases", int(positive_count))
    history_four.metric(
        "Highest score",
        f"{history_scores.max():.1%}",
    )

    display_df = history_df[
        [
            "Time",
            "Filename",
            "Displayed Score",
            "Threshold",
            "Classification",
            "Score Band",
        ]
    ].iloc[::-1]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Time": st.column_config.TextColumn("Analyzed"),
            "Filename": st.column_config.TextColumn("File"),
            "Displayed Score": st.column_config.TextColumn(
                "Pneumonia score"
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold",
                format="%.2f",
            ),
            "Classification": st.column_config.TextColumn(
                "Classification"
            ),
            "Score Band": st.column_config.TextColumn("Score band"),
        },
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    download_column, clear_column = st.columns(2)

    with download_column:
        st.download_button(
            "Download analysis history",
            data=csv_data,
            file_name="medvision_analysis_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with clear_column:
        if st.button("Clear history", width="stretch"):
            st.session_state.analysis_history = []
            st.session_state.last_file_key = None
            st.rerun()


st.divider()
st.caption(
    "MedVision AI · Biomedical Engineering & Computer Vision Project · "
    "Educational Research Use Only"
)