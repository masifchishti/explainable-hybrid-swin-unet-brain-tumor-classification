from __future__ import annotations

import hashlib
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_PATH,
)
from gradcam import create_visualizations, generate_notebook_cam
from history import save_prediction
from model_loader import load_model
from predictor import predict
from report import generate_report_bytes


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid Swin-UNet MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# COMPLETE UI STYLING
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --primary: #0b6477;
        --primary-dark: #083d56;
        --primary-light: #14919b;
        --card-bg: #ffffff;
        --border: #dce7eb;
        --text: #102a43;
        --muted: #6b7f8f;
        --success-bg: #e9f8f1;
        --success-border: #b7e7d1;
    }

    /* Keep the scrollbar permanently reserved.
       This prevents left-right page vibration. */
    html {
        overflow-y: scroll !important;
        scrollbar-gutter: stable !important;
    }

    body,
    .stApp,
    section.main,
    div[data-testid="stAppViewContainer"] {
        overflow-anchor: none !important;
        transform: none !important;
        will-change: auto !important;
    }

    /* Main page */
    .stApp {
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #f5f9fa 100%
            );
        color: var(--text);
    }

    .main .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Prevent UI cards from shaking or moving */
    div[data-testid="stMetric"],
    div[data-testid="stAlert"],
    div[data-testid="stImage"],
    div[data-testid="stDownloadButton"],
    div[data-testid="stFileUploader"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlock"],
    div[data-testid="stColumn"] {
        animation: none !important;
        transition: none !important;
        transform: none !important;
        will-change: auto !important;
    }

    /* Titles */
    h1 {
        color: var(--primary-dark);
        font-size: 2.45rem !important;
        line-height: 1.15 !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
        margin-bottom: 1.7rem !important;
    }

    h2 {
        color: var(--primary-dark);
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }

    h3,
    h4 {
        color: var(--text);
        font-weight: 700 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: transparent;
    }

    div[data-testid="stFileUploaderDropzone"] {
        min-height: 76px;
        border: 1.5px dashed #9fcbd3;
        border-radius: 12px;
        background: #f9fcfd;
        transition: none !important;
    }

    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary-light);
        background: #f3fbfc;
    }

    /* Images */
    [data-testid="stImage"] {
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
    }

    [data-testid="stImage"] img {
        display: block;
        border-radius: 12px;
        border: 1px solid #e1e8eb;
        box-shadow: 0 5px 16px rgba(16, 42, 67, 0.07);
        transform: none !important;
    }

    [data-testid="stImage"] p {
        color: var(--muted);
        text-align: center;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(16, 42, 67, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-dark);
        font-size: 1.9rem;
        font-weight: 800;
    }

    /* Prediction banner */
    div[data-testid="stAlert"] {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-radius: 12px;
        color: #176044;
        box-shadow: none;
    }

    /* Analyze MRI button */
    div[data-testid="stButton"] > button {
        min-height: 46px !important;
        border: none !important;
        border-radius: 11px !important;
        color: #ffffff !important;
        font-weight: 750 !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;

        background:
            linear-gradient(
                135deg,
                #16a34a 0%,
                #0f766e 52%,
                #f97316 100%
            ) !important;

        box-shadow:
            0 7px 18px rgba(15, 118, 110, 0.22) !important;

        animation: none !important;
        transition: none !important;
        transform: none !important;
    }

    div[data-testid="stButton"] > button:hover {
        color: #ffffff !important;
        border: none !important;

        background:
            linear-gradient(
                135deg,
                #15803d 0%,
                #0d9488 52%,
                #ea580c 100%
            ) !important;

        box-shadow:
            0 8px 20px rgba(15, 118, 110, 0.28) !important;

        transform: none !important;
    }

    div[data-testid="stButton"] > button:active,
    div[data-testid="stButton"] > button:focus {
        color: #ffffff !important;
        border: none !important;
        transform: none !important;
    }

    /* Download buttons */
    div[data-testid="stDownloadButton"] > button {
        min-height: 42px;
        width: 100%;
        border-radius: 9px;
        border: 1px solid #b9d7dc;
        background: white;
        color: var(--primary-dark);
        font-weight: 650;
        box-shadow: none;

        animation: none !important;
        transition: none !important;
        transform: none !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--primary);
        background: #f2fafb;
        color: var(--primary-dark);
        transform: none !important;
    }

    /* Stable probability table */
    .probability-table {
        width: 100% !important;
        table-layout: fixed;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .probability-table thead th {
        padding: 0.72rem 0.8rem;
        text-align: left;
        background: #f4f8f9;
        color: #526777;
        font-weight: 650;
        border-bottom: 1px solid var(--border);
    }

    .probability-table tbody td {
        padding: 0.68rem 0.8rem;
        color: var(--text);
        border-bottom: 1px solid #e7eef1;
    }

    .probability-table tbody tr:last-child td {
        border-bottom: none;
    }

    .probability-table th:last-child,
    .probability-table td:last-child {
        text-align: right;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        color: var(--text);
        font-weight: 550;
    }

    /* Captions */
    .stCaption,
    div[data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    /* Divider */
    hr {
        border-color: #dce7eb !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }

    /* Transparent Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }

        h1 {
            font-size: 1.85rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def get_model_runtime():
    return load_model(MODEL_PATH)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def threshold_as_percent(value: float) -> float:
    value = float(value)

    if value <= 1.0:
        return value * 100.0

    return value


def confidence_status(confidence: float) -> str:
    high_threshold = threshold_as_percent(
        HIGH_CONFIDENCE_THRESHOLD
    )

    medium_threshold = threshold_as_percent(
        MEDIUM_CONFIDENCE_THRESHOLD
    )

    if confidence >= high_threshold:
        return "High Confidence"

    if confidence >= medium_threshold:
        return "Medium Confidence"

    return "Low Confidence"


def read_input_image(
    file_object,
) -> tuple[Image.Image, bytes]:
    raw_bytes = file_object.getvalue()

    try:
        image = Image.open(
            BytesIO(raw_bytes)
        ).convert("RGB")

        image.load()

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "The selected file is not a readable JPG, JPEG, or PNG image."
        ) from error

    if image.width < 32 or image.height < 32:
        raise ValueError(
            "The selected image is too small to analyze."
        )

    return image, raw_bytes


def image_to_png_bytes(
    image_rgb: np.ndarray,
) -> bytes:
    buffer = BytesIO()

    Image.fromarray(
        image_rgb
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def is_tumor_result(
    result: dict,
) -> bool:
    model_class = str(
        result.get(
            "model_class",
            "",
        )
    ).strip().lower()

    display_class = str(
        result.get(
            "display_class",
            "",
        )
    ).strip().lower()

    no_tumor_labels = {
        "healthy",
        "no tumor",
        "no_tumor",
        "notumor",
        "no-tumor",
    }

    return (
        model_class not in no_tumor_labels
        and display_class not in no_tumor_labels
    )


# =========================================================
# ANALYSIS PIPELINE
# =========================================================

def run_analysis(
    image: Image.Image,
    raw_bytes: bytes,
    source_filename: str,
    model,
    device,
) -> dict:
    image_rgb = np.asarray(
        image,
        dtype=np.uint8,
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    result = predict(
        model,
        device,
        image_bgr,
    )

    cam = generate_notebook_cam(
        model=model,
        input_tensor=result["input_tensor"],
        target_class_index=result["predicted_index"],
    )

    visuals = create_visualizations(
        input_tensor=result["input_tensor"],
        cam=cam,
        create_tumor_box=is_tumor_result(result),
    )

    status = confidence_status(
        result["confidence"]
    )

    report_bytes = generate_report_bytes(
        original_rgb=visuals["original_rgb"],
        heatmap_rgb=visuals["heatmap_rgb"],
        bbox_rgb=visuals["bbox_rgb"],
        combined_rgb=visuals["combined_rgb"],
        prediction=result["display_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_time_ms=result["inference_time_ms"],
        model_name=MODEL_NAME,
        confidence_status=status,
    )

    try:
        save_prediction(
            image_rgb=image_rgb,
            original_filename=source_filename,
            result=result,
        )

    except Exception:
        # A history-saving issue must not stop prediction.
        pass

    return {
        "source_hash": hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        "source_filename": source_filename,
        "result": result,
        "status": status,
        "visualizations": visuals,
        "report_bytes": report_bytes,
    }


# =========================================================
# MAIN INTERFACE
# =========================================================

def render_home(
    model,
    device,
) -> None:
    st.title(
        "🧠 Hybrid Swin-UNet Brain MRI System"
    )

    left_panel, right_panel = st.columns(
        [
            0.92,
            1.08,
        ],
        gap="large",
    )

    selected_file = None
    filename = "camera_capture.png"
    image = None
    raw_bytes = None
    current_hash = None
    analyze_clicked = False

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with left_panel:
        st.subheader("MRI Input")

        input_method = st.radio(
            "Input method",
            (
                "Upload MRI",
                "Camera",
            ),
            horizontal=True,
        )

        if input_method == "Upload MRI":
            selected_file = st.file_uploader(
                "Upload MRI",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],
                label_visibility="collapsed",
            )

            if selected_file is not None:
                filename = selected_file.name

        else:
            selected_file = st.camera_input(
                "Capture an MRI film or printed MRI scan"
            )

        if selected_file is not None:
            try:
                image, raw_bytes = read_input_image(
                    selected_file
                )

                current_hash = hashlib.sha256(
                    raw_bytes
                ).hexdigest()

            except ValueError as error:
                st.error(str(error))

        if image is not None:
            preview_column, action_column = st.columns(
                [
                    1.35,
                    0.65,
                ],
                vertical_alignment="center",
            )

            with preview_column:
                st.image(
                    image,
                    caption=filename,
                    width=310,
                )

            with action_column:
                st.markdown(
                    "<div style='height:38px'></div>",
                    unsafe_allow_html=True,
                )

                analyze_clicked = st.button(
                    "🔍 Analyze MRI",
                    type="primary",
                    use_container_width=True,
                )

                st.caption(
                    f"{image.width} × {image.height} px"
                )

                st.caption(
                    f"Model: {MODEL_NAME}"
                )

        else:
            st.info(
                "Upload or capture an MRI image to begin."
            )

    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    if (
        analyze_clicked
        and image is not None
        and raw_bytes is not None
    ):
        with st.spinner(
            "Running model inference and CAM generation..."
        ):
            try:
                st.session_state.analysis_result = (
                    run_analysis(
                        image=image,
                        raw_bytes=raw_bytes,
                        source_filename=filename,
                        model=model,
                        device=device,
                    )
                )

            except Exception as error:
                st.exception(error)

    analysis = st.session_state.analysis_result

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with right_panel:
        st.subheader("Results Panel")

        if analysis is None:
            st.info(
                "Upload an MRI and click **Analyze MRI**."
            )

        elif (
            current_hash is None
            or analysis["source_hash"] != current_hash
        ):
            st.info(
                "Click **Analyze MRI** to generate results for this image."
            )

        else:
            result = analysis["result"]

            st.success(
                f'Prediction: {result["display_class"]}'
            )

            metric_column_1, metric_column_2 = (
                st.columns(2)
            )

            metric_column_1.metric(
                "Confidence",
                f'{result["confidence"]:.2f}%',
            )

            metric_column_2.metric(
                "Inference time",
                f'{result["inference_time_ms"]:.2f} ms',
            )

            st.caption(
                analysis["status"]
            )

            st.markdown(
                "#### Class Probabilities"
            )

            probability_frame = pd.DataFrame(
                {
                    "Class": list(
                        result["probabilities"].keys()
                    ),
                    "Probability (%)": list(
                        result["probabilities"].values()
                    ),
                }
            )

            probability_frame[
                "Probability (%)"
            ] = probability_frame[
                "Probability (%)"
            ].round(2)

            # Stable HTML table instead of interactive st.dataframe.
            # This removes the Hugging Face blinking/vibration.
            probability_html = probability_frame.to_html(
                index=False,
                classes="probability-table",
                border=0,
                formatters={
                    "Probability (%)": (
                        lambda value: f"{value:.2f}"
                    )
                },
            )

            st.markdown(
                probability_html,
                unsafe_allow_html=True,
            )

            st.download_button(
                "⬇ Download PDF Report",
                data=analysis["report_bytes"],
                file_name=(
                    "brain_tumor_analysis_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )

    # =====================================================
    # SINGLE EXPLAINABILITY SECTION
    # =====================================================

    analysis = st.session_state.analysis_result

    if (
        analysis is not None
        and current_hash is not None
        and analysis["source_hash"] == current_hash
    ):
        result = analysis["result"]
        visuals = analysis["visualizations"]

        st.divider()

        st.subheader(
            "Explainability and Localization"
        )

        image_columns = st.columns(
            3,
            gap="medium",
        )

        with image_columns[0]:
            st.image(
                visuals["original_rgb"],
                caption="Original MRI",
                use_container_width=True,
            )

        with image_columns[1]:
            st.image(
                visuals["heatmap_rgb"],
                caption="CAM Heatmap",
                use_container_width=True,
            )

        with image_columns[2]:
            if is_tumor_result(result):
                localization_caption = (
                    "CAM-based Localization"
                )
            else:
                localization_caption = (
                    "No Tumor — No Bounding Box"
                )

            st.image(
                visuals["combined_rgb"],
                caption=localization_caption,
                use_container_width=True,
            )

        download_columns = st.columns(3)

        download_columns[0].download_button(
            "Original PNG",
            data=image_to_png_bytes(
                visuals["original_rgb"]
            ),
            file_name="original_mri.png",
            mime="image/png",
            use_container_width=True,
        )

        download_columns[1].download_button(
            "CAM PNG",
            data=image_to_png_bytes(
                visuals["heatmap_rgb"]
            ),
            file_name="cam_heatmap.png",
            mime="image/png",
            use_container_width=True,
        )

        download_columns[2].download_button(
            "Overlay PNG",
            data=image_to_png_bytes(
                visuals["combined_rgb"]
            ),
            file_name=(
                "cam_localization_overlay.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

        st.caption(
            "The CAM and bounding box are explainability-based "
            "localization estimates and are not pixel-level "
            "tumor segmentation."
        )


# =========================================================
# APPLICATION START
# =========================================================

def main() -> None:
    try:
        model, device = get_model_runtime()

    except Exception as error:
        st.error(
            "The model could not be loaded."
        )
        st.exception(error)
        st.stop()

    render_home(
        model,
        device,
    )


if __name__ == "__main__":
    main()