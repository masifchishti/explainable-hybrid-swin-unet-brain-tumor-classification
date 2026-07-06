from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Mapping

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _image_buffer(image_rgb: np.ndarray) -> BytesIO:
    """Convert an RGB NumPy image into an in-memory PNG."""
    buffer = BytesIO()
    PILImage.fromarray(image_rgb.astype(np.uint8)).save(
        buffer,
        format="PNG",
    )
    buffer.seek(0)
    return buffer


def _report_image(
    image_rgb: np.ndarray,
    width_mm: float = 68,
    height_mm: float = 68,
) -> ReportImage:
    """Create a fixed-size ReportLab image."""
    return ReportImage(
        _image_buffer(image_rgb),
        width=width_mm * mm,
        height=height_mm * mm,
    )


def generate_report_bytes(
    *,
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    bbox_rgb: np.ndarray,
    combined_rgb: np.ndarray,
    prediction: str,
    confidence: float,
    probabilities: Mapping[str, float],
    inference_time_ms: float,
    model_name: str,
    confidence_status: str,
) -> bytes:
    """
    Generate a stable PDF report in memory.

    The layout intentionally avoids KeepTogether inside table cells because
    that combination can produce extremely large flowable heights in some
    ReportLab versions.
    """
    output = BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Brain Tumor AI Analysis Report",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MedicalTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0B6477"),
        fontSize=18,
        leading=22,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0B6477"),
        fontSize=12,
        leading=15,
        spaceBefore=4,
        spaceAfter=6,
    )

    caption_style = ParagraphStyle(
        "ImageCaption",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )

    note_style = ParagraphStyle(
        "ResearchNotice",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4B5563"),
    )

    story = [
        Paragraph("Brain Tumor AI Analysis Report", title_style),
        Spacer(1, 4 * mm),
    ]

    summary_data = [
        ["Field", "Result"],
        ["Prediction", prediction],
        ["Confidence", f"{confidence:.2f}%"],
        ["Confidence status", confidence_status],
        ["Inference time", f"{inference_time_ms:.2f} ms"],
        ["Model", model_name],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[46 * mm, 116 * mm],
        repeatRows=1,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6477")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E6F4F7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.extend(
        [
            Paragraph("Analysis summary", section_style),
            summary_table,
            Spacer(1, 5 * mm),
            Paragraph("Class probabilities", section_style),
        ]
    )

    probability_data = [["Class", "Probability"]]
    probability_data.extend(
        [
            [class_name, f"{float(value):.2f}%"]
            for class_name, value in probabilities.items()
        ]
    )

    probability_table = Table(
        probability_data,
        colWidths=[96 * mm, 66 * mm],
        repeatRows=1,
    )
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.extend(
        [
            probability_table,
            PageBreak(),
            Paragraph("Visual analysis", section_style),
        ]
    )

    # Directly place images and captions in table cells.
    # No KeepTogether objects are used.
    image_table = Table(
        [
            [
                _report_image(original_rgb),
                _report_image(heatmap_rgb),
            ],
            [
                Paragraph("Original MRI", caption_style),
                Paragraph("CAM heatmap", caption_style),
            ],
            [
                _report_image(bbox_rgb),
                _report_image(combined_rgb),
            ],
            [
                Paragraph("Localization box", caption_style),
                Paragraph("Overlay and localization", caption_style),
            ],
        ],
        colWidths=[82 * mm, 82 * mm],
        rowHeights=[
            70 * mm,
            8 * mm,
            70 * mm,
            8 * mm,
        ],
    )

    image_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    story.extend(
        [
            image_table,
            Spacer(1, 4 * mm),
            Paragraph(
                "Research-use notice: this software is an AI research "
                "demonstration and is not a substitute for diagnosis by "
                "a qualified medical professional.",
                note_style,
            ),
        ]
    )

    document.build(story)
    return output.getvalue()
