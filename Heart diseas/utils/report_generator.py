"""
report_generator.py
====================
Generates a downloadable PDF prediction report using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor


# ── Brand colours ─────────────────────────────────────────────────────────────
PRIMARY    = HexColor("#6C63FF")
SUCCESS    = HexColor("#00D4AA")
DANGER     = HexColor("#FF4B6E")
WARNING    = HexColor("#FFB347")
DARK_BG    = HexColor("#0E1117")
CARD_BG    = HexColor("#1E2130")
LIGHT_TEXT = HexColor("#E8EAED")
SUBTEXT    = HexColor("#9AA0B4")
WHITE      = colors.white
BLACK      = colors.black


def _build_styles():
    """Create a set of custom paragraph styles."""
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title", parent=styles["Title"],
            fontSize=24, textColor=WHITE, alignment=TA_CENTER,
            spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=styles["Normal"],
            fontSize=12, textColor=SUBTEXT, alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "section_header": ParagraphStyle(
            "section_header", parent=styles["Heading2"],
            fontSize=14, textColor=PRIMARY, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=styles["Normal"],
            fontSize=10, textColor=LIGHT_TEXT, spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "label", parent=styles["Normal"],
            fontSize=9, textColor=SUBTEXT,
        ),
        "risk_high": ParagraphStyle(
            "risk_high", parent=styles["Normal"],
            fontSize=20, textColor=DANGER, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4,
        ),
        "risk_mod": ParagraphStyle(
            "risk_mod", parent=styles["Normal"],
            fontSize=20, textColor=WARNING, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4,
        ),
        "risk_low": ParagraphStyle(
            "risk_low", parent=styles["Normal"],
            fontSize=20, textColor=SUCCESS, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer", parent=styles["Normal"],
            fontSize=8, textColor=SUBTEXT, alignment=TA_CENTER,
        ),
    }
    return custom


def generate_pdf_report(
    patient_data: dict,
    prediction_result: dict,
    model_metrics: dict,
    feature_names: list,
) -> bytes:
    """
    Generate a PDF prediction report.

    Parameters
    ----------
    patient_data      : {feature_name: value}
    prediction_result : {
        "probability": float,
        "prediction":  int (0 or 1),
        "risk_level":  str ("Low" | "Moderate" | "High"),
        "confidence":  float,
        "model_name":  str,
    }
    model_metrics     : {metric_name: value}
    feature_names     : list of feature names

    Returns
    -------
    bytes — PDF content
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = _build_styles()
    story  = []
    W = A4[0] - 4*cm  # usable width

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("❤️  Heart Disease Prediction Report", styles["title"]))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=10))

    # ── Prediction Summary Box ────────────────────────────────────────────────
    prob  = prediction_result.get("probability", 0)
    risk  = prediction_result.get("risk_level", "Unknown")
    conf  = prediction_result.get("confidence", 0)
    model = prediction_result.get("model_name", "N/A")

    risk_style_map = {"High": "risk_high", "Moderate": "risk_mod", "Low": "risk_low"}
    risk_style = risk_style_map.get(risk, "risk_mod")
    risk_color = {"High": DANGER, "Moderate": WARNING, "Low": SUCCESS}.get(risk, PRIMARY)

    summary_data = [
        [Paragraph("PREDICTION RESULT", styles["label"]),
         Paragraph("PROBABILITY", styles["label"]),
         Paragraph("CONFIDENCE", styles["label"]),
         Paragraph("MODEL USED", styles["label"])],
        [Paragraph(risk, styles[risk_style]),
         Paragraph(f"{prob*100:.1f}%", styles[risk_style]),
         Paragraph(f"{conf*100:.1f}%", styles[risk_style]),
         Paragraph(model, styles["body"])],
    ]

    summary_table = Table(summary_data, colWidths=[W/4]*4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CARD_BG),
        ("TEXTCOLOR",     (0,0), (-1,-1), LIGHT_TEXT),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS",(0,0), (-1,-1), [6,6,6,6]),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [CARD_BG, CARD_BG]),
        ("GRID",          (0,0), (-1,-1), 0.5, HexColor("#2a2d3e")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # ── Patient Information ────────────────────────────────────────────────────
    story.append(Paragraph("Patient Input Parameters", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SUBTEXT, spaceAfter=6))

    feature_display = {
        "age":      "Age (years)",
        "sex":      "Sex (1=Male, 0=Female)",
        "cp":       "Chest Pain Type",
        "trestbps": "Resting BP (mm Hg)",
        "chol":     "Cholesterol (mg/dl)",
        "fbs":      "Fasting Blood Sugar > 120",
        "restecg":  "Resting ECG",
        "thalach":  "Max Heart Rate",
        "exang":    "Exercise Angina",
        "oldpeak":  "ST Depression (Oldpeak)",
        "slope":    "ST Slope",
        "ca":       "Major Vessels (CA)",
        "thal":     "Thalassemia",
    }

    patient_rows = [
        [Paragraph("Feature", styles["label"]), Paragraph("Value", styles["label"])]
    ]
    for feat, val in patient_data.items():
        display_name = feature_display.get(feat, feat.title())
        patient_rows.append([
            Paragraph(display_name, styles["body"]),
            Paragraph(str(val), styles["body"]),
        ])

    patient_table = Table(patient_rows, colWidths=[W*0.6, W*0.4])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("BACKGROUND",    (0, 1), (-1, -1), CARD_BG),
        ("TEXTCOLOR",     (0, 1), (-1, -1), LIGHT_TEXT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CARD_BG, HexColor("#252839")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#2a2d3e")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 16))

    # ── Model Performance ─────────────────────────────────────────────────────
    story.append(Paragraph("Model Performance Metrics", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SUBTEXT, spaceAfter=6))

    metric_rows = [
        [Paragraph("Metric", styles["label"]),
         Paragraph("Score", styles["label"])]
    ]
    for metric, value in model_metrics.items():
        display_val = f"{value:.4f}" if isinstance(value, float) else str(value)
        metric_rows.append([
            Paragraph(metric, styles["body"]),
            Paragraph(display_val, styles["body"]),
        ])

    metric_table = Table(metric_rows, colWidths=[W*0.6, W*0.4])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("BACKGROUND",    (0, 1), (-1, -1), CARD_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CARD_BG, HexColor("#252839")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#2a2d3e")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 1), (-1, -1), LIGHT_TEXT),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 20))

    # ── Clinical Disclaimer ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=SUBTEXT, spaceAfter=8))
    disclaimer = (
        "<b>⚠️ Clinical Disclaimer:</b> This prediction is generated by a machine "
        "learning model for informational purposes only. It should NOT be used as a "
        "substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )
    story.append(Paragraph(disclaimer, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Heart Disease Prediction System  •  AI-Powered Healthcare Analytics  •  "
        f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}",
        styles["footer"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
