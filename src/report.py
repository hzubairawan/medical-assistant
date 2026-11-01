"""
report.py
=========
Builds a professional PDF report of one assessment with ReportLab.

Note for beginners:
    ReportLab's built-in fonts only support Western characters. If the guidance
    is in Urdu / Hindi / Arabic we cannot draw it in a PDF without bundling a
    Unicode font, so we raise PdfUnsupported and the app offers the Markdown
    download instead (Markdown keeps every language).
"""

import io
import re
from xml.sax.saxutils import escape

from src import config


class PdfUnsupported(Exception):
    """Raised when the PDF cannot be created (missing library or script)."""


def _latin(text) -> str:
    """Keep only characters the standard PDF fonts can draw (drops emoji)."""
    return str(text or "").encode("cp1252", errors="ignore").decode("cp1252")


def _inline(text) -> str:
    """Escape XML and convert **bold** Markdown into ReportLab <b> tags."""
    text = escape(_latin(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("*", "").replace("`", "")


def _check_supported(entry: dict):
    """Refuse to build a PDF if too much text would be lost (non-Latin scripts)."""
    result = entry["result"]
    parts = [result["summary"], entry.get("narrative") or ""]
    parts += result["recommended_next_steps"] + result["warning_signs"]
    original = "".join(parts)
    kept = _latin(original)
    visible = len(re.sub(r"\s", "", original))
    if visible and len(re.sub(r"\s", "", kept)) / visible < 0.85:
        raise PdfUnsupported(
            "PDF export supports Latin-script languages only. "
            "Use the Markdown download for this language."
        )


def build_pdf_report(entry: dict) -> bytes:
    """Return the PDF file as bytes."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise PdfUnsupported("Install ReportLab to enable PDF export (pip install reportlab).") from exc

    _check_supported(entry)

    inputs, result = entry["inputs"], entry["result"]
    level = result["urgency_level"]
    teal = colors.HexColor("#0f766e")

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("t", parent=base["Title"], textColor=teal, fontSize=21, spaceAfter=2, alignment=0),
        "meta": ParagraphStyle("m", parent=base["Normal"], textColor=colors.HexColor("#6b7280"), fontSize=8.5, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=teal, fontSize=12.5, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=3),
        "bullet": ParagraphStyle("bl", parent=base["BodyText"], fontSize=9.5, leading=13.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "box": ParagraphStyle("box", parent=base["BodyText"], fontSize=8.8, leading=12, textColor=colors.HexColor("#7f1d1d")),
        "urgency": ParagraphStyle(
            "u", parent=base["BodyText"], fontSize=13, leading=17, fontName="Helvetica-Bold",
            textColor=colors.HexColor(config.URGENCY_TEXT_COLORS[level]),
        ),
    }

    def bullets(items):
        return [Paragraph(_inline(item), styles["bullet"], bulletText="-") for item in items] or [
            Paragraph("(none)", styles["body"])
        ]

    def boxed(paragraph, background, border):
        table = Table([[paragraph]], colWidths=[174 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return table

    story = [
        Paragraph("MediGuide AI - Guidance Report", styles["title"]),
        Paragraph(f"Generated {entry['timestamp']} | Model: {escape(str(entry['model']))} | Educational use only", styles["meta"]),
        boxed(Paragraph(_inline(config.DISCLAIMER_PLAIN), styles["box"]), "#fdecea", "#c62828"),
        Spacer(1, 8),
        boxed(
            Paragraph(f"URGENCY LEVEL: {level} - {_inline(config.URGENCY_MEANINGS[level])}", styles["urgency"]),
            config.URGENCY_COLORS[level], config.URGENCY_COLORS[level],
        ),
        Paragraph("Patient information", styles["h2"]),
    ]

    rows = [
        ("Age / gender", f"{inputs['age']} / {inputs['gender']}"),
        ("Symptoms", inputs["symptoms"]),
        ("Duration", inputs["duration"]),
        ("Severity (1-10)", str(inputs["severity"])),
        ("Existing conditions", inputs["conditions"]),
        ("Medications", inputs["medications"]),
        ("Notes", inputs["notes"]),
    ]
    info = Table(
        [[Paragraph(f"<b>{escape(label)}</b>", styles["body"]), Paragraph(_inline(value), styles["body"])] for label, value in rows],
        colWidths=[42 * mm, 132 * mm],
    )
    info.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0fdfa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [info, Paragraph("Summary", styles["h2"]), Paragraph(_inline(result["summary"]), styles["body"])]

    story.append(Paragraph("Possible conditions (education only - NOT a diagnosis)", styles["h2"]))
    for condition in result["possible_conditions"]:
        story.append(Paragraph(
            f"<b>{_inline(condition['name'])}</b> - {_inline(condition['reason'])}", styles["bullet"], bulletText="-",
        ))
    if not result["possible_conditions"]:
        story.append(Paragraph("(none)", styles["body"]))

    story.append(Paragraph("Recommended next steps", styles["h2"]))
    story += bullets(result["recommended_next_steps"])
    story.append(Paragraph("Questions to ask a healthcare professional", styles["h2"]))
    story += bullets(result["questions_for_doctor"])
    story.append(Paragraph("Warning signs - get urgent help if you notice these", styles["h2"]))
    story += bullets(result["warning_signs"])

    narrative = entry.get("narrative")
    if narrative:
        story.append(Paragraph("General information", styles["h2"]))
        for line in narrative.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                story.append(Paragraph(f"<b>{_inline(line.lstrip('# '))}</b>", styles["body"]))
            elif re.match(r"^[-*•]\s+", line):
                story.append(Paragraph(_inline(re.sub(r"^[-*•]\s+", "", line)), styles["bullet"], bulletText="-"))
            else:
                story.append(Paragraph(_inline(line), styles["body"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(18 * mm, 10 * mm, "Educational AI prototype - not medical advice. Consult a qualified healthcare professional.")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm, title="MediGuide AI Guidance Report",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
