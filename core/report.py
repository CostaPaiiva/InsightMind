from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from ydata_profiling import ProfileReport

def build_html_report(df: pd.DataFrame, quality_metrics: dict, insights: list[str], include_profiling: bool = True) -> bytes:
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>Relatório InsightMind</title></head><body>")
    parts.append("<h1>Relatório InsightMind</h1>")

    parts.append("<h2>Métricas de Qualidade</h2>")
    parts.append("<pre>" + _escape_html(str(quality_metrics)) + "</pre>")

    parts.append("<h2>Insights</h2><ul>")
    for x in insights:
        parts.append("<li>" + _escape_html(x) + "</li>")
    parts.append("</ul>")

    if include_profiling:
        profile = ProfileReport(df, title="Profiling do Dataset", minimal=True)
        parts.append("<hr/>")
        parts.append(profile.to_html())

    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")

def build_pdf_report(df: pd.DataFrame, quality_metrics: dict, insights: list[str], figs_png: list[bytes]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "Relatório InsightMind")
    y -= 28

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Métricas de Qualidade")
    y -= 16

    c.setFont("Helvetica", 9)
    y = _draw_multiline(c, str(quality_metrics), 40, y, max_width=520, line_height=11, bottom_margin=80)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Insights")
    y -= 16

    c.setFont("Helvetica", 10)
    for item in insights[:20]:
        y = _draw_multiline(c, "• " + item, 40, y, max_width=520, line_height=12, bottom_margin=80)
        if y < 120:
            c.showPage()
            y = h - 60

    for png in figs_png:
        c.showPage()
        y = h - 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Gráfico")
        y -= 20
        img = ImageReader(BytesIO(png))
        c.drawImage(img, 40, 120, width=520, height=520, preserveAspectRatio=True, anchor='c')

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf

def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _draw_multiline(c, text: str, x: int, y: int, max_width: int, line_height: int, bottom_margin: int) -> int:
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", c._fontsize) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w
            if y < bottom_margin:
                c.showPage()
                y = A4[1] - 60
                c.setFont("Helvetica", c._fontsize)
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y
