"""Convierte documento_tecnico.md -> HTML estilizado -> PDF (Chrome headless)."""
import markdown
import pathlib
import subprocess
import shutil

here = pathlib.Path(__file__).parent
md_text = (here / "documento_tecnico.md").read_text(encoding="utf-8")

body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
       font-size: 11pt; line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 22pt; color: #1a3a5c; margin: 0 0 4px; }
h1 + h2 { margin-top: 0; color: #2c6b9e; border: none; font-size: 15pt; }
h2 { font-size: 15pt; color: #1a3a5c; border-bottom: 2px solid #2c6b9e;
     padding-bottom: 4px; margin-top: 26px; }
h3 { font-size: 12.5pt; color: #2c6b9e; margin-top: 18px; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.8pt; }
th, td { border: 1px solid #b8c4d0; padding: 5px 8px; text-align: left; }
th { background: #2c6b9e; color: #fff; }
tr:nth-child(even) td { background: #f0f5fa; }
img { max-width: 84%; display: block; margin: 12px auto; border: 1px solid #ddd; }
blockquote { border-left: 4px solid #f0ad4e; background: #fdf7ec; margin: 14px 0;
             padding: 8px 14px; color: #5a4a2a; }
code { background: #eef1f4; padding: 1px 5px; border-radius: 3px;
       font-family: 'Consolas', monospace; font-size: 9.5pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 18px 0; }
.pagebreak { page-break-before: always; }
strong { color: #1a3a5c; }
"""

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<style>{CSS}</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body>{body}</body></html>"""

html_path = here / "_documento_tecnico.html"
html_path.write_text(html, encoding="utf-8")

chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
pdf_path = here / "documento_tecnico.pdf"
subprocess.run([
    chrome, "--headless", "--no-sandbox", "--disable-gpu",
    "--virtual-time-budget=12000", "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
    html_path.as_uri(),
], check=True, capture_output=True)
print("PDF generado:", pdf_path, pdf_path.stat().st_size, "bytes")
