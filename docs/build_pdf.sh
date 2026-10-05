#!/bin/sh
# Markdown -> HTML (pandoc) -> PDF (WeasyPrint), images PNG dans img/
cd "$(dirname "$0")"
pandoc principe.md -f gfm -t html5 -s --metadata title="Inversion d'embeddings et protection des vecteurs d'un RAG de santé" --metadata lang=fr -o principe.html
../.venv/bin/python - <<'PY'
import re
s = open("principe.html", encoding="utf-8").read()
s = re.sub(r'<header id="title-block-header">.*?</header>', '', s, flags=re.S)
open("principe.html", "w", encoding="utf-8").write(s)
from weasyprint import HTML, CSS
HTML("principe.html", base_url=".").write_pdf("principe.pdf", stylesheets=[CSS("print.css")])
print("principe.pdf written")
PY
