#!/bin/sh
# Wait for the domain corpus builders, then assemble corpus_v2 / corpus_corr (embeds the domain passages).
cd "$(dirname "$0")/.."
until grep -q "done" logs/domain_wiki.log && grep -q "done" logs/domain_web.log; do sleep 60; done
echo "== make_corpus_v2 $(date)"
.venv/bin/python scripts/make_corpus_v2.py >> logs/make_corpus_v2.log 2>&1 && echo "== prep done $(date)" || echo "FAILED make_corpus_v2"
