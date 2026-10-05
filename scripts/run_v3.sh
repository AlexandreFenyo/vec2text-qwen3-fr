#!/bin/sh
# Corrector v3: the v1 corrector (best so far) adapted to inverter_v2 hypotheses on passages never seen by inverter_v2.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }

until grep -q "done" logs/fresh_wiki.log && grep -q "done" logs/fresh_web.log && grep -q "done" logs/fresh_domain.log; do sleep 60; done
echo "== make_corpus_corr3 $(date)"
$PY scripts/make_corpus_corr3.py >> logs/make_corpus_corr3.log 2>&1 || { echo "FAILED corpus"; exit 1; }

echo "== hypotheses $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter_v2 --data data/corpus_corr3 --splits val,train --bs 2048 \
  >> logs/gen_hyp_corr3.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }

echo "== corrector_v3 $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/corrector --resume --max_tokens 32 --data data/corpus_corr3 --out ckpt/corrector_v3 \
  --bs 128 --micro_bs 32 --epochs 2 --lr 3e-5 --lr_proj 3e-4 --warmup 300 --eval_every 2000 --save_every 2000 --seed 2 \
  --compile >> logs/train_corrector_v3.log 2>&1 || { echo "FAILED corrector"; exit 1; }

echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter_v2 --corrector ckpt/corrector_v3 --data data/corpus_v2 --n 300 --configs 1x4,5x4,10x8 --show 3 --out logs/eval_v3_300.json > logs/eval_v3_300.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter_v2 --corrector ckpt/corrector_v3 --data data/corpus_v2 --split test_domain --n 300 --configs 1x4,5x4,10x8 --show 3 --out logs/eval_v3_domain_300.json > logs/eval_v3_domain_300.log 2>&1
echo "== all done $(date)"
