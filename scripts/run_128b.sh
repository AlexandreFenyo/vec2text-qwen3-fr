#!/bin/sh
# Fallback 128-token chain after the inverter128 crash: use its step-1000 checkpoint as is.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus128
N=160000
echo "== hypotheses $(date)"
$PY scripts/gen_hypotheses.py --inverter ckpt/inverter128 --data $D --splits val,train --limit $N --bs 768 > logs/gen_hyp128.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }
echo "== corrector128 $(date)"
$PY -m v2t.train --stage corrector --init ckpt/corrector --max_tokens 128 --data $D --out ckpt/corrector128 --limit $N \
  --bs 128 --micro_bs 8 --epochs 1 --lr 3e-5 --lr_proj 2e-4 --warmup 50 --eval_every 300 --save_every 300 \
  --compile --pad_to 16 > logs/train_corrector128.log 2>&1 || { echo "FAILED corrector"; exit 1; }
echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128 --data $D --n 150 --configs 0x1,0x4,1x4,3x4 --show 2 --out logs/eval128_150.json > logs/eval128_150.log 2>&1
echo "== demo $(date)"
$PY invert.py --vectors examples/vecs128.tsv --metadata examples/meta128.tsv --steps 3 --beam 4 --out examples/result128.tsv > logs/demo128.log 2>&1
echo "== all done $(date)"
