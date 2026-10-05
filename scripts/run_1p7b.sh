#!/bin/sh
# 128-token pipeline with Qwen3-1.7B-Base as decoder (frozen embeddings, 8-bit AdamW), with retries/resume.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus128
NHYP=300000
BASE=Qwen/Qwen3-1.7B-Base
MB_INV=${MB_INV:-8}
MB_COR=${MB_COR:-4}
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }

echo "== inverter128_1p7b $(date)"
retry 5 $PY -m v2t.train --stage inversion --base $BASE --resume --max_tokens 128 --data $D --out ckpt/inverter128_1p7b \
  --bs 128 --micro_bs $MB_INV --epochs 1 --lr 3e-5 --lr_proj 5e-4 --warmup 200 --eval_every 500 --save_every 500 \
  --freeze_embed --optim adamw8bit --compile --pad_to 16 >> logs/train_inverter128_1p7b.log 2>&1 || { echo "FAILED inverter"; exit 1; }

echo "== hypotheses $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter128_1p7b --data $D --splits val,train --limit $NHYP --bs 384 \
  >> logs/gen_hyp128_1p7b.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }

echo "== corrector128_1p7b $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/inverter128_1p7b --resume --max_tokens 128 --data $D --out ckpt/corrector128_1p7b --limit $NHYP \
  --bs 128 --micro_bs $MB_COR --epochs 1 --lr 3e-5 --lr_proj 5e-4 --warmup 200 --eval_every 500 --save_every 500 \
  --freeze_embed --optim adamw8bit --compile --pad_to 16 >> logs/train_corrector128_1p7b.log 2>&1 || { echo "FAILED corrector"; exit 1; }

echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128_1p7b --corrector ckpt/corrector128_1p7b --data $D --n 200 --configs 0x4,1x4,3x4,5x4 --show 2 --no_repeat 6 --out logs/eval128_1p7b_200.json > logs/eval128_1p7b_200.log 2>&1
echo "== demo $(date)"
$PY invert.py --vectors examples/vecs128.tsv --metadata examples/meta128.tsv --steps 5 --beam 4 --inverter ckpt/inverter128_1p7b --corrector ckpt/corrector128_1p7b --out examples/result128_1p7b.tsv > logs/demo128_1p7b.log 2>&1
echo "== all done $(date)"
