#!/bin/sh
# Second training round for the 128-token models (continue from ckpt/*128 with a lower LR), into ckpt/*128_r2.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus128
NHYP=400000
until ! pgrep -f "[e]val.py" >/dev/null; do sleep 20; done
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }
echo "== inverter128_r2 $(date)"
retry 4 $PY -m v2t.train --stage inversion --init ckpt/inverter128 --resume --max_tokens 128 --data $D --out ckpt/inverter128_r2 \
  --bs 128 --micro_bs 16 --epochs 1 --lr 2e-5 --lr_proj 1e-4 --warmup 100 --eval_every 500 --save_every 500 --seed 1 \
  --compile --pad_to 16 >> logs/train_inverter128_r2.log 2>&1 || { echo "FAILED inverter"; exit 1; }
echo "== hypotheses $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter128_r2 --data $D --splits val,train --limit $NHYP --bs 768 \
  >> logs/gen_hyp128_r2.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }
echo "== corrector128_r2 $(date)"
retry 4 $PY -m v2t.train --stage corrector --init ckpt/corrector128 --resume --max_tokens 128 --data $D --out ckpt/corrector128_r2 --limit $NHYP \
  --bs 128 --micro_bs 8 --epochs 1 --lr 2e-5 --lr_proj 1e-4 --warmup 100 --eval_every 500 --save_every 500 --seed 1 \
  --compile --pad_to 16 >> logs/train_corrector128_r2.log 2>&1 || { echo "FAILED corrector"; exit 1; }
echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128_r2 --corrector ckpt/corrector128_r2 --data $D --n 200 --configs 0x4,3x4,5x4 --show 2 --no_repeat 6 --out logs/eval128_r2_200.json > logs/eval128_r2_200.log 2>&1
echo "== all done $(date)"
