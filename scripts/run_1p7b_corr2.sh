#!/bin/sh
# Continue the 1.7B corrector: hypotheses for the whole 600k corpus, then one more epoch on 600k (warm from corrector128_1p7b).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus128
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }

if [ "$(wc -l < $D/train.hyp.jsonl)" -lt 600000 ]; then
echo "== hypotheses 600k $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter128_1p7b --data $D --splits train --start 300288 --limit 600000 --bs 384 \
  >> logs/gen_hyp128_1p7b_600k.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }
fi

echo "== corrector128_1p7b_v2 $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/corrector128_1p7b --resume --max_tokens 128 --data $D --out ckpt/corrector128_1p7b_v2 \
  --bs 128 --micro_bs 4 --epochs 1 --lr 2e-5 --lr_proj 2e-4 --warmup 100 --eval_every 500 --eval_bs 16 --save_every 500 --seed 1 \
  --freeze_embed --optim adamw8bit --compile --pad_to 16 >> logs/train_corrector128_1p7b_v2.log 2>&1 || { echo "FAILED corrector"; exit 1; }

echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128_1p7b --corrector ckpt/corrector128_1p7b_v2 --data $D --n 200 --configs 0x4,1x4,3x4,5x4 --show 2 --no_repeat 6 --out logs/eval128_1p7b_v2_200.json > logs/eval128_1p7b_v2_200.log 2>&1
echo "== all done $(date)"
