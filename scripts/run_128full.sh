#!/bin/sh
# Full-quality 128-token pipeline with automatic retries/resume after a crash (e.g. GPU OOM caused by Windows).
# Starts once the fallback chain (run_128b.sh) is finished. Warm-starts from the fallback checkpoints.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus128
NHYP=400000

until grep -q -E "== all done|FAILED" logs/run_128b.log || ! pgrep -f "[r]un_128b.sh" >/dev/null; do sleep 30; done
echo "== start $(date)"
rm -rf ckpt/inverter128_fallback ckpt/corrector128_fallback
cp -r ckpt/inverter128 ckpt/inverter128_fallback
[ -d ckpt/corrector128 ] && cp -r ckpt/corrector128 ckpt/corrector128_fallback
rm -f ckpt/inverter128/train_state.json ckpt/inverter128/optim.pt ckpt/corrector128/train_state.json ckpt/corrector128/optim.pt

retry() {  # retry <n> <cmd...> : rerun on failure (training commands use --resume so they continue)
  n=$1; shift
  i=0
  while [ $i -lt $n ]; do
    "$@" && return 0
    i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60
  done
  return 1
}

echo "== inverter128 $(date)"
retry 4 $PY -m v2t.train --stage inversion --init ckpt/inverter128_fallback --resume --max_tokens 128 --data $D --out ckpt/inverter128 \
  --bs 128 --micro_bs 16 --epochs 1 --lr 3e-5 --lr_proj 2e-4 --warmup 100 --eval_every 500 --save_every 500 \
  --compile --pad_to 16 >> logs/train_inverter128_full.log 2>&1 || { echo "FAILED inverter"; exit 1; }

echo "== hypotheses $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter128 --data $D --splits val,train --limit $NHYP --bs 768 \
  >> logs/gen_hyp128_full.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }

echo "== corrector128 $(date)"
retry 4 $PY -m v2t.train --stage corrector --init ckpt/corrector128_fallback --resume --max_tokens 128 --data $D --out ckpt/corrector128 --limit $NHYP \
  --bs 128 --micro_bs 8 --epochs 1 --lr 3e-5 --lr_proj 2e-4 --warmup 100 --eval_every 500 --save_every 500 \
  --compile --pad_to 16 >> logs/train_corrector128_full.log 2>&1 || { echo "FAILED corrector"; exit 1; }

echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128 --data $D --n 200 --configs 0x1,0x4,1x4,3x4,5x4 --show 2 --out logs/eval128_full_200.json > logs/eval128_full_200.log 2>&1
echo "== demo $(date)"
$PY invert.py --vectors examples/vecs128.tsv --metadata examples/meta128.tsv --steps 5 --beam 4 --out examples/result128.tsv > logs/demo128_full.log 2>&1
echo "== all done $(date)"
