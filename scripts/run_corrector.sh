#!/bin/sh
# Wait for hypothesis generation to finish, then train the corrector (stage 2).
cd "$(dirname "$0")/.."
until grep -q "train: embedded" logs/gen_hyp.log; do
  if grep -q Traceback logs/gen_hyp.log; then echo "gen_hypotheses failed"; exit 1; fi
  sleep 30
done
exec .venv/bin/python -m v2t.train --stage corrector --init ckpt/inverter --data data/corpus --out ckpt/corrector \
  --bs 128 --micro_bs 32 --epochs 1 --eval_every 1000 --save_every 3000 --warmup 300 --compile
