#!/bin/sh
# v2 of the 0.6B 32-token models: 16 prefix tokens, 3 epochs, domain data, fresh-hypothesis corrector, second-round corrector.
# Starts when the 1.7B chain (run_1p7b_corr2.sh) is finished. Retries/resume on crash.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }

until grep -q -E "== all done|FAILED" logs/run_1p7b_corr2.log || ! pgrep -f "[r]un_1p7b_corr2.sh" >/dev/null; do sleep 60; done
until [ -f data/corpus_corr/train.emb.npy ] && [ -f data/corpus_v2/test_domain.emb.npy ]; do sleep 60; done
echo "== start $(date)"

echo "== inverter_v2 $(date)"
retry 5 $PY -m v2t.train --stage inversion --base ckpt/inverter --n_prefix 16 --resume --max_tokens 32 --data data/corpus_v2 --out ckpt/inverter_v2 \
  --bs 128 --micro_bs 64 --epochs 3 --lr 5e-5 --lr_proj 5e-4 --warmup 300 --eval_every 2000 --save_every 2000 \
  --compile >> logs/train_inverter_v2.log 2>&1 || { echo "FAILED inverter"; exit 1; }

echo "== hypotheses0 $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter_v2 --data data/corpus_corr --splits val,train --bs 2048 \
  >> logs/gen_hyp_corr.log 2>&1 || { echo "FAILED hypotheses0"; exit 1; }

echo "== corrector_v2a $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/inverter_v2 --resume --max_tokens 32 --data data/corpus_corr --out ckpt/corrector_v2a \
  --bs 128 --micro_bs 32 --epochs 1 --lr 5e-5 --lr_proj 5e-4 --warmup 300 --eval_every 2000 --save_every 2000 \
  --compile >> logs/train_corrector_v2a.log 2>&1 || { echo "FAILED corrector a"; exit 1; }

echo "== eval a $(date)"
$PY scripts/eval.py --inverter ckpt/inverter_v2 --corrector ckpt/corrector_v2a --data data/corpus_v2 --n 300 --configs 0x4,5x4 --show 0 --out logs/eval_v2a_300.json > logs/eval_v2a_300.log 2>&1

echo "== hypotheses1 $(date)"
retry 3 $PY scripts/gen_hypotheses_step1.py --corrector ckpt/corrector_v2a --data data/corpus_corr --out data/corpus_mix --splits val,train --bs 512 \
  >> logs/gen_hyp_step1.log 2>&1 || { echo "FAILED hypotheses1"; exit 1; }

echo "== corrector_v2b $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/corrector_v2a --resume --max_tokens 32 --data data/corpus_mix --out ckpt/corrector_v2b \
  --bs 128 --micro_bs 32 --epochs 1 --lr 3e-5 --lr_proj 3e-4 --warmup 300 --eval_every 2000 --save_every 2000 --seed 1 \
  --compile >> logs/train_corrector_v2b.log 2>&1 || { echo "FAILED corrector b"; exit 1; }

echo "== eval b $(date)"
$PY scripts/eval.py --inverter ckpt/inverter_v2 --corrector ckpt/corrector_v2b --data data/corpus_v2 --n 300 --configs 0x1,0x4,1x4,5x4,10x8 --show 3 --out logs/eval_v2b_300.json > logs/eval_v2b_300.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter_v2 --corrector ckpt/corrector_v2b --data data/corpus_v2 --split test_domain --n 300 --configs 0x4,5x4,10x8 --show 3 --out logs/eval_v2b_domain_300.json > logs/eval_v2b_domain_300.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter --corrector ckpt/corrector --data data/corpus_v2 --split test_domain --n 300 --configs 0x4,5x4 --show 0 --out logs/eval_v1_domain_300.json > logs/eval_v1_domain_300.log 2>&1
echo "== all done $(date)"
