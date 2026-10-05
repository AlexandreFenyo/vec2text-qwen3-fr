#!/bin/sh
# corrector128_v3: the 128-token corrector retrained on inverter128 hypotheses for passages inverter128 never saw.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
D=data/corpus_corr128
retry() { n=$1; shift; i=0; while [ $i -lt $n ]; do "$@" && return 0; i=$((i+1)); echo "   attempt $i failed ($(date)), retrying in 60s"; sleep 60; done; return 1; }

until grep -q "done" logs/fresh128_wiki.log && grep -q "done" logs/fresh128_web.log && grep -q "done" logs/fresh_domain128.log; do sleep 60; done
echo "== make corpus $(date)"
$PY scripts/make_fresh_corpus.py --dirs data/fresh128 data/fresh_domain128 --exclude data/corpus128/train.jsonl \
  --val data/corpus128 --test_domain data/fresh_domain128/test.web.jsonl --out $D >> logs/make_corpus_corr128.log 2>&1 || { echo "FAILED corpus"; exit 1; }

echo "== hypotheses $(date)"
retry 3 $PY scripts/gen_hypotheses.py --inverter ckpt/inverter128 --data $D --splits val,train --bs 768 \
  >> logs/gen_hyp_corr128.log 2>&1 || { echo "FAILED hypotheses"; exit 1; }

echo "== corrector128_v3 $(date)"
retry 5 $PY -m v2t.train --stage corrector --init ckpt/corrector128 --resume --max_tokens 128 --data $D --out ckpt/corrector128_v3 \
  --bs 128 --micro_bs 8 --epochs 2 --lr 3e-5 --lr_proj 3e-4 --warmup 200 --eval_every 1000 --eval_bs 32 --save_every 1000 --seed 3 \
  --compile --pad_to 16 >> logs/train_corrector128_v3.log 2>&1 || { echo "FAILED corrector"; exit 1; }

echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128_v3 --data data/corpus128 --n 200 --configs 1x4,3x4,5x4 --show 2 --no_repeat 6 --out logs/eval128_v3_200.json > logs/eval128_v3_200.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128_v3 --data $D --split test_domain --n 200 --configs 0x4,1x4,3x4,5x4 --show 2 --no_repeat 6 --out logs/eval128_v3_domain_200.json > logs/eval128_v3_domain_200.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128 --data $D --split test_domain --n 200 --configs 0x4,5x4 --show 0 --no_repeat 6 --out logs/eval128_v1_domain_200.json > logs/eval128_v1_domain_200.log 2>&1
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128_v3 --data data/corpus128 --n 100 --configs 10x8 --show 0 --no_repeat 6 --out logs/eval128_v3_10x8_100.json > logs/eval128_v3_10x8_100.log 2>&1
echo "== all done $(date)"
