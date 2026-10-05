#!/bin/sh
# Full 128-token pipeline, warm-started from the 32-token models.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=data/corpus128
until grep -q "done" logs/corpus128_wiki.log && grep -q "done" logs/corpus128_web.log; do sleep 10; done
echo "== merge $(date)"
$PY - <<'PYEOF'
import json, random
random.seed(0)
for split in ["train", "val", "test"]:
    lines = []
    for src in ["wiki", "web"]:
        lines += [l for l in open(f"data/corpus128/{split}.{src}.jsonl", encoding="utf-8").read().split("\n") if l]
    random.shuffle(lines)
    open(f"data/corpus128/{split}.jsonl", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(split, len(lines))
PYEOF
echo "== embed $(date)"
for s in val test train; do $PY scripts/embed_corpus.py $D/$s.jsonl $D/$s.emb.npy --bs 256 || exit 1; done
echo "== inverter128 $(date)"
$PY -m v2t.train --stage inversion --init ckpt/inverter --max_tokens 128 --data $D --out ckpt/inverter128 \
  --bs 128 --micro_bs 16 --epochs 1 --lr 3e-5 --lr_proj 2e-4 --warmup 100 --eval_every 500 --save_every 1000 \
  --compile --pad_to 16 > logs/train_inverter128.log 2>&1 || exit 1
echo "== hypotheses $(date)"
$PY scripts/gen_hypotheses.py --inverter ckpt/inverter128 --data $D --splits val,train --limit 400000 --bs 1024 > logs/gen_hyp128.log 2>&1 || exit 1
echo "== corrector128 $(date)"
$PY -m v2t.train --stage corrector --init ckpt/corrector --max_tokens 128 --data $D --out ckpt/corrector128 --limit 400000 \
  --bs 128 --micro_bs 8 --epochs 1 --lr 3e-5 --lr_proj 2e-4 --warmup 100 --eval_every 500 --save_every 1000 \
  --compile --pad_to 16 > logs/train_corrector128.log 2>&1 || exit 1
echo "== eval $(date)"
$PY scripts/eval.py --inverter ckpt/inverter128 --corrector ckpt/corrector128 --data $D --n 200 --configs 0x1,0x4,1x4,3x4,5x4 --show 3 --out logs/eval128_200.json > logs/eval128_200.log 2>&1
echo "== all done $(date)"
