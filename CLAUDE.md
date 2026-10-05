# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a vec2text-style embedding inverter (Morris et al., 2023: inversion model + iterative corrector) for **`Qwen/Qwen3-Embedding-0.6B`**, focused on **French** text (RAG use case). Conversation with the user is in French.

Decisions already made (don't revisit unless asked):
- `mistral-embed` was rejected: API-only, no public weights.
- TF Projector demo embeddings (`data/projector/`) were rejected: precomputed tables, no usable encoder for sentences.
- Encoder = Qwen3-Embedding-0.6B, chosen because its tokenizer is identical to the generator **`Qwen/Qwen3-0.6B-Base`**, which is the intended starting point for the inverter/corrector.

## Environment

- WSL2 on Windows 11, RTX 5090 (32 GB, sm_120; ~1.7 GB used by Windows), 62 GB RAM.
- Python venv at `.venv/` (created with `uv`). Install packages with `uv pip install --python .venv/bin/python <pkg>`; run with `.venv/bin/python`.
- PyTorch must come from the cu128 (or newer) index for Blackwell support: `--index-url https://download.pytorch.org/whl/cu128`.
- Models live in the default HF cache (`~/.cache/huggingface/hub`). No `HF_TOKEN` set.
- No git repo yet.

## Layout

- `v2t/model.py`: `Vec2TextQwen` = Qwen3-0.6B-Base + `EmbProj` MLPs mapping a 1024-d embedding to `n_prefix` (8) virtual tokens prepended to the decoder input. Corrector variant gets 3 prefixes (target, hypothesis, difference) + hypothesis tokens + `<|im_start|>` separator. Loss only on target tokens + `<|endoftext|>`. Checkpoint = HF `save_pretrained` dir + `proj.pt` + `v2t_config.json`; load with `Vec2TextQwen.load(path)`.
- `v2t/data.py`: `Corpus` (texts jsonl + fp16 `.emb.npy` memmap, optional `.hyp.*` for the corrector).
- `v2t/train.py`: custom loop (fp32 master weights + bf16 autocast, AdamW, length-bucketed batches, right padding, no attention mask). `--compile` pads lengths to multiples of 8 and uses static-shape `torch.compile`.
- `v2t/infer.py`: `Inverter` = inversion model → beam → re-embed → keep best-by-cosine, then corrector loop (vec2text algorithm).
- `invert.py` (user-facing tool: Projector-format vectors → texts), `embed.py` (texts → Projector-format vectors).
- `scripts/`: `build_corpus.py`, `embed_corpus.py`, `gen_hypotheses.py`, `eval.py`, `check_setup.py`.
- `data/corpus/{train,val,test}.jsonl` + `.emb.npy` (3.0M / 3k / 3k passages ≤32 tokens, 60% Wikipedia FR, 40% FineWeb-2 FR). `data/smoke/` is a 3k-sample copy for quick tests. `data/projector/` = TF Projector demo data (unused).
- `ckpt/inverter`, `ckpt/corrector`: trained models. `logs/`: run logs.

## Commands

- `.venv/bin/python scripts/check_setup.py`: checks tokenizer compatibility, embedding properties, and GPU encoding throughput.
- Pipeline (all from the repo root):
  1. `scripts/build_corpus.py --source wiki|web --n_train N` then merge into `train.jsonl` (shuffle; split on `\n` only, texts may contain other Unicode line separators).
  2. `scripts/embed_corpus.py data/corpus/X.jsonl data/corpus/X.emb.npy`
  3. `python -m v2t.train --stage inversion --out ckpt/inverter [--limit N] [--compile]`
  4. `scripts/gen_hypotheses.py --inverter ckpt/inverter [--limit N]` → `train.hyp.jsonl`, `train.hyp.emb.npy`
  5. `python -m v2t.train --stage corrector --init ckpt/inverter --out ckpt/corrector`
  6. `scripts/eval.py --inverter ckpt/inverter --corrector ckpt/corrector --configs 0x1,0x4,1x4,5x4`
- Smoke test of training: `python -m v2t.train --stage inversion --data data/smoke --out ckpt/smoke_inv --bs 64 --micro_bs 64 --max_steps 120`.
- Tool: `python invert.py --vectors vecs.tsv [--metadata meta.tsv] [--steps 3 --beam 4] [--out out.tsv]`.

## 128-token variant

- `data/corpus128/` (600k train passages of 16..128 tokens, same sources), `ckpt/inverter128`, `ckpt/corrector128`.
- Built by `scripts/run_128.sh` (waits for the corpus builders, merges, embeds, trains both stages, evals). Both stages are **warm-started** from the 32-token checkpoints (`--init ckpt/inverter` / `--init ckpt/corrector`, `--max_tokens 128`): `train.py --init` loads a same-type checkpoint fully, or builds a corrector from an inverter checkpoint.
- `max_tokens` is stored in `v2t_config.json`; generation length and hypothesis truncation derive from it. `invert.py` defaults to the `*128` checkpoints when present.
- `--pad_to 16` with `--compile` for long sequences (fewer recompiles). micro_bs 16 (inverter, 18.4 GB) / 8 (corrector).
- **Crash recovery**: each save also writes `optim.pt` + `train_state.json`; `train.py --resume` reloads them from `--out` and replays the same batch order, skipping already-trained steps. `scripts/run_128full.sh` wraps each stage in `retry` (uses `--resume`) because WSL can lose GPU memory to Windows (`dxgk make_resident: -12` in dmesg → `cudaErrorMemoryAllocation`); this killed the first inverter128 run at step ~1700 on 2026-10-01 09:46.
- Long outputs loop (greedy/beam repeats a sentence until max length). `Inverter.no_repeat_ngram_size` (HF `no_repeat_ngram_size`) fixes it; `invert.py` defaults to 6 for models with `max_tokens > 64`, `eval.py --no_repeat N`.
- Models stop by themselves at their training length (40% of training windows are full-length and cut mid-sentence, so the model learned "EOS at 128 tokens"); raising `max_new_tokens` alone changes nothing. `invert.py --min_new_tokens N` (HF `min_new_tokens`, via `Inverter.min_new_tokens`) forces longer outputs; quality degrades past the training length.
- `v2t/env.py: quiet_offline()` (called first thing in `invert.py` / `embed.py`) sets `HF_HUB_OFFLINE=1` when both Qwen models are in the HF cache (otherwise every `from_pretrained` does a HEAD request to the Hub and prints the unauthenticated-requests warning) and disables the transformers 5 "Loading weights" progress bars. Unset `HF_HUB_OFFLINE` to download a new model.
- Inference memory: the corrector step runs bs×beam×beam sequences; `Inverter.invert(bs=None)` picks bs from `max_tokens` and beam (16 for 128 tokens/beam 4). bs=64 OOMs at 128 tokens.
- Fallback chain `scripts/run_128b.sh` (inverter128 step-1000 checkpoint + corrector on 160k) ran first; its checkpoints are kept as `ckpt/*128_fallback`.
- Throughput: inverter128 ~1.1 it/s at bs 128 (~140 samples/s, ~5.3k target tok/s). Val loss per token is ~2.0 (vs 1.24 at 32 tokens): a 1024-d embedding carries much less information per token at 128 tokens.

## 1.7B decoder variant (`scripts/run_1p7b.sh`, started 2026-10-01 22:19)

- Decoder `Qwen/Qwen3-1.7B-Base` (same tokenizer; hidden 2048, same KV layout as 0.6B so inference memory per sequence is unchanged). Outputs `ckpt/inverter128_1p7b`, `ckpt/corrector128_1p7b`. Not warm-started (different size).
- Memory recipe: `--freeze_embed` (tied 311M embedding matrix frozen → 1445M trainable) + `--optim adamw8bit` (bitsandbytes 0.50.2 works on sm_120). Eager micro_bs 8 at 128 tokens: 25.3 GB, 32 samples/s; micro_bs 4: 22.3 GB, 20 samples/s. Chain uses micro_bs 8 (inverter) / 4 (corrector) with retry+resume as the OOM safety net.
- Hypotheses at bs 384 (KV cache ~13 GB); corrector on 300k samples. Actual: inverter 2h20 (0.65 it/s compiled, 24 GB), hypotheses 1h18, corrector 2h10 (0.35 it/s), eval 22 min → done 04:33 on 2026-10-02.
- Results: inverter val 1.745 (0.6B: 1.841), corrector val 1.698 (0.6B: 1.760) — better teacher-forced losses — but test BLEU is much worse (logs/eval128_1p7b_200.json): 0x4 BLEU 10.1 / F1 47.8; 1x4 13.3; 3x4 13.7 / F1 54.2 / exact 2.5% / cos 0.90; 5x4 13.8. The corrector barely helps (0.6B: 10.6 → 23.4). Note: `gen_hypotheses` for the 1.7B overwrote `data/corpus128/{train,val}.hyp.*` (now 1.7B hypotheses).
- Diagnosis (2026-10-02 04:40): no loading bug (bf16 inference val loss == training val loss for both sizes). Greedy inverter hypotheses on val: 1.7B F1 41.8 / cos 0.83 vs 0.6B F1 47.5 / cos 0.87; one greedy corrector step: 1.7B → F1 47.2, 0.6B → F1 54.7. Both models give their own greedy output ~half the NLL of the reference (exposure bias, similar ratio). So the 1.7B decodes fluent but less faithful text despite lower teacher-forced loss; suspects: frozen (tied) lm_head and the under-trained corrector (val still dropping steeply at 300k).
- Follow-up `scripts/run_1p7b_corr2.sh` (hypotheses rows 300k..600k appended via `gen_hypotheses --start`, corrector v2 on 600k warm from corrector128_1p7b → `ckpt/corrector128_1p7b_v2`, done 12:50): corrector val 1.644; test (no_repeat 6, logs/eval128_1p7b_v2_200.json): 1x4 BLEU 14.4 / F1 55.9; 3x4 16.9 / 59.3 / exact 4% / cos 0.923; 5x4 17.4 / 59.7 / 4%. **Verdict: the 1.7B decoder (frozen tied embeddings, 8-bit AdamW, 900k corrector samples) stays well below the 0.6B (3x4: 23.4 / 66.4 / 13%). Not adopted; `ckpt/*128_1p7b*` kept for reference.**

## v2 plan for the 0.6B 32-token models (queued 2026-10-02 06:50, `scripts/run_v2.sh`)

User domain: French social security + human health. Levers applied: (1) 3 epochs, (4) 16 prefix tokens, (3) corrector hypotheses partly on passages unseen by the inverter, (2) second-round corrector trained on its own step-1 outputs, (6) domain data.
- `scripts/build_domain_corpus.py`: keyword-filtered Wikipedia FR + FineWeb-2 FR → `data/domain/` (≤32-token passages containing a domain keyword).
- `scripts/prep_v2.sh` → `scripts/make_corpus_v2.py`: `data/corpus_v2` (inverter: general train[:2.6M] + domain train; val = both; `test`, `test_domain`), `data/corpus_corr` (corrector: fresh general[2.6M:] + 1.6M seen). General embeddings reused from `data/corpus`.
- Chain: `inverter_v2` (`--base ckpt/inverter --n_prefix 16`, 3 epochs) → hypotheses on corpus_corr → `corrector_v2a` → `scripts/gen_hypotheses_step1.py` (apply corrector once → `data/corpus_mix`, texts×2 with hyp0+hyp1) → `corrector_v2b` → evals (`logs/eval_v2b_300.json`, `logs/eval_v2b_domain_300.json`, baseline `logs/eval_v1_domain_300.json`).
- Started 12:50 (after the 1.7B work). inverter_v2: 3 epochs in 4 h (4.9 it/s), val 1.651 (step 2k) → 1.223 (step 68.7k, plateau from ~60k). corrector_v2a (2M, 1 epoch, 2 h): val 1.067 (harder val: fresh hypotheses).
- Intermediate test (general, 300, logs/eval_v2a_300.json): inverter alone 0x4 BLEU 31.7 / F1 71.4 / exact 19.3% / cos 0.928 (v1: 25.6 / 66.6 / 16%) → clear gain from levers 1+4(+6); but 5x4 with corrector_v2a = 46.8 / 79.6 / 33.3% < v1's 56.9 / 84.2 / 37.3%: corrector_v2a (15.6k steps, 48 prefix tokens) is under-trained vs v1's. corrector_v2b (4M mixture, 31k more steps) must close that gap.
- Final (chain done 2026-10-03 02:57; corrector_v2b val 1.165 on the mixture val). General test 300 (logs/eval_v2b_300.json): 0x1 24.8 / 66.6 / 15%; 0x4 31.7 / 71.4 / 19.3%; 1x4 43.1 / 77.4 / 29%; 5x4 48.7 / 79.9 / 35%; 10x8 55.6 / 83.3 / 36.7% / cos 0.972 (3 s/sample). Domain test 300 (logs/eval_v2b_domain_300.json): 0x4 34.7 / 72.4 / 11%; 5x4 59.2 / 85.6 / 33.7%; 10x8 65.6 / 88.2 / 35%. v1 on the domain test (logs/eval_v1_domain_300.json): 0x4 25.1 / 63.8 / 7.3%; 5x4 60.6 / 85.2 / 32%.
- Reading: inverter_v2 ≫ inverter v1 alone (+6 BLEU general, +10 domain), but corrector_v2b under-corrects: equal to v1 at 1 step (43.1 vs 43.6) and gains little from steps 2–5 (→48.7 vs v1 →56.9); v2b only +1.9 over v2a despite 2× steps. Likely cause: the step-1 hypotheses on *seen* training passages are near-perfect (corrector_v2a had trained on those pairs), so half of the mixture teaches "copy the hypothesis" → conservative corrector. Next idea: corrector trained only on hypotheses for passages unseen by the inverter (needs ~2M fresh passages).
- Mixed pair **inverter_v2 + corrector (v1)** (logs/eval_inv2_corr1_*.json): general 1x4 45.8 / 5x4 56.8 / 84.2 / exact 38.3%; domain 5x4 61.6 / 86.0 / 33%. v1 at 10x8 on domain: 68.7 / 88.4 / 37%. → `invert.py --profile short` = inverter_v2 + corrector; `--profile long` = the 128 pair. The v1 corrector remains the best corrector.

## v3 corrector (queued 2026-10-03 05:56, `scripts/run_v3.sh`)

- Goal: adapt the best corrector (`ckpt/corrector`, v1, n_prefix 8) to inverter_v2 hypotheses on passages **never seen by inverter_v2**.
- Fresh data: `data/fresh/` (build_corpus.py `--skip_docs` 400k wiki / 300k web, 900k + 700k passages, `--n_eval 0`), `data/fresh_domain/` (build_domain_corpus.py on FineWeb-2 shard `000_00010.parquet` via `--data_files`, 150k), plus general train[2.6M:] (400k). `scripts/make_corpus_corr3.py` → `data/corpus_corr3` (dedup against inverter data; val = corpus_v2 val).
- Chain: hypotheses (inverter_v2) → `corrector_v3` (`--init ckpt/corrector`, 2 epochs, lr 3e-5) → evals `logs/eval_v3_300.json`, `logs/eval_v3_domain_300.json`. Ran 05:56–11:14 (corpus 2.14M fresh; corrector 33.4k steps at 2.8 it/s, 3.5 h).
- **Results: big win.** corrector_v3 val 0.942 (v2a: 1.067, still decreasing at the end). inverter_v2 + corrector_v3, general test: 1x4 48.1 / 81.0 / 33%; 5x4 62.8 / 86.8 / 43%; 10x8 72.4 / 90.0 / 47.7% / cos 0.985. Domain: 1x4 52.8 / 82.1 / 25.7%; 5x4 72.1 / 90.4 / 42%; 10x8 78.9 / 92.8 / 42.3% / cos 0.9925. Previous best (inv_v2 + corr v1): general 5x4 56.8, domain 5x4 61.6 / 10x8 68.7. → `--profile short` now = inverter_v2 + corrector_v3. Lesson: **train the corrector on hypotheses for passages the inverter never saw**; unlike v2b, more correction steps keep paying (10x8 ≫ 5x4).
- `scripts/run_v3_128.sh` (queued 2026-10-03 13:43): same recipe at 128 tokens. Fresh 16..128-token passages (`data/fresh128`: wiki skip 620k docs, web skip 470k; `data/fresh_domain128`: shard `000_00011`, with a 1000-passage domain test split), `scripts/make_fresh_corpus.py` → `data/corpus_corr128` (+ `test_domain`), hypotheses by inverter128 (bs 768), `corrector128_v3` (`--init ckpt/corrector128`, 2 epochs, micro_bs 8, eval_bs 32), evals `logs/eval128_v3_*.json`. Gotcha hit: `--stop_at HH:MM` already in the past stops the builder after 1000 docs and still prints "done".
- run_v3_128 results (done 21:12; 590k fresh passages, corrector128_v3 val 1.705 vs 1.760): general 128 test (200, no_repeat 6): 1x4 18.3 / 62.5 / 6%; 3x4 23.8 / 67.6 / 12.5%; 5x4 25.2 / 68.7 / 11.5%; 10x8 (100) 26.1 / 70.3 / 16% / cos 0.959 (20 s/sample). Domain-128 test (200): v3 5x4 16.4 / 57.6 / 4% vs v1 15.0 / 55.7 / 6%; inverter alone 6.8 / 43.7 (hard set). → small gain (+1–2 BLEU), adopted in `--profile long`; confirms the 128-token ceiling is the embedding's information, not the corrector.
- Natural next steps: (a) same recipe for the 128-token pair (fresh 128-token passages → corrector128_v3); (b) corrector v4 = second round on *fresh* step-1 hypotheses (the v2b idea done right); (c) 12M-passage inverter.
- `iter_docs(source, data_files)` streams a given parquet with the `parquet` builder (`hf://datasets/HuggingFaceFW/fineweb-2/<path>`); passing `data_files` to the `fineweb-2` builder fails (needs a config name).

## Rotation tooling (`make_rotation.py`, `rotate_vectors.py`, `sweep_rotation.py`)

- Rotation = Q R Qᵀ with Q a Haar-random orthonormal basis (QR of a Gaussian matrix) and R block-diagonal 2×2 rotations; `--basis canonical` uses coordinate planes (weak: planes public). `--degrees D` = uniform angles in [0, D] (up to 360); `--fixed_angle 90` = K = B Aᵀ − A Bᵀ (Kᵀ = −K, K² = −I, inverse −K); 0/180/360 refused (±I). `--seed_hex` = 256-bit seed (HMAC of master key + user id) for per-user key derivation — the chosen design: one 90° key per user.
- Measured facts (2026-10-03): random-plane rotation leaves the inverter's output readable up to ~100° max angle because 93% of the perturbation falls outside the ~100-d semantic subspace (useful-subspace cosine 0.73 at 100°); breakdown at 105°. Principal-component planes hide more per degree but the per-plane angles are readable from the leaked covariance (6.9° error on the top planes, no pairs needed). A secret permutation of coordinates is weak: 3 known pairs recover it fully; 100k leaked vectors + per-coordinate moments recover 49% of it. Procrustes recovers a dense rotation with ~1024 pairs (200 → cos 0.63, 500 → 0.88). Decoy vectors do not hinder Procrustes (they never form pairs) but can pad partition sizes. Per-user keys cap pairs per key → Procrustes infeasible.
- `scripts/attack_pairs.py` (key-period calibration): Procrustes with p known pairs, then inversion with profile short 5x4 on 60 test texts (no-rotation reference BLEU ~63 / F1 ~87): p ≤ 200 → BLEU ≤ 1.5 / F1 ≤ 23% (baseline noise); p = 300 → 3.4 / 29%; p = 500 → 10.3 / 48% (topic readable); p = 1000 → exact key. Recommendation: ≤ 100 known pairs per key (safety factor 2 for smarter estimators), i.e. X ≈ 100 / (fraction of texts an attacker can know).

## Results log

- Inverter (1 epoch, 3.0M passages, bs 128, lr 5e-5/5e-4, 75 min): val loss 2.19 (step 1k) → 1.235 (step 23.4k, still decreasing). Test (300 samples): greedy BLEU 18.1 / token-F1 60.0 / exact 10% / cos 0.882; beam 4 + cosine re-rank BLEU 25.6 / F1 66.6 / exact 16% / cos 0.914.
- Hypothesis generation (greedy, bs 2048): ~975 samples/s. bs 1024 was only ~620/s (per-step Python overhead dominates).
- Corrector (1 epoch, 3.0M, init from inverter, micro_bs 32, 150 min): val loss 1.437 → 1.013. Test (300, logs/eval_full_300.json): 1x4 BLEU 43.6 / F1 78.4 / exact 30%; 3x4 BLEU 54.2 / F1 83.0 / exact 36.7%; 5x4 BLEU 56.9 / F1 84.2 / exact 37.3% / cos 0.971 (0.36 s/sample).
- 128-token fallback (inverter128 @ step 1000, val 1.98; corrector on 160k, val 1.85): inverter alone on 150 test passages: greedy BLEU 7.8 / F1 47; beam 4 BLEU 10.4 / F1 50 / cos 0.86.
- 128-token full: inverter128 1 epoch on 600k from the step-1000 checkpoint (1.4 it/s, 52 min): val 1.841. Hypotheses 400k in 51 min (bs 768, ~130/s).
- 128-token full corrector (1 epoch on 400k, from the fallback corrector, 0.9 it/s, 70 min): val 1.874 → 1.760. Test (200, logs/eval128_full_200.json, no_repeat off): 0x1 BLEU 8.5 / F1 48.5; 0x4 10.2 / 51.1; 1x4 17.9 / 61.5 / exact 6%; 3x4 22.9 / 65.7 / 12%; 5x4 23.6 / 66.7 / 13% / cos 0.948 (3.2 s/sample).
- A/B `--no_repeat 6` on the same 200 (logs/eval128_norepeat6_200.json): 0x4 BLEU 10.6 / F1 52.5 / cos 0.886; 3x4 BLEU 23.4 / F1 66.4 / exact 13% / cos 0.948 → small but consistent gain, kept as default for 128-token models.
- Round 2 (`scripts/run_128round2.sh`, 17:17–20:38): one more epoch of both 128 models at lr 2e-5 → `ckpt/*128_r2`. Inverter val 1.8414 (vs 1.8406): plateau. Test with no_repeat 6 (logs/eval128_r2_200.json): 0x4 BLEU 10.8 / F1 52.7 / cos 0.894; 3x4 22.2 / 65.4 / 11.5%; 5x4 23.3 / 66.2 / 13.5% / cos 0.950. Within noise of round 1 → **round 1 (`ckpt/inverter128`, `ckpt/corrector128`) stays the default**; `_r2` kept as alternates. Conclusion: at 128 tokens the limit is the information in a 1024-d vector, not training length.
- Whole pipeline wall-clock on this machine: ~6.5 h (corpus 10 min, embeddings 20 min, inverter 75 min, hypotheses 70 min, corrector 150 min, eval 10 min).

## Gotchas

- `pkill -f <pattern>` kills the calling shell if the pattern appears anywhere in the same command line (e.g. a relaunch in the same command). Use `pkill -f "[v]2t.train"` and never relaunch in the same Bash call.
- `torch.compile` wraps `lm.model` in an `OptimizedModule`: `state_dict` keys get a `model._orig_mod.` prefix. `Vec2TextQwen.save()` unwraps before `save_pretrained`.
- `str.splitlines()` splits on U+2028/U+0085 etc.; corpus jsonl must be split on `\n` only.
- `pgrep -f "[v]2t.train"` still matches a command line containing `v2t/train.py` (the `.` is a wildcard). Kill by PID obtained in a *separate* Bash call, never in a command that also mentions the target.
- Validation OOM: `evaluate()` computes fp32 logits for all target tokens of a batch; with the 1.7B at 128 tokens, bs 128 needs ~15 GB → use `--eval_bs 16`. The 1.7B corrector v2 died 5× at its first eval (step 500, before any checkpoint) on 2026-10-02 06:28–08:14 because of this.

## Measured training throughput (RTX 5090)

- Inversion stage, eager, micro-batch 64: ~5 it/s ≈ 320 samples/s, 18.5 GB. micro-batch 128 gives no gain (28 GB).
- `torch.compile(dynamic=True)` recompiles forever: unusable. Static shapes with pad-to-8 compile fine (~5 min warm-up over the handful of shapes).

## Verified encoder facts (scripts/check_setup.py)

- Embedding and generator tokenizers: same vocab (151,669 tokens), same ids for the same text. The only difference: the embedding tokenizer appends `<|endoftext|>` (TemplateProcessing post-processor), and pooling takes that last token. Generator tokenizer adds nothing.
- Encoding pipeline (sentence-transformers): Transformer → last-token Pooling → Normalize. Output is 1024-d, L2-normalized (Matryoshka truncation possible, re-normalize after truncating).
- Prompts: `document` = "" (no instruction), `query` = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:". Inversion targets are documents → encode **without** prompt; keep this consistent everywhere.
- Tokenizer padding side defaults to `right`; sentence-transformers handles last-token pooling correctly with it. If you write custom pooling, pick the last non-pad token (or use left padding).
- Throughput: ~2,300 short French sentences/s in bf16, batch 512, SDPA (no flash-attn), ~4 GB VRAM.
