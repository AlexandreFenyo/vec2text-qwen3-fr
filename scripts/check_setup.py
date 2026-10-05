"""Sanity checks: tokenizer compatibility between Qwen3-Embedding-0.6B and Qwen3-0.6B-Base,
embedding properties, and encoding throughput on the GPU."""
import json, time, torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

EMB, GEN = "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-0.6B-Base"
te, tg = AutoTokenizer.from_pretrained(EMB), AutoTokenizer.from_pretrained(GEN)

print("vocab sizes:", len(te), len(tg), "| vocab identical:", te.get_vocab() == tg.get_vocab())
print("special:", te.eos_token, te.pad_token, te.padding_side, "|", tg.eos_token, tg.pad_token, tg.padding_side)
pe = json.loads(te.backend_tokenizer.to_str()).get("post_processor")
pg = json.loads(tg.backend_tokenizer.to_str()).get("post_processor")
print("post_processor emb:", json.dumps(pe)[:300]); print("post_processor gen:", json.dumps(pg)[:300])

samples = ["Le Conseil d'État a rendu sa décision le 3 mars 2024.",
           "L'assurée doit transmettre l'arrêt de travail sous 48 heures à sa CPAM.",
           "Ça coûte 12,50 € — « déjà » payé ? Oui… œuvre, naïf, Noël."]
for s in samples:
    a, b = te(s)["input_ids"], tg(s)["input_ids"]
    print(len(a), len(b), "same ids (ignoring trailing specials):", a[:len(b)] == b, "| emb tail:", te.convert_ids_to_tokens(a[-2:]))
    assert tg.decode(b) == s

m = SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
print("max_seq_length:", m.max_seq_length, "| prompts:", m.prompts, "| modules:", [type(x).__name__ for x in m])
e = m.encode(samples, convert_to_tensor=True)
print("dim:", e.shape, "norms:", e.float().norm(dim=1).tolist())
print("cos:", (e.float() @ e.float().T).round(decimals=3).tolist())

# Throughput on ~32-token French sentences
sents = [f"{samples[i % 3]} Phrase numéro {i}." for i in range(20000)]
m.encode(sents[:512], batch_size=512)
torch.cuda.synchronize(); t = time.time()
m.encode(sents, batch_size=512)
torch.cuda.synchronize(); dt = time.time() - t
print(f"throughput: {len(sents)/dt:.0f} sentences/s (bf16, batch 512), peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
