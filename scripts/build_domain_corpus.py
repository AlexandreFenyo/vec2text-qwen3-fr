"""Domain corpus (French social security + human health): stream Wikipedia FR and FineWeb-2 FR, keep documents
with enough distinct domain keywords, cut them into passages (same procedure as build_corpus.py).
Output: {out_dir}/{train,val,test}.{wiki,web}.jsonl"""
import argparse, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import iter_docs, passages_from_doc
from transformers import AutoTokenizer

KEYWORDS = """sécurité sociale|assurance maladie|cnam|cpam|ameli|caisse primaire|caisse nationale|cotisation|cotisations|
prestation|prestations|allocation|allocations|retraite|carsat|cnav|urssaf|mutuelle|complémentaire santé|remboursement|
remboursé|indemnités journalières|arrêt de travail|arrêt maladie|médecin|médecins|médical|médicale|médicaux|hôpital|
hôpitaux|hospitalisation|patient|patients|traitement|traitements|maladie|maladies|vaccin|vaccins|vaccination|symptôme|
symptômes|diagnostic|pharmacie|médicament|médicaments|ordonnance|infirmier|infirmière|santé publique|agence régionale de santé|
haute autorité de santé|affection de longue durée|ticket modérateur|franchise médicale|carte vitale|protection sociale|
prévoyance|invalidité|handicap|maternité|congé maternité|accident du travail|maladie professionnelle|caf|allocations familiales|
pôle emploi|france travail|assurance chômage|chirurgie|chirurgien|thérapie|thérapeutique|cancer|diabète|hypertension|
épidémie|pandémie|virus|bactérie|infection|antibiotique|anesthésie|urgences|consultation|généraliste|spécialiste|
dépistage|prévention|soins|parcours de soins|dossier médical|télémédecine|ehpad|dépendance|autonomie|pension|
minimum vieillesse|rsa|complémentaire|tiers payant|mutualité|régime général|msa|régime agricole|ssi|travailleurs indépendants|
feuille de soins|tarif conventionné|secteur 1|secteur 2|dépassement d'honoraires|forfait hospitalier|affection|pathologie|
anatomie|physiologie|cellule|organe|sang|cœur|poumon|foie|rein|cerveau|grossesse|accouchement|nourrisson|pédiatrie|
gériatrie|psychiatrie|psychologue|dépression|addiction|tabac|alcool|nutrition|obésité|cholestérol|allergie|asthme""".replace("\n", "")
KW_RE = re.compile(r"\b(" + KEYWORDS.lower() + r")\b")
# cheap pre-filter (str.count is C-fast): a document needs at least 2 distinct anchors before the full regex runs
ANCHORS = ("santé", "médec", "maladie", "assur", "patient", "soin", "retraite", "cotis", "hôpital", "traitement",
           "sécurité sociale", "vaccin", "symptôm", "allocation", "handicap", "infect", "chirurg", "cancer", "pension", "mutuelle")


def domain_score(text):
    low = text.lower()
    if sum(1 for a in ANCHORS if a in low) < 2:
        return 0, 0
    hits = [m.group(1) for m in KW_RE.finditer(low)]
    return len(set(hits)), len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wiki", "web"], required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_eval", type=int, default=1500)
    ap.add_argument("--max_docs", type=int, default=None, help="stop after scanning this many documents")
    ap.add_argument("--stop_at", default=None, help="wall-clock HH:MM at which to stop and finish gracefully")
    ap.add_argument("--min_distinct", type=int, default=4, help="distinct domain keywords required")
    ap.add_argument("--min_density", type=float, default=1.0, help="keyword hits per 1000 characters required")
    ap.add_argument("--max_per_doc", type=int, default=8)
    ap.add_argument("--max_tok", type=int, default=32)
    ap.add_argument("--min_tok", type=int, default=4)
    ap.add_argument("--out_dir", default="data/domain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_docs", type=int, default=0, help="skip the first N documents of the stream (documents already used)")
    ap.add_argument("--data_files", default=None, help="web only: parquet path(s) inside the fineweb-2 repo to stream instead of the default order")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    rng = random.Random(args.seed)
    seen = set()
    files = {s: open(os.path.join(args.out_dir, f"{s}.{args.source}.jsonl"), "w") for s in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": args.n_train, "val": args.n_eval, "test": args.n_eval}
    t0, ndocs, kept = time.time(), 0, 0
    for doc in iter_docs(args.source, args.data_files):
        ndocs += 1
        if ndocs <= args.skip_docs:
            continue
        if args.max_docs and ndocs > args.max_docs:
            break
        if args.stop_at and ndocs % 1000 == 0 and time.strftime("%H:%M") >= args.stop_at:
            print(f"{args.source}: stop_at {args.stop_at} reached", flush=True)
            break
        if ndocs % 20000 == 0:
            print(f"{args.source}: scanned={ndocs} kept_docs={kept} {counts} {time.time()-t0:.0f}s", flush=True)
        distinct, hits = domain_score(doc)
        if distinct < args.min_distinct or hits * 1000 / max(1, len(doc)) < args.min_density:
            continue
        kept += 1
        split = "val" if counts["val"] < targets["val"] else "test" if counts["test"] < targets["test"] else "train"
        if counts[split] >= targets[split]:
            break
        # only keep passages that themselves contain at least one domain keyword
        for txt in passages_from_doc(doc, tok, rng, args.max_per_doc, args.max_tok, args.min_tok):
            if txt in seen or not KW_RE.search(txt):
                continue
            seen.add(txt)
            files[split].write(json.dumps({"text": txt, "src": args.source}, ensure_ascii=False) + "\n")
            counts[split] += 1
    for f in files.values():
        f.close()
    print(f"{args.source}: done scanned={ndocs} kept_docs={kept} {counts} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
