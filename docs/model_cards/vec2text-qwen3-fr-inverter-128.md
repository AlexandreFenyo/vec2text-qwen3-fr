---
language:
- fr
license: apache-2.0
base_model:
- Qwen/Qwen3-0.6B-Base
tags:
- vec2text
- embedding-inversion
- qwen3
- text-generation
- french
- privacy
pipeline_tag: text-generation
library_name: transformers
---

# Inverseur vec2text pour Qwen3-Embedding-0.6B, textes français ≤ 128 tokens

Ce modèle est l'**inverseur** (étape 0) de la chaîne vec2text pour des passages longs (16 à 128 tokens) : il
reçoit un vecteur `Qwen/Qwen3-Embedding-0.6B` et génère une première hypothèse du texte. Le vecteur est projeté
en 8 pseudo-tokens placés devant un décodeur `Qwen/Qwen3-0.6B-Base` réentraîné. Il s'utilise avec le correcteur
[`fenyo/vec2text-qwen3-fr-corrector-128`](https://huggingface.co/fenyo/vec2text-qwen3-fr-corrector-128) (profil
`long` de `invert.py`). À cette longueur, un vecteur de 1 024 nombres ne détermine plus le texte : on obtient une
paraphrase fidèle au sujet, avec une partie des entités exactes et des valeurs chiffrées généralement fausses.

Code, documentation complète (en français) et scripts d'entraînement : **https://github.com/AlexandreFenyo/vec2text-qwen3-fr**.

![Chaîne d'encodage et d'inversion](https://raw.githubusercontent.com/AlexandreFenyo/vec2text-qwen3-fr/main/docs/img/encodage_decodage.png)

## Ce que fait le modèle

L'inverseur reçoit les pseudo-tokens calculés à partir du vecteur e par un petit réseau (linéaire 1024→1024, GELU, linéaire 1024→n×1024, normalisation) et génère le texte token par token. Les poids du
décodeur (Qwen3-0.6B-Base, 28 couches) sont entièrement réentraînés par descente de gradient sur des paires
(texte, vecteur) fabriquées avec l'encodeur public ; l'encodeur lui-même n'est pas modifié.

## Entraînement

Démarrage à chaud depuis l'inverseur 32 tokens, puis 1 époque sur 600 k passages de 16 à 128 tokens (Wikipédia
FR, FineWeb-2 FR), AdamW, lots de 128, 52 min sur une RTX 5090. Perte de validation 1,84 nats/token (la limite
est l'information contenue dans le vecteur : une seconde époque n'apporte rien).

## Résultats

| Configuration (200 passages de test ≤ 128 tokens) | BLEU | F1 tokens | Exact | Cosinus |
|---|---|---|---|---|
| inverseur seul, faisceau 4, blocage des 6-grammes répétés | 10,6 | 52,5 % | 1,5 % | 0,886 |
| + correcteur, 5 étapes × faisceau 4 | 25,2 | 68,7 % | 11,5 % | 0,956 |
| + correcteur, 10 étapes × faisceau 8 | 26,1 | 70,3 % | 16 % | 0,959 |

BLEU et F1 mesurent le recouvrement avec le texte d'origine ; « exact » est la part de textes reproduits à
l'identique ; le cosinus est la similarité entre le vecteur du texte reconstruit et le vecteur d'entrée.

## Utilisation

Avec l'outil du dépôt (téléchargement automatique des modèles) :

```bash
git clone https://github.com/AlexandreFenyo/vec2text-qwen3-fr.git && cd vec2text-qwen3-fr
.venv/bin/python embed.py --texts phrases.txt --out vecs.tsv --metadata meta.tsv
.venv/bin/python invert.py --vectors vecs.tsv --metadata meta.tsv --profile long --steps 5 --beam 4 --out resultat.tsv
```

Chargement direct en Python (classe `Vec2TextQwen` du dépôt) :

```python
from huggingface_hub import snapshot_download
from v2t.model import Vec2TextQwen
m = Vec2TextQwen.load(snapshot_download("fenyo/vec2text-qwen3-fr-inverter-128"))   # bfloat16, GPU
# inverseur : m.generate(e)  ; correcteur : m.generate(e, e_hyp, hyp_ids)   (voir v2t/infer.py)
```

Fichiers : `model.safetensors` (décodeur, bfloat16), `proj.pt` (projections des vecteurs en pseudo-tokens),
`v2t_config.json` (`n_prefix` = 8, `max_tokens` = 128, `corrector` = false).
Modèle associé : [`fenyo/vec2text-qwen3-fr-corrector-128`](https://huggingface.co/fenyo/vec2text-qwen3-fr-corrector-128).

## Limites et usage responsable

Ce modèle reconstruit du texte à partir de vecteurs d'embeddings ; il sert à **mesurer** le risque de fuite d'une
base vectorielle et à **tester** les protections (dé-identification des textes, rotation secrète des vecteurs),
décrites dans le dépôt. Il ne doit pas être utilisé pour reconstruire des données personnelles à partir de
vecteurs dont on n'est pas légitimement détenteur. Les textes d'exemple du dépôt sont synthétiques. Qualité
mesurée sur du français général et des textes de santé / sécurité sociale ; non évaluée sur d'autres langues ni
sur des vecteurs produits avec une instruction de requête ou tronqués (Matryoshka).

## Référence

J. X. Morris, V. Kuleshov, V. Shmatikov, A. M. Rush. *Text Embeddings Reveal (Almost) As Much As Text*. EMNLP 2023.
arXiv:2310.06816. Les choix d'ingénierie, les mesures et la règle de protection sont décrits dans https://github.com/AlexandreFenyo/vec2text-qwen3-fr.
