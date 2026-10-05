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

# Correcteur vec2text pour Qwen3-Embedding-0.6B, textes français ≤ 128 tokens

Ce modèle est le **correcteur** de la chaîne vec2text pour des passages longs (jusqu'à 128 tokens) : à partir
du vecteur cible, du vecteur de l'hypothèse courante, de leur différence (trois jeux de 8 pseudo-tokens) et du
texte de l'hypothèse, il génère une version corrigée, appliquée en boucle avec recherche en faisceau et
re-classement par cosinus. Il s'utilise avec l'inverseur
[`fenyo/vec2text-qwen3-fr-inverter-128`](https://huggingface.co/fenyo/vec2text-qwen3-fr-inverter-128) (profil `long`).

Code, documentation complète (en français) et scripts d'entraînement : **https://github.com/AlexandreFenyo/vec2text-qwen3-fr**.

![Chaîne d'encodage et d'inversion](https://raw.githubusercontent.com/AlexandreFenyo/vec2text-qwen3-fr/main/docs/img/encodage_decodage.png)

## Ce que fait le modèle

Le correcteur reçoit la séquence `[pseudo-tokens de e] [pseudo-tokens de φ(h)] [pseudo-tokens de e − φ(h)] tokens de h <|im_start|>` et génère le texte corrigé ; e est le vecteur cible, h l'hypothèse courante, φ l'encodeur. Les poids du
décodeur (Qwen3-0.6B-Base, 28 couches) sont entièrement réentraînés par descente de gradient sur des paires
(texte, vecteur) fabriquées avec l'encodeur public ; l'encodeur lui-même n'est pas modifié.

## Entraînement

Version v3 : correcteur 128 initial réentraîné 2 époques sur 590 k triplets dont les hypothèses ont été produites
par l'inverseur 128 sur des passages qu'il n'avait jamais vus (dont 80 k du domaine santé / sécurité sociale).
AdamW, lots de 128, 4 h 15 sur une RTX 5090, perte de validation 1,76 → 1,71. Le gain par rapport à la
version précédente est modeste (+1 à +2 BLEU) : à 128 tokens, le correcteur n'est plus le facteur limitant.

## Résultats

| Configuration (200 passages de test ≤ 128 tokens, avec l'inverseur 128) | BLEU | F1 tokens | Exact | Cosinus |
|---|---|---|---|---|
| 1 étape, faisceau 4 | 18,3 | 62,5 % | 6 % | 0,935 |
| 5 étapes, faisceau 4 | 25,2 | 68,7 % | 11,5 % | 0,956 |
| 10 étapes, faisceau 8 (100 passages) | 26,1 | 70,3 % | 16 % | 0,959 |

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
m = Vec2TextQwen.load(snapshot_download("fenyo/vec2text-qwen3-fr-corrector-128"))   # bfloat16, GPU
# inverseur : m.generate(e)  ; correcteur : m.generate(e, e_hyp, hyp_ids)   (voir v2t/infer.py)
```

Fichiers : `model.safetensors` (décodeur, bfloat16), `proj.pt` (projections des vecteurs en pseudo-tokens),
`v2t_config.json` (`n_prefix` = 8, `max_tokens` = 128, `corrector` = true).
Modèle associé : [`fenyo/vec2text-qwen3-fr-inverter-128`](https://huggingface.co/fenyo/vec2text-qwen3-fr-inverter-128).

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
