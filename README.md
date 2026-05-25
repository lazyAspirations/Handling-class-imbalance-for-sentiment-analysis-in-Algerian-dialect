# Gestion du déséquilibre de classes pour l’analyse de sentiments en dialecte algérien

Ce projet explore et compare plusieurs stratégies de rééquilibrage pour la classification de sentiments sur le corpus **TWIFL** (tweets en dialecte algérien). Le modèle de base est **DziriBERT**, un Transformer pré‑entraîné spécifiquement pour le darija.

## Objectif

Remédier au déséquilibre entre les classes `Positive` (47,7 %), `Negative` (29,5 %) et `Neutral` (22,7 %) dans le corpus TWIFL. L’accent est mis sur l’amélioration de la détection des classes minoritaires (notamment `Neutral`) sans dégrader la performance globale.

## Stratégies évaluées

| Famille | Méthodes |
|---------|----------|
| Modification de la fonction de perte | Class Weighting, Focal Loss (γ=1,2), combinaison des deux |
| Rééquilibrage dans l’espace des embeddings | SMOTE, ADASYN (sur les vecteurs `[CLS]` de DziriBERT) + MLP |
| Augmentation de données | Back‑translation (ar → fr → ar) appliquée uniquement à la classe `Neutral` (taux +20 %, +50 %, +100 %) |

## Résultats principaux

- **Meilleur F1-macro** : Back‑translation +100 % → **0.6989** (gain de +0.0081 par rapport à la baseline).
- **Meilleur F1-Neutral** : Back‑translation +100 % → **0.5379** (contre 0.5254 pour la baseline).
- **Meilleur AUC-PR** : Back‑translation +20 % → **0.7557**.
- **Approche la plus simple et efficace sans modification des données** : Focal Loss γ=2 (F1-macro = 0.6937).
- **Méthodes à éviter sur ce corpus** : SMOTE/ADASYN sur embeddings (F1-macro ≤ 0.6306) – elles dégradent nettement les performances.

## Utilisation

### 1. Installation des dépendances

pip install transformers datasets torch scikit-learn imbalanced-learn sentencepiece

2. Téléchargement du corpus
Le corpus TWIFL est disponible sur HuggingFace :

python
from datasets import load_dataset
dataset = load_dataset("arbml/Twifil")

3. Prétraitement
Exécutez le pipeline décrit dans le rapport :

Suppression des URLs, mentions, doublons

Conversion des émojis en texte (demojize)

Normalisation des répétitions de lettres

Conversion des chiffres arabes orientaux → occidentaux

Conservation du code‑switching (arabe/français)

Suppression des tweets de moins de 2 mots

4. Fine‑tuning de DziriBERT (baseline)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("alger-ia/dziribert", num_labels=3)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    seed=42,
    metric_for_best_model="f1_macro"
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()

5. Appliquer une stratégie
Focal Loss : utilisez FocalLoss personnalisée avec gamma=2.

Back‑translation : utilisez les modèles Helsinki-NLP/opus-mt-ar-fr et opus-mt-fr-ar. Filtrez les paraphrases par similarité cosinus (seuils 0,5–0,85).

SMOTE sur embeddings : extrayez les vecteurs [CLS] avec model.bert(...), appliquez SMOTE(k_neighbors=5), puis entraînez un MLP.

Auteur
Aissat Mohamed Moncef – Master 1 DS & NLP, Université Saad Dahleb Blida 1
Encadrée par Dr. Soraya Cheriguene – Année universitaire 2025/2026
