import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Mapping of models and their local save paths
models = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "offline_models/pubmed_bert",
    "ProsusAI/finbert": "offline_models/finbert",
    "cardiffnlp/twitter-roberta-base-emotion": "offline_models/twitter_emotion",
    "BAAI/bge-large-en-v1.5": "offline_models/bge_embedding",
    "facebook/bart-large-mnli": "offline_models/bart_large_mnli",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "offline_models/msmarco_cross_encoder"
}

# Download and save each model
for model_name, save_path in models.items():
    os.makedirs(save_path, exist_ok=True)
    print(f"⬇️ Downloading: {model_name} to {save_path}")

    # Use AutoModelForSequenceClassification for classification models
    if model_name in ["facebook/bart-large-mnli", "cross-encoder/ms-marco-MiniLM-L-6-v2"]:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save to local disk
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Saved: {model_name} to {save_path}")

print("\n🎉 All models downloaded and saved offline.")
