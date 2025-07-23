import shutil
import os
import time
import uuid
import pandas as pd
import json
import re
import nltk
import torch
import base64
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever  # Still valid here

from sentence_transformers import CrossEncoder
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import math
import random
from visulize import (
    plot_emotions, 
    plot_hernia_repairs,
    plot_wordcloud,
)

# --- Configuration Management ---
class Config:
    DATASET_PATH_A = "data/patients_100a.json"
    MODEL_SAVE_DIR = "./qol_classifier_fine_tuned"
    CHROMA_DB_DIR = "./chroma_db"
    LOGGING_DIR = "./logs_thematic"
    RESULTS_DIR = "./results_thematic"
    HYPEROPT_LOG_DIR = "./hyperopt_logs"

    THEMATIC_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
    EMOTION_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL_NAME = "llama3:8b-instruct-q8_0"

    # # OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # OLLAMA_MODEL_NAME = "llama3:8b-instruct-q8_0"

    RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    QOL_THEMES = [
        "Symptoms and Function",
        "Body Image",
        "Mental Health",
        "Interpersonal Relationships",
        "Employment/Financial Concerns"
    ]

    LEARNING_RATE = 2e-5
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    NUM_TRAIN_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    CHUNK_SIZE_CLASSIFICATION = 512
    CHUNK_OVERLAP_CLASSIFICATION = 50
    CHUNK_SIZE_SENTIMENT_EMOTION = 512
    CHUNK_OVERLAP_SENTIMENT_EMOTION = 50
    MAX_MODEL_INPUT_LENGTH = 512

    RAG_TOP_K_RETRIEVAL = 10
    RAG_TOP_K_FINAL = 5

    RANDOM_SEED = 42

    N_SPLITS_CV = 3

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

set_seed(Config.RANDOM_SEED)

# --- Logging Setup ---
os.makedirs(Config.LOGGING_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.HYPEROPT_LOG_DIR, exist_ok=True)

log_file_path = os.path.join(Config.LOGGING_DIR, f"patient_qol_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

logger = logging.getLogger(__name__)
logger.info("Configuration and logging setup complete.")

mlb_thematic = MultiLabelBinarizer(classes=Config.QOL_THEMES)
mlb_thematic.fit([Config.QOL_THEMES])
logger.info("Global MultiLabelBinarizer initialized.")

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class PatientDataProcessor:
    def __init__(self, qol_themes: List[str], mlb_thematic_instance: MultiLabelBinarizer, tokenizer: AutoTokenizer):
        self.qol_themes = qol_themes
        self.mlb_thematic = mlb_thematic_instance
        self.tokenizer = tokenizer
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def extract_narrative_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        narrative_fields_keys = [
            "Symptoms_Restrictions_Activities", "Symptoms_Adaptations_Face",
            "Symptoms_Affect_Movement", "Symptoms_Pain_Coping",
            "BodyImage_SelfConscious_Embarrassed", "BodyImage_Others_Noticing",
            "MentalHealth_Affected_Elaborate", "MentalHealth_Coping_Strategies",
            "Relationships_Social_Affected", "Relationships_Intimate_Affected",
            "Employment_What_Do_You_Do", "Employment_Ability_To_Work_Affected",
            "Employment_Changed_Work", "Employment_Financial_Affected",
            "SharedDecisionMaking_Questions", "SharedDecisionMaking_Hopes",
            "SharedDecisionMaking_Matters_To_You"
        ]
        for field in narrative_fields_keys:
            df[field] = df['Narratives'].apply(
                lambda x: str(x.get(field, '').replace('N/A', '').strip()) if isinstance(x, dict) else ''
            )
        # existing_narrative_cols = [col for col in narrative_fields_keys if col in df.columns]
        # df['combined_narrative'] = df[existing_narrative_cols].apply(
        #     lambda row: ' '.join(filter(None, row.values.astype(str))).strip(), axis=1
        # )
        # df['combined_narrative'] = df['combined_narrative'].str.replace(r'\\s+', ' ', regex=True).fillna('')
        # logger.info("Combined narrative created from relevant fields.")
        # return df
        df['combined_narrative'] = df[narrative_fields_keys].apply(
            lambda row: ' '.join(filter(None, row.values.astype(str))).strip(), axis=1
        )
        df['combined_narrative'] = df['combined_narrative'].str.replace(r'\s+', ' ', regex=True).fillna('')
        return df

    def preprocess_narrative_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        words = [self.lemmatizer.lemmatize(word) for word in text.split()]
        text = ' '.join(words)
        return text

    # def assign_robust_multi_labels(self, row: pd.Series) -> List[str]:
    #     labels_for_this_row = []
    #     # narrative_lower = row['combined_narrative'].lower()
    #     narrative_lower = row['combined_narrative'].lower()

    #     symptom_keywords = [
    #         "pain", "ache", "discomfort", "pulling", "stiffness", "fragile", "limited", "movement",
    #         "bending", "lifting", "walking", "sitting", "standing", "sleep", "exhausting",
    #         "struggle", "strain", "wince", "jolt", "burden", "mobility", "physical", "fatigue",
    #         "nausea", "swelling", "shortness of breath", "cramp", "soreness", "weakness",
    #         "vomiting", "dizzy", "coughs", "breathing", "tightness", "headache", "migraine",
    #         "fever", "aching", "cramping", "stiff", "tired", "debilitating", "debilitated"
    #     ]
    #     body_image_keywords = [
    #         "self-conscious", "embarrassed", "looks", "bulge", "disfigured", "ruins my figure",
    #         "hiding something", "pregnant", "ashamed", "unattractive", "scrutinized", "appearance",
    #         "scar", "figure", "deformity", "visible", "unappealing", "misshapen", "ugly",
    #         "self image", "confidence", "blemish", "complexion", "features"
    #     ]
    #     mental_health_keywords = [
    #         "anxious", "worry", "fear", "dread", "nervous", "stressed", "panic", "hopeless",
    #         "pointless", "trapped", "despair", "low", "depressed", "sad", "down", "miserable",
    #         "unhappy", "frustrated", "irritable", "snap", "overwhelmed", "consumed", "draining",
    #         "mental health", "mood", "rumination", "isolated", "withdrawn", "guilt",
    #         "emotional eating", "distraction", "volatile", "anger", "stress", "crying", "agitated",
    #         "moody", "anxiety", "depression", "PTSD", "trauma", "coping", "therapy", "counselling",
    #         "depressive", "panic attacks", "insomnia", "suicidal", "self harm"
    #     ]
    #     relationship_keywords = [
    #         "social", "friends", "partner", "intimacy", "isolated", "withdrawn", "strain", "guilt",
    #         "awkward", "less desirable", "spontaneity", "relationships", "dating", "sexual",
    #         "family", "loved one", "connection", "communication", "support", "depend", "burden",
    #         "marriage", "divorce", "friendship", "social life", "close"
    #     ]
    #     employment_keywords = [
    #         "job", "work", "employment", "financial", "income", "bills", "livelihood", "unemployed",
    #         "shifts", "productive", "career", "salary", "pay", "money", "redundancy", "debt",
    #         "cost", "expenditure", "workplace", "colleague", "promotion", "leave", "disability",
    #         "benefits", "earning", "budget", "expenses"
    #     ]

    #     if any(kw in narrative_lower for kw in symptom_keywords) or \
    #     any(row[f] and row[f] != '' for f in ["Symptoms_Restrictions_Activities", "Symptoms_Adaptations_Face", "Symptoms_Affect_Movement", "Symptoms_Pain_Coping"]):
    #         labels_for_this_row.append("Symptoms and Function")
    #     if any(kw in narrative_lower for kw in body_image_keywords) or \
    #     any(row[f] and row[f] != '' for f in ["BodyImage_SelfConscious_Embarrassed", "BodyImage_Others_Noticing"]):
    #         labels_for_this_row.append("Body Image")
    #     if any(kw in narrative_lower for kw in mental_health_keywords) or \
    #     any(row[f] and row[f] != '' for f in ["MentalHealth_Affected_Elaborate", "MentalHealth_Coping_Strategies"]):
    #         labels_for_this_row.append("Mental Health")
    #     if any(kw in narrative_lower for kw in relationship_keywords) or \
    #     any(row[f] and row[f] != '' for f in ["Relationships_Social_Affected", "Relationships_Intimate_Affected"]):
    #         labels_for_this_row.append("Interpersonal Relationships")
    #     if any(kw in narrative_lower for kw in employment_keywords) or \
    #     any(row[f] and row[f] != '' for f in ["Employment_What_Do_You_Do", "Employment_Ability_To_Work_Affected", "Employment_Changed_Work", "Employment_Financial_Affected"]):
    #         labels_for_this_row.append("Employment/Financial Concerns")

    #     if not labels_for_this_row and narrative_lower.strip() != '':
    #         labels_for_this_row.append("Symptoms and Function")
    #     return labels_for_this_row

    def assign_robust_multi_labels(self, row: pd.Series) -> List[str]:
        labels = []
        narrative = row.get("combined_narrative", "").lower()

        def keyword_match(keywords: List[str]) -> bool:
            return bool(set(narrative.split()) & set(keywords))

        def has_field_data(fields: List[str]) -> bool:
            return any(str(row.get(f, "")).strip() for f in fields)

        if keyword_match([
            "pain", "ache", "discomfort", "pulling", "stiffness", "fragile", "limited", "movement",
            "bending", "lifting", "walking", "sitting", "standing", "sleep", "exhausting",
            "struggle", "strain", "wince", "jolt", "burden", "mobility", "physical", "fatigue",
            "nausea", "swelling", "shortness of breath", "cramp", "soreness", "weakness",
            "vomiting", "dizzy", "coughs", "breathing", "tightness", "headache", "migraine",
            "fever", "aching", "cramping", "stiff", "tired", "debilitating", "debilitated"
        ]) or has_field_data(["Symptoms_Restrictions_Activities", "Symptoms_Adaptations_Face",
                            "Symptoms_Affect_Movement", "Symptoms_Pain_Coping"]):
            labels.append("Symptoms and Function")

        if keyword_match([
            "self-conscious", "embarrassed", "looks", "bulge", "disfigured", "ruins my figure",
            "hiding something", "pregnant", "ashamed", "unattractive", "scrutinized", "appearance",
            "scar", "figure", "deformity", "visible", "unappealing", "misshapen", "ugly",
            "self image", "confidence", "blemish", "complexion", "features"
        ]) or has_field_data(["BodyImage_SelfConscious_Embarrassed", "BodyImage_Others_Noticing"]):
            labels.append("Body Image")

        if keyword_match([
            "anxious", "worry", "fear", "dread", "nervous", "stressed", "panic", "hopeless",
            "pointless", "trapped", "despair", "low", "depressed", "sad", "down", "miserable",
            "unhappy", "frustrated", "irritable", "snap", "overwhelmed", "consumed", "draining",
            "mental health", "mood", "rumination", "isolated", "withdrawn", "guilt",
            "emotional eating", "distraction", "volatile", "anger", "stress", "crying", "agitated",
            "moody", "anxiety", "depression", "PTSD", "trauma", "coping", "therapy", "counselling",
            "depressive", "panic attacks", "insomnia", "suicidal", "self harm"
        ]) or has_field_data(["MentalHealth_Affected_Elaborate", "MentalHealth_Coping_Strategies"]):
            labels.append("Mental Health")

        if keyword_match([
            "social", "friends", "partner", "intimacy", "isolated", "withdrawn", "strain", "guilt",
            "awkward", "less desirable", "spontaneity", "relationships", "dating", "sexual",
            "family", "loved one", "connection", "communication", "support", "depend", "burden",
            "marriage", "divorce", "friendship", "social life", "close"
        ]) or has_field_data(["Relationships_Social_Affected", "Relationships_Intimate_Affected"]):
            labels.append("Interpersonal Relationships")

        if keyword_match([
            "job", "work", "employment", "financial", "income", "bills", "livelihood", "unemployed",
            "shifts", "productive", "career", "salary", "pay", "money", "redundancy", "debt",
            "cost", "expenditure", "workplace", "colleague", "promotion", "leave", "disability",
            "benefits", "earning", "budget", "expenses"
        ]) or has_field_data(["Employment_What_Do_You_Do", "Employment_Ability_To_Work_Affected",
                            "Employment_Changed_Work", "Employment_Financial_Affected"]):
            labels.append("Employment/Financial Concerns")

        if not labels and narrative.strip():
            labels.append("Symptoms and Function")

        return labels


    # def _prepare_dataframe_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df['preprocessed_narrative'] = df['Narratives'].apply(self.preprocess_narrative_text)
    #     df['labels_as_names'] = df.apply(self.assign_robust_multi_labels, axis=1)
    #     df['labels_encoded'] = list(self.mlb_thematic.transform(df['labels_as_names']))
    #     df['labels_as_indices'] = df['labels_as_names'].apply(lambda names: [self.qol_themes.index(name) for name in names])
    #     df['dominant_label_id'] = df.apply(
    #         lambda row: random.choice(row['labels_as_indices']) if row['labels_as_indices'] else random.randint(0, len(self.qol_themes) - 1),
    #         axis=1
    #     )
    #     return df

    def _prepare_dataframe_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        # One-row DataFrame
        df = df.copy()
        df['preprocessed_narrative'] = df['combined_narrative'].apply(self.preprocess_narrative_text)
        df['labels_as_names'] = df.apply(self.assign_robust_multi_labels, axis=1)
        df['labels_encoded'] = list(self.mlb_thematic.transform(df['labels_as_names']))
        df['labels_as_indices'] = df['labels_as_names'].apply(
            lambda names: [self.qol_themes.index(name) for name in names]
        )
        df['dominant_label_id'] = df['labels_as_indices'].apply(
            lambda indices: random.choice(indices) if indices else random.randint(0, len(self.qol_themes) - 1)
        )
        return df


    # def prepare_data_for_classification(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dataset, Dataset, DataCollatorWithPadding]:
    #     df_processed = self._prepare_dataframe_for_classification(df.copy())
    #     logger.info("Narratives preprocessed and labels assigned for main dataset.")
    #     logger.info(f"Example multi-labels for first patient: {df_processed['labels_encoded'].iloc[0]}")
    #     logger.info(f"Class distribution of 'dominant_label_id' before oversampling: {Counter(df_processed['dominant_label_id'])}")

    #     text_splitter_for_classification = RecursiveCharacterTextSplitter(
    #         chunk_size=Config.CHUNK_SIZE_CLASSIFICATION,
    #         chunk_overlap=Config.CHUNK_OVERLAP_CLASSIFICATION,
    #         length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=True)),
    #     )

    #     split_documents_for_classification = []
    #     for index, row in df_processed.iterrows():
    #         original_narrative = row['preprocessed_narrative']
    #         if not original_narrative.strip():
    #             logger.debug(f"Skipping empty narrative for patient ID: {row['Patient_ID']}")
    #             continue
    #         chunks = text_splitter_for_classification.split_text(original_narrative)
    #         if not chunks:
    #             chunks = [original_narrative]
    #         for i, chunk in enumerate(chunks):
    #             split_documents_for_classification.append({
    #                 'preprocessed_narrative': chunk,
    #                 'labels': row['labels_encoded'],
    #                 'patient_id': row['Patient_ID'],
    #                 'chunk_index': i,
    #                 'dominant_label_id': row['dominant_label_id']
    #             })

    #     split_df_for_classification = pd.DataFrame(split_documents_for_classification)
    #     logger.info(f"Original number of narratives: {len(df_processed)}")
    #     logger.info(f"Number of chunks for classification after splitting: {len(split_df_for_classification)}")

    #     X_resample = split_df_for_classification['preprocessed_narrative'].tolist()
    #     y_resample = split_df_for_classification['dominant_label_id'].tolist()

    #     ros = RandomOverSampler(random_state=Config.RANDOM_SEED)
    #     resampled_indices_flat, y_resampled_labels_single = ros.fit_resample(
    #         np.arange(len(X_resample)).reshape(-1, 1),
    #         np.array(y_resample)
    #     )
    #     resampled_df = split_df_for_classification.iloc[resampled_indices_flat.flatten()].copy()

    #     logger.info(f"Class distribution after oversampling: {Counter(y_resampled_labels_single)}")
    #     logger.info(f"Number of samples after oversampling: {len(resampled_df)}")

    #     train_dataset_hf = Dataset.from_pandas(resampled_df[['preprocessed_narrative', 'labels']].astype({'labels': object}))
    #     eval_dataset_hf = Dataset.from_pandas(split_df_for_classification[['preprocessed_narrative', 'labels']].astype({'labels': object}))

    #     def tokenize_function(examples):
    #         return self.tokenizer(examples["preprocessed_narrative"], truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)

    #     train_dataset = train_dataset_hf.map(tokenize_function, batched=True, remove_columns=['preprocessed_narrative'], num_proc=os.cpu_count() if os.cpu_count() else 1)
    #     eval_dataset = eval_dataset_hf.map(tokenize_function, batched=True, remove_columns=['preprocessed_narrative'], num_proc=os.cpu_count() if os.cpu_count() else 1)
    #     logger.info("Datasets tokenized for classification.")

    #     class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    #         def __call__(self, features):
    #             labels = [feature.pop('labels') for feature in features]
    #             batch = super().__call__(features)
    #             batch['labels'] = torch.tensor(labels, dtype=torch.float32)
    #             return batch

    #     data_collator = CustomDataCollatorWithPadding(tokenizer=self.tokenizer)
    #     logger.info("CustomDataCollatorWithPadding created.")
    #     return df_processed, train_dataset, eval_dataset, data_collator

    def prepare_data_for_classification(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dataset, DataCollatorWithPadding]:
        df_processed = self._prepare_dataframe_for_classification(df)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE_CLASSIFICATION,
            chunk_overlap=Config.CHUNK_OVERLAP_CLASSIFICATION,
            length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=True)),
        )

        chunks_data = []
        for _, row in df_processed.iterrows():
            chunks = text_splitter.split_text(row['preprocessed_narrative']) or [row['preprocessed_narrative']]
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    'preprocessed_narrative': chunk,
                    'labels': row['labels_encoded'],
                    'patient_id': row['Patient_ID'],
                    'chunk_index': i,
                    'dominant_label_id': row['dominant_label_id']
                })

        chunk_df = pd.DataFrame(chunks_data)

        hf_dataset = Dataset.from_pandas(chunk_df[['preprocessed_narrative', 'labels']].astype({'labels': object}))

        def tokenize_function(example):
            return self.tokenizer(example["preprocessed_narrative"], truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=['preprocessed_narrative'])

        class CustomDataCollatorWithPadding(DataCollatorWithPadding):
            def __call__(self, features):
                labels = [feature.pop('labels') for feature in features]
                batch = super().__call__(features)
                batch['labels'] = torch.tensor(labels, dtype=torch.float32)
                return batch

        data_collator = CustomDataCollatorWithPadding(tokenizer=self.tokenizer)

        return df_processed, tokenized_dataset, data_collator

    
class NLPAnalysisService:
    def __init__(self, thematic_tokenizer_instance: AutoTokenizer, sentiment_model_name: str, emotion_model_name: str, sentence_embedder: Any = None):
        self.thematic_tokenizer = thematic_tokenizer_instance
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice set for sentiment/emotion analysis: {self.device}")

        logger.info(f"Loading sentiment analysis model: {sentiment_model_name} on device: {self.device}")
        self.sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model_name, device=self.device)

        logger.info(f"Loading emotion classification model: {emotion_model_name} on device: {self.device}")
        try:
            self.emotion_analyzer = pipeline('sentiment-analysis', model=emotion_model_name, return_all_scores=True, device=self.device)
            print(f"Emotion analysis pipeline '{emotion_model_name}' loaded successfully.")
        except Exception as e:
            logger.warning(f"Error loading emotion analysis model: {e}. Falling back to basic sentiment for emotion analysis.")
            self.emotion_analyzer = self.sentiment_analyzer

        self.sentiment_emotion_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE_SENTIMENT_EMOTION,
            chunk_overlap=Config.CHUNK_OVERLAP_SENTIMENT_EMOTION,
            length_function=lambda t: len(self.thematic_tokenizer.encode(t, add_special_tokens=True)),
        )
        self.sentence_embedder = sentence_embedder

        logger.info(f"Loading zero-shot classification model: facebook/bart-large-mnli on device: {self.device}")
        self.mental_health_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=self.device)
        self.mental_health_candidate_labels = [
            "anxiety", "hopelessness", "social withdrawal", "frustration or irritability",
            "low mood or depression", "feeling overwhelmed", "rumination"
        ]

    def analyze_sentiment_and_emotions_of_chunks(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str) or text.strip() == "":
            return {"overall_sentiment": {"label": "NEUTRAL", "score": 0.0}, "emotions": {}, "sentiment_details_per_chunk": []}

        chunks = self.sentiment_emotion_text_splitter.split_text(text)
        if not chunks: chunks = [text]

        sentiment_results_per_chunk = []
        emotion_results_per_chunk = []

        for chunk in chunks:
            chunk_tokens = self.thematic_tokenizer.encode(chunk, truncation=True, max_length=Config.MAX_MODEL_INPUT_LENGTH)
            truncated_chunk = self.thematic_tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            if truncated_chunk.strip():
                sentiment_result = self.sentiment_analyzer(truncated_chunk)
                if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                    sentiment_results_per_chunk.append(sentiment_result[0])
                else:
                    logger.warning(f"Sentiment analysis returned unexpected format for chunk: {truncated_chunk[:50]}...")

                emotion_output = self.emotion_analyzer(truncated_chunk)
                if emotion_output and isinstance(emotion_output, list) and len(emotion_output) > 0:
                    if isinstance(emotion_output[0], list) and len(emotion_output[0]) > 0 and isinstance(emotion_output[0][0], dict):
                        emotion_results_per_chunk.append(emotion_output[0])
                    elif isinstance(emotion_output[0], dict):
                        emotion_results_per_chunk.append(emotion_output)
                    else:
                        logger.warning(f"Emotion analysis returned unexpected structure within list: {emotion_output}")
                        emotion_results_per_chunk.append([])
                else:
                    logger.warning(f"Emotion analysis returned unexpected format for chunk: {truncated_chunk[:50]}...")
                    emotion_results_per_chunk.append([])

        overall_sentiment_label = "NEUTRAL"
        overall_sentiment_score = 0.0
        if sentiment_results_per_chunk:
            positive_scores = [s['score'] for s in sentiment_results_per_chunk if s['label'] == 'POSITIVE']
            negative_scores = [s['score'] for s in sentiment_results_per_chunk if s['label'] == 'NEGATIVE']
            avg_pos_score = np.mean(positive_scores) if positive_scores else 0
            avg_neg_score = np.mean(negative_scores) if negative_scores else 0

            if avg_pos_score > avg_neg_score + 0.1 and avg_pos_score > 0.5:
                overall_sentiment_label = "POSITIVE"
                overall_sentiment_score = avg_pos_score
            elif avg_neg_score > avg_pos_score + 0.1 and avg_neg_score > 0.5:
                overall_sentiment_label = "NEGATIVE"
                overall_sentiment_score = avg_neg_score
            else:
                most_confident_sentiment = max(sentiment_results_per_chunk, key=lambda x: x['score'], default={'label': 'NEUTRAL', 'score': 0.0})
                overall_sentiment_label = most_confident_sentiment['label']
                overall_sentiment_score = most_confident_sentiment['score']

        aggregated_emotions = {}
        valid_emotion_chunks_count = 0
        for chunk_emotions_list in emotion_results_per_chunk:
            if chunk_emotions_list:
                valid_emotion_chunks_count += 1
                for emotion_dict in chunk_emotions_list:
                    if isinstance(emotion_dict, dict) and 'label' in emotion_dict and 'score' in emotion_dict:
                        label = emotion_dict['label']
                        score = emotion_dict['score']
                        aggregated_emotions[label] = aggregated_emotions.get(label, 0.0) + score
                    else:
                        logger.warning(f"Expected dictionary inside chunk_emotions_list but got type {type(emotion_dict)}: {emotion_dict}. Skipping.")

        if valid_emotion_chunks_count > 0 and aggregated_emotions:
            for label in aggregated_emotions:
                aggregated_emotions[label] /= valid_emotion_chunks_count
        sorted_emotions = dict(sorted(aggregated_emotions.items(), key=lambda item: item[1], reverse=True))

        return {
            "overall_sentiment": {"label": overall_sentiment_label, "score": overall_sentiment_score},
            "emotions": sorted_emotions,
            "sentiment_details_per_chunk": sentiment_results_per_chunk
        }

    def identify_mental_health_patterns_advanced(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str) or text.strip() == "":
            return {"detected_patterns": [], "details": {}}
        zero_shot_result = self.mental_health_classifier(text, self.mental_health_candidate_labels, multi_label=True)
        detected_patterns = {
            "detected_patterns": [
                label for label, score in zip(zero_shot_result['labels'], zero_shot_result['scores']) if score > 0.5
            ],
            "details": {
                label: score for label, score in zip(zero_shot_result['labels'], zero_shot_result['scores'])
            }
        }
        return detected_patterns
    
# class RAGSystem:
#     def __init__(self, embedding_model_name: str, chroma_db_dir: str, ollama_base_url: str, ollama_model_name: str, thematic_tokenizer_instance: AutoTokenizer, reranker_model_name: str):
#     # def __init__(self, embedding_model_name: str, chroma_db_dir: str, ollama_model_name: str, thematic_tokenizer_instance: AutoTokenizer, reranker_model_name: str):
#         try:
#             self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
#             logger.info(f"Embedding function loaded: {embedding_model_name}")
#         except Exception as e:
#             logger.error(f"FATAL ERROR: Could not load embedding model {embedding_model_name}: {e}. Ensure model exists and network is stable.")
#             self.embedding_function = None

#         self.chroma_db_dir = chroma_db_dir
#         self.ollama_llm = Ollama(model=ollama_model_name, base_url=ollama_base_url)
#         # self.ollama_llm = Ollama(model=ollama_model_name)
#         self.thematic_tokenizer = thematic_tokenizer_instance
#         self.vectorstore = None

#         try:
#             self.reranker = CrossEncoder(reranker_model_name)
#             logger.info(f"Reranker model loaded: {reranker_model_name}")
#         except Exception as e:
#             logger.warning(f"Could not load reranker model {reranker_model_name}: {e}. Reranking will be skipped.")
#             self.reranker = None

#         self._initialize_vectorstore()

#     def _initialize_vectorstore(self):
#         """Initializes or loads the ChromaDB vector store."""
#         os.makedirs(self.chroma_db_dir, exist_ok=True)

#         if self.embedding_function is None:
#             logger.error("ChromaDB initialization skipped: Embedding function failed to load.")
#             self.vectorstore = None
#             return

#         try:
#             self.vectorstore = Chroma(persist_directory=self.chroma_db_dir, embedding_function=self.embedding_function)

#             if self.vectorstore._collection.count() == 0:
#                 logger.warning("ChromaDB collection is empty at path or just created. It will be populated when documents are added.")
#             else:
#                 logger.info(f"Loaded existing ChromaDB from {self.chroma_db_dir} with {self.vectorstore._collection.count()} documents.")
#         except Exception as e:
#             logger.error(f"FATAL ERROR: Could not initialize or load ChromaDB at {self.chroma_db_dir}. Details: {e}")
#             logger.error(f"This might be due to issues with the embedding model, FAISS, disk permissions, or corrupted existing database files.")
#             self.vectorstore = None

#         try:
#             if self.ollama_llm:
#                 logger.info(f"Attempting to connect to Ollama at: {self.ollama_llm.base_url}")
#                 test_response = self.ollama_llm.invoke("Hi", temperature=0, timeout=10)
#                 logger.info(f"Ollama connection successful. Test response: {test_response[:50]}...")
#             else:
#                 logger.warning("Ollama LLM object was not initialized, skipping connection test.")
#         except Exception as e:
#             logger.error(f"FAILED TO CONNECT TO OLLAMA at {self.ollama_llm.base_url if self.ollama_llm else 'N/A'}. Details: {e}")
#             logger.error("Ensure Ollama is running, the model is pulled, and ngrok (or direct access) is correctly configured.")
#             self.ollama_llm = None

#     def populate_vectorstore(self, documents: List[Document]):
#         """Adds documents to the vector store in batches."""
#         if self.vectorstore is None:
#             logger.error("Vectorstore not initialized. Cannot add documents. Check prior initialization errors.")
#             return

#         logger.info(f"Preparing to add {len(documents)} documents to ChromaDB.")
#         if not documents:
#             logger.warning("No documents provided to populate ChromaDB. Skipping population.")
#             return

#         try:
#             if self.vectorstore._collection.count() > 0:
#                 logger.info("Clearing existing ChromaDB collection before adding new documents.")
#                 existing_ids = self.vectorstore._collection.get()['ids']
#                 if existing_ids:
#                     self.vectorstore._collection.delete(ids=existing_ids)
#         except Exception as e:
#             logger.warning(f"Could not clear existing ChromaDB collection (might be empty or corrupted): {e}")

#         MAX_BATCH_SIZE = 5461

#         num_documents = len(documents)
#         if num_documents > MAX_BATCH_SIZE:
#             logger.info(f"Documents exceed max batch size ({MAX_BATCH_SIZE}). Splitting into batches.")
#             num_batches = math.ceil(num_documents / MAX_BATCH_SIZE)
#             for i in range(num_batches):
#                 start_index = i * MAX_BATCH_SIZE
#                 end_index = min((i + 1) * MAX_BATCH_SIZE, num_documents)
#                 batch = documents[start_index:end_index]
#                 try:
#                     self.vectorstore.add_documents(batch)
#                     logger.info(f"Added batch {i+1}/{num_batches} ({len(batch)} documents).\n")
#                 except Exception as e:
#                     logger.error(f"Error adding batch {i+1} to ChromaDB: {e}")
#                     logger.error("Vectorstore population failed for this batch. Check documents or vectorstore state.")
#                     return
#             logger.info(f"ChromaDB populated and persisted with all {num_documents} documents in batches.\n")
#         else:
#             try:
#                 self.vectorstore.add_documents(documents)
#                 logger.info(f"ChromaDB populated and persisted with {len(documents)} documents.\n")
#             except Exception as e:
#                 logger.error(f"Error adding documents to ChromaDB: {e}. Vectorstore population failed. This might indicate an issue with the documents themselves or the vectorstore state.\n")

#     def _get_rag_prompt_template(self) -> str:
#         """
#         Returns the refined RAG prompt template for structured output.
#         Emphasizes direct extraction from context and strict adherence to presence/absence.
#         CRITICAL CHANGE: Added emphasis on preserving original formatting and spaces.
#         """
#         return """You are an AI assistant for healthcare professionals, analyzing patient narratives.
#         Your task is to synthesize information from the provided patient narrative excerpts to generate a concise, empathetic, and structured summary of the patient's Quality of Life (QoL) concerns.

#         **CRITICAL INSTRUCTIONS - PLEASE READ CAREFULLY:**
#         1.  **STRICTLY USE ONLY THE PROVIDED \"Retrieved Patient Narrative Excerpts\".**
#             * **DO NOT INFER, GUESS, OR ADD EXTERNAL INFORMATION.**
#             * **DO NOT FABRICATE OR HALLUCINATE TEXT.** Every piece of information must come directly from the excerpts.
#             * **NEVER INCLUDE \"Document ID\", \"page_content\", or any raw, truncated text from the excerpts.** Summarize the content, do not quote internal processing details.
#             * **ENSURE ALL INFORMATION DIRECTLY REFLECTS THE PATIENT'S NARRATIVE.** Do not skew or rephrase to introduce new meanings.
#         2.  For each of the five QoL themes below, **EXTRACT AND SUMMARIZE SPECIFIC DETAILS, EXAMPLES, AND KEY PHRASES DIRECTLY FROM THE NARRATIVE.**
#             * **IMPORTANT: When quoting directly, preserve original wording, spacing, and punctuation.**
#             * Aim to use the patient's own words or very close paraphrases where possible, but ensure the output is a coherent summary, not just a list of raw phrases.
#             * Be concise and factual.
#         3.  If a theme is genuinely NOT mentioned or strongly inferable *from the provided excerpts*, and you cannot find *any relevant information within the excerpts for that specific theme*, then and *only then*, state: \"Not explicitly mentioned in narrative for this theme.\"
#             * Do NOT invent or assume absence if information is present.
#             * If a theme *is* mentioned, provide the details; do not state \"Not explicitly mentioned\".
#         4.  Maintain a professional and objective tone throughout the summary.
#         5.  Do not include outside information or recommendations on the patient's situation. Your sole responsibility is to accurately summarize the patient's QoL themes based *only* on the provided text.

#         **Themes to Address in the Summary (Extract specific details and examples from context):**
#         * **Physical Symptoms and Functional Limitations**: Describe any pain, discomfort, physical restrictions, or impact on daily activities (e.g., movement, sleep, work tasks) *as explicitly stated in the narrative excerpts*. Provide concrete examples if available.
#         * **Body Image Concerns**: Detail any self-consciousness, embarrassment, appearance concerns, or feelings about their body due to the condition (e.g., hernia, scars) *as explicitly stated in the narrative excerpts*. Provide concrete examples if available.
#         * **Mental Health Challenges**: Summarize emotional states such as anxiety, worry, fear, stress, frustration, low mood, depression, feelings of being overwhelmed, or social isolation *as explicitly described in the narrative excerpts*. Provide concrete examples if available.
#         * **Impact on Interpersonal Relationships**: Explain effects on social interactions, friendships, family dynamics, or intimate relationships *as explicitly described in the narrative excerpts*. Provide concrete examples if available.
#         * **Employment and Financial Concerns**: Cover issues related to work ability, job changes, financial strain, income impact, or career concerns *as explicitly described in the narrative excerpts*. Provide concrete examples if available.

#         ---
#         Retrieved Patient Narrative Excerpts:
#         {context}
#         ---

#         Patient's Overall Quality of Life Summary (structured by themes):
#         """

#     def _get_patient_specific_dense_docs(self, query_text: str, patient_id_filter: str) -> List[Document]:
#         """
#         Helper method to perform dense retrieval (similarity search) for a specific patient.
#         This method is designed to be called by a RunnableLambda.
#         """
#         if self.vectorstore is None:
#             logger.error("Vectorstore not initialized in _get_patient_specific_dense_docs.")
#             return []
#         try:
#             results_with_scores = self.vectorstore.similarity_search_with_score(
#                 query=query_text,
#                 k=Config.RAG_TOP_K_RETRIEVAL,
#                 filter={"patient_id": patient_id_filter}
#             )
#             return [doc for doc, _score in results_with_scores]
#         except Exception as e:
#             logger.error(f"Error during patient-specific dense retrieval for patient {patient_id_filter}: {e}")
#             return []

#     def get_rag_summary_for_patient(self, patient_id: str, query: str) -> str:
#         """Generates a RAG summary for a specific patient by filtering retrieval."""
#         if self.vectorstore is None:
#             logger.error("Vectorstore not initialized. Cannot generate patient-specific RAG summary.")
#             return "Error: RAG system not ready."
#         if self.ollama_llm is None:
#             logger.error("Ollama LLM not initialized or connected. Cannot generate patient-specific RAG summary.")
#             return "Error: LLM not available."

#         global documents_for_vectorstore
#         patient_docs = [doc for doc in documents_for_vectorstore if doc.metadata.get("patient_id") == patient_id]

#         if not patient_docs:
#             logger.warning(f"No documents found for patient ID: {patient_id}. Cannot generate RAG summary.")
#             return f"No relevant narrative excerpts found for patient {patient_id} to generate a RAG summary."

#         patient_texts = [doc.page_content for doc in patient_docs]
#         patient_metadatas = [doc.metadata for doc in patient_docs]

#         bm25_retriever = BM25Retriever.from_texts(texts=patient_texts, metadatas=patient_metadatas)
#         bm25_retriever.k = Config.RAG_TOP_K_RETRIEVAL

#         dense_retriever_runnable = RunnableLambda(
#             lambda q: self._get_patient_specific_dense_docs(q, patient_id)
#         )

#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[bm25_retriever, dense_retriever_runnable],
#             weights=[0.5, 0.5]
#         )
#         logger.info(f"Ensemble (Hybrid) Retriever configured for patient: {patient_id}.")

#         def _rerank_documents_local(retrieved_docs: List[Document], current_query: str) -> List[Document]:
#             if not self.reranker:
#                 logger.warning("Reranker not available. Skipping re-ranking.")
#                 return retrieved_docs[:Config.RAG_TOP_K_FINAL]

#             if not retrieved_docs:
#                 return []

#             sentence_pairs = [(current_query, doc.page_content) for doc in retrieved_docs]
#             scores = self.reranker.predict(sentence_pairs)

#             ranked_indices = np.argsort(scores)[::-1]
#             ranked_docs = [retrieved_docs[i] for i in ranked_indices]

#             logger.info(f"Re-ranked {len(retrieved_docs)} documents, returning top {Config.RAG_TOP_K_FINAL}.")
#             return ranked_docs[:Config.RAG_TOP_K_FINAL]

#         rag_chain_for_patient = (
#             {"context": ensemble_retriever, "question": RunnablePassthrough()}
#             | RunnablePassthrough.assign(context=RunnableLambda(lambda x: _rerank_documents_local(x["context"], x["question"])))\
#             | ChatPromptTemplate.from_template(self._get_rag_prompt_template())\
#             | self.ollama_llm\
#             | StrOutputParser()
#         )

#         try:
#             response = rag_chain_for_patient.invoke(query)
#             return response
#         except Exception as e:
#             logger.error(f"Error invoking RAG chain for patient {patient_id}: {e}")
#             return f"Error generating summary for patient {patient_id}: {e}"

# --- pipeline.py ---

class RAGSystem:
    def __init__(self, embedding_model_name, chroma_db_dir, ollama_base_url, ollama_model_name, thematic_tokenizer_instance, reranker_model_name):
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.chroma_db_dir = chroma_db_dir
        self.ollama_llm = Ollama(model=ollama_model_name, base_url=ollama_base_url)
        self.thematic_tokenizer = thematic_tokenizer_instance
        self.vectorstore = None

        try:
            self.reranker = CrossEncoder(reranker_model_name)
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self.reranker = None

        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        try:
            self.vectorstore = Chroma(persist_directory=self.chroma_db_dir, embedding_function=self.embedding_function)
            logger.info("ChromaDB initialized.")
        except Exception as e:
            logger.error(f"Chroma init failed: {e}")
            self.vectorstore = None

        try:
            response = self.ollama_llm.invoke("Hi")
            logger.info(f"Ollama connected: {response[:50]}")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            self.ollama_llm = None

    def populate_vectorstore(self, documents: List[Document]):
        if self.vectorstore is None:
            logger.error("Vectorstore is not initialized.")
            return
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"{len(documents)} documents added to vectorstore.")
        except Exception as e:
            logger.error(f"Vectorstore population failed: {e}")

    def _get_patient_specific_dense_docs(self, query_text: str, patient_id_filter: str) -> List[Document]:
        if self.vectorstore is None:
            return []
        try:
            results = self.vectorstore.similarity_search_with_score(query=query_text, k=5, filter={"patient_id": patient_id_filter})
            return [doc for doc, _ in results]
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    def _get_rag_prompt_template(self) -> str:
        """
        Returns the refined RAG prompt template for structured output.
        Emphasizes direct extraction from context and strict adherence to presence/absence.
        CRITICAL CHANGE: Added emphasis on preserving original formatting and spaces.
        """
        return """You are an AI assistant for healthcare professionals, analyzing patient narratives.
        Your task is to synthesize information from the provided patient narrative excerpts to generate a concise, empathetic, and structured summary of the patient's Quality of Life (QoL) concerns.

        **CRITICAL INSTRUCTIONS - PLEASE READ CAREFULLY:**
        1.  **STRICTLY USE ONLY THE PROVIDED \"Retrieved Patient Narrative Excerpts\".**
            * **DO NOT INFER, GUESS, OR ADD EXTERNAL INFORMATION.**
            * **DO NOT FABRICATE OR HALLUCINATE TEXT.** Every piece of information must come directly from the excerpts.
            * **NEVER INCLUDE \"Document ID\", \"page_content\", or any raw, truncated text from the excerpts.** Summarize the content, do not quote internal processing details.
            * **ENSURE ALL INFORMATION DIRECTLY REFLECTS THE PATIENT'S NARRATIVE.** Do not skew or rephrase to introduce new meanings.
        2.  For each of the five QoL themes below, **EXTRACT AND SUMMARIZE SPECIFIC DETAILS, EXAMPLES, AND KEY PHRASES DIRECTLY FROM THE NARRATIVE.**
            * **IMPORTANT: When quoting directly, preserve original wording, spacing, and punctuation.**
            * Aim to use the patient's own words or very close paraphrases where possible, but ensure the output is a coherent summary, not just a list of raw phrases.
            * Be concise and factual.
        3.  If a theme is genuinely NOT mentioned or strongly inferable *from the provided excerpts*, and you cannot find *any relevant information within the excerpts for that specific theme*, then and *only then*, state: \"Not explicitly mentioned in narrative for this theme.\"
            * Do NOT invent or assume absence if information is present.
            * If a theme *is* mentioned, provide the details; do not state \"Not explicitly mentioned\".
        4.  Maintain a professional and objective tone throughout the summary.
        5.  Do not include outside information or recommendations on the patient's situation. Your sole responsibility is to accurately summarize the patient's QoL themes based *only* on the provided text.

        **Themes to Address in the Summary (Extract specific details and examples from context):**
        * **Physical Symptoms and Functional Limitations**: Describe any pain, discomfort, physical restrictions, or impact on daily activities (e.g., movement, sleep, work tasks) *as explicitly stated in the narrative excerpts*. Provide concrete examples if available.
        * **Body Image Concerns**: Detail any self-consciousness, embarrassment, appearance concerns, or feelings about their body due to the condition (e.g., hernia, scars) *as explicitly stated in the narrative excerpts*. Provide concrete examples if available.
        * **Mental Health Challenges**: Summarize emotional states such as anxiety, worry, fear, stress, frustration, low mood, depression, feelings of being overwhelmed, or social isolation *as explicitly described in the narrative excerpts*. Provide concrete examples if available.
        * **Impact on Interpersonal Relationships**: Explain effects on social interactions, friendships, family dynamics, or intimate relationships *as explicitly described in the narrative excerpts*. Provide concrete examples if available.
        * **Employment and Financial Concerns**: Cover issues related to work ability, job changes, financial strain, income impact, or career concerns *as explicitly described in the narrative excerpts*. Provide concrete examples if available.

        ---
        Retrieved Patient Narrative Excerpts:
        {context}
        ---

        Patient's Overall Quality of Life Summary (structured by themes):
        """

    def get_rag_summary_for_patient(self, patient_id: str, query: str, all_documents: List[Document]) -> str:
        if self.vectorstore is None or self.ollama_llm is None:
            return "Error: RAG system not ready."

        patient_docs = [doc for doc in all_documents if doc.metadata.get("patient_id") == patient_id]
        if not patient_docs:
            return f"No excerpts for patient {patient_id}"

        texts = [doc.page_content for doc in patient_docs]
        metadatas = [doc.metadata for doc in patient_docs]
        bm25 = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25.k = 5

        dense_retriever = RunnableLambda(lambda q: self._get_patient_specific_dense_docs(q, patient_id))
        ensemble = EnsembleRetriever(retrievers=[bm25, dense_retriever], weights=[0.5, 0.5])

        def rerank(docs, q):
            if not self.reranker or not docs:
                return docs[:5]
            pairs = [(q, d.page_content) for d in docs]
            scores = self.reranker.predict(pairs)
            return [docs[i] for i in np.argsort(scores)[::-1][:5]]

        chain = (
            {"context": ensemble, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(context=RunnableLambda(lambda x: rerank(x["context"], x["question"])))
            | ChatPromptTemplate.from_template(self._get_rag_prompt_template())
            | self.ollama_llm
            | StrOutputParser()
        )

        try:
            return chain.invoke(query)
        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            return f"Error: {e}"


class PatientReportGenerator:
    def __init__(self, thematic_model: AutoModelForSequenceClassification, thematic_tokenizer: AutoTokenizer, nlp_analysis_service: Any, qol_themes: List[str], mlb_thematic_instance: MultiLabelBinarizer):
        self.thematic_model = thematic_model
        self.thematic_tokenizer = thematic_tokenizer
        self.nlp_analysis_service = nlp_analysis_service
        self.qol_themes = qol_themes
        self.mlb_thematic = mlb_thematic_instance
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.thematic_model.to(self.device)

    # def generate_comprehensive_patient_report(self, patient_id: str, patients_df: pd.DataFrame, rag_system: Any) -> str:
    # def generate_comprehensive_patient_report(self, patients_df: pd.DataFrame, rag_system: Any, all_documents: List[Document]) -> str:
    #     """Generates a comprehensive QoL report for a given patient ID."""
    #     patient_data = patients_df

    #     if patient_data.empty:
    #         logger.error(f"Patient ID not found in the dataset.")
    #         return f"Error: Patient ID not found."

    #     # patient_row = patient_data
    #     patient_row = patient_data.iloc[0]
    #     report_parts = []

    #     metadata = patient_row.get('Metadata', {})
    #     print(f"DEBUG: Metadata for patient: {metadata}")  # Debugging line to check metadata content

    #     patient_id = patient_row['Patient_ID']
    #     report_parts.append("# Patient Quality of Life Report 📄")
    #     report_parts.append(f"### Basic Information (Patient ID: **{patient_id}**)\\n")

    #     report_parts.append(f"- **Patient ID:** {patient_id}")

    #     # logger.info(f"DEBUG: 'Able_to_lie_flat_comfortably' from metadata for {patient_id}: {metadata.get('Able_to_lie_flat_comfortably', 'KEY_NOT_FOUND')}")
    #     report_parts.append(f"- **Name:** {patient_id}")
    #     report_parts.append(f"- **Age:** {metadata.get('Age', 'N/A')}")
    #     report_parts.append(f"- **Gender:** {metadata.get('Gender', 'N/A')}")
    #     report_parts.append(f"- **Ethnicity:** {metadata.get('Ethnicity', 'N/A')}")
    #     report_parts.append(f"- **Marital Status:** {metadata.get('Marital_Status', 'N/A')}")
    #     report_parts.append(f"- **Occupation:** {metadata.get('Job_Title', 'N/A')} ({metadata.get('Occupation_Category', 'N/A')})\\n")

    #     report_parts.append(f"- **Previous Hernia Repairs:**")
    #     if metadata.get('Previous_Hernia_Repairs'):
    #         for repair in metadata['Previous_Hernia_Repairs']:
    #             report_parts.append(
    #                 f"   - Year: {repair.get('Year', 'N/A')}, Type: {repair.get('Type', 'N/A')}, "
    #                 f"Mesh: {repair.get('Mesh_Used', 'N/A')}, Wound Breakdown: {repair.get('Wound_Breakdown', 'N/A')}, "
    #                 f"Healing Time: {repair.get('Healing_Time', 'N/A')}"
    #             )
    #     else:
    #         report_parts.append("   - None reported.")

    #     report_parts.append("\n- **Medical History:**")
    #     medical_history = metadata.get('Medical_History', {})
    #     if medical_history:
    #         for condition, status in medical_history.items():
    #             if status not in ["No", "N/A", "", None] and condition != "Prior_Major_Surgeries":
    #                 report_parts.append(f"   - {condition.replace('_', ' ').title()}: {status}")
    #         if medical_history.get('Prior_Major_Surgeries'):
    #             report_parts.append(f"   - Prior Major Surgeries: {', '.join(medical_history['Prior_Major_Surgeries'])}\n")
    #         else:
    #             report_parts.append("   - No prior major surgeries reported.\n")
    #     else:
    #         report_parts.append("   - No medical history reported.\n")

    #     report_parts.append("\n- **Current Medications:**")
    #     medications = metadata.get('Medications', {})
    #     if medications:
    #         for med, dosage in medications.items():
    #             report_parts.append(f"   - {med}: {dosage}")
    #     else:
    #         report_parts.append("   - No current medications reported.")

    #     report_parts.append(f"- **Able to lie flat comfortably:** {metadata.get('Able_to_lie_flat_comfortably', 'N/A')}")
    #     report_parts.append(f"- **QoL Areas Affected (Self-reported from Metadata):** {', '.join(metadata.get('QoL_Areas_Affected', ['None specified']))}\\n")

    #     # AI-CLASSIFIED QUALITY OF LIFE THEMES
    #     report_parts.append("## AI-Classified Quality of Life Themes 🤖\\n")
    #     text_for_classification = patient_row['preprocessed_narrative']

    #     try:
    #         inputs = self.thematic_tokenizer(text_for_classification, return_tensors="pt", truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)
    #         inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #         with torch.no_grad():
    #             logits = self.thematic_model(**inputs).logits

    #         probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    #         predicted_labels_indices = np.where(probabilities > 0.5)[0]

    #         predicted_theme_names = [self.qol_themes[idx] for idx in predicted_labels_indices if idx < len(self.qol_themes)]

    #         if predicted_theme_names:
    #             report_parts.append(f"- **Detected Themes:** {', '.join(predicted_theme_names)}")
    #             confidence_scores_str = ", ".join([f"{self.qol_themes[idx]}: {probabilities[idx]:.2f}" for idx in predicted_labels_indices])
    #             report_parts.append(f"   (Confidence Scores: {confidence_scores_str})")
    #         else:
    #             report_parts.append("- **Detected Themes:** No specific themes confidently classified by AI (all scores below 0.5).\\n")
    #             if len(probabilities) > 0:
    #                 top_idx = np.argmax(probabilities)
    #                 report_parts.append(f"   (Highest predicted: {self.qol_themes[top_idx]}: {probabilities[top_idx]:.2f})\\n")
    #             else:
    #                 report_parts.append("   (No probabilities available.)\\n")
    #     except Exception as e:
    #         logger.error(f"Error during thematic classification for patient {patient_id}: {e}")
    #         report_parts.append("- **Detected Themes:** Error during classification.\\n")

    #     report_parts.append("\n## Sentiment and Emotion Analysis 📊")
    #     if 'sentiment_and_emotions' in patient_row:
    #         overall_sentiment = patient_row['sentiment_and_emotions'].get('overall_sentiment', {})
    #         emotions = patient_row['sentiment_and_emotions'].get('emotions', {})

    #         report_parts.append(f"- *Overall Sentiment:* *{overall_sentiment.get('label', 'N/A').upper()}* (Score: {overall_sentiment.get('score', 0.0):.2f})")

    #         if emotions:

    #             top_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:4])
    #             emotion_str = ", ".join([f"**{label.capitalize()}**: {score:.2f}" for label, score in top_emotions.items()])
    #             report_parts.append(f"- *Key Emotions Detected (Average Intensity):* {emotion_str}")
    #         else:
    #             report_parts.append("- No specific emotions confidently detected.")
    #     else:
    #         report_parts.append("- Sentiment and emotion analysis results not available.")


    #     report_parts.append("\n### MENTAL HEALTH PATTERN DETECTION (ZERO-SHOT CLASSIFICATION)")

    #     if 'mental_health_linguistic_signals' in patient_row:
    #         mh_signals = patient_row['mental_health_linguistic_signals']
    #         # if mh_signals and 'details' in mh_signals:
    #         if isinstance(mh_signals, dict) and 'details' in mh_signals:
    #             sorted_patterns = sorted(mh_signals['details'].items(), key=lambda item: item[1], reverse=True)[:3]
    #             pattern_str = ", ".join([f"*{label.capitalize()}*: {score:.2f}\n" for label, score in sorted_patterns])
    #             report_parts.append(f"- *Top Detected Mental Health Patterns:* {pattern_str}")
    #         else:
    #             report_parts.append("- No specific mental health linguistic signals detected.\n")
    #     else:
    #         report_parts.append("- Mental health linguistic signals not available.\n")


    #     report_parts.append("\n## AI-Generated Comprehensive QoL Summary (via RAG) 📝\n")

    #     if rag_system:
    #         logger.info(f"Rag system initialized for patient {patient_id}. Generating comprehensive summary...")

    #         rag_query = f"""
    #         Generate a comprehensive Quality of Life summary for Patient ID: {patient_id}.
    #         Consider the following context:
    #         - Age: {metadata.get('Age', 'N/A')}
    #         - Gender: {metadata.get('Gender', 'N/A')}
    #         - Occupation: {metadata.get('Job_Title', 'N/A')}
    #         - Previous Hernia Repairs: {len(metadata.get('Previous_Hernia_Repairs', []))} reported.

    #         Based on the patient's combined narrative, provide a structured summary addressing the five QoL themes:
    #         -   **Physical Symptoms and Functional Limitations**: What pain, discomfort, or limitations in daily activities are mentioned?
    #         -   **Body Image Concerns**: Detail any self-consciousness, embarrassment, or concerns about appearance?
    #         -   **Mental Health Challenges**: What emotional states, anxieties, or mood issues are evident?
    #         -   **Impact on Interpersonal Relationships**: How are social, family, or intimate relationships affected?
    #         -   **Employment and Financial Concerns**: What impact is there on work ability, job, or financial stability?

    #         If a theme is not explicitly mentioned in the patient's narrative, clearly state 'Not explicitly mentioned in narrative for this theme.'

    #         Patient's Combined Narrative: {patient_row['combined_narrative']}
    #         """
    #         try:
    #             rag_summary = rag_system.get_rag_summary_for_patient(patient_id, rag_query, all_documents)
    #             report_parts.append(rag_summary)
    #         except Exception as e:
    #             logger.error(f"Error during RAG summary generation for patient {patient_id}: {e}")
    #             report_parts.append(f"Error: Unable to generate RAG summary for {patient_id}.")
    #     else:
    #         logger.warning("RAG system not initialized. Cannot generate comprehensive summary.")
    #         report_parts.append("RAG system not initialized. Cannot generate comprehensive summary.")

    #     return "\n".join(report_parts)




    def generate_comprehensive_patient_report(self, patients_df: pd.DataFrame, rag_system: Any, all_documents: List[Document]) -> dict:
        """Generates a comprehensive QoL report for a given patient ID in JSON format."""

        patient_data = patients_df

        if patient_data.empty:
            logger.error(f"Patient ID not found in the dataset.")
            return {"error": "Patient ID not found."}

        patient_row = patient_data.iloc[0]
        metadata = patient_row.get('Metadata', {})
        patient_id = patient_row['Patient_ID']

        # --- METADATA ---
        metadata_section = {
            "Patient ID": patient_id,
            "Name": patient_id,
            "Age": metadata.get('Age', 'N/A'),
            "Gender": metadata.get('Gender', 'N/A'),
            "Ethnicity": metadata.get('Ethnicity', 'N/A'),
            "Marital Status": metadata.get('Marital_Status', 'N/A'),
            "Occupation": {
                "Job Title": metadata.get('Job_Title', 'N/A'),
                "Category": metadata.get('Occupation_Category', 'N/A')
            },
            "Previous Hernia Repairs": metadata.get('Previous_Hernia_Repairs', []),
            "Medical History": metadata.get('Medical_History', {}),
            "Medications": metadata.get('Medications', {}),
            "Able to Lie Flat Comfortably": metadata.get('Able_to_lie_flat_comfortably', 'N/A'),
            "QoL Areas Affected": metadata.get('QoL_Areas_Affected', ['None specified'])
        }

        # --- DETECTED THEMES ---
        detected_themes_section = {}
        text_for_classification = patient_row['preprocessed_narrative']

        try:
            inputs = self.thematic_tokenizer(text_for_classification, return_tensors="pt", truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.thematic_model(**inputs).logits

            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            predicted_labels_indices = np.where(probabilities > 0.5)[0]

            predicted_theme_names = [self.qol_themes[idx] for idx in predicted_labels_indices if idx < len(self.qol_themes)]

            if predicted_theme_names:
                confidence_scores = {self.qol_themes[idx]: float(probabilities[idx]) for idx in predicted_labels_indices}
                detected_themes_section = {
                    "Detected Themes": predicted_theme_names,
                    "Confidence Scores": confidence_scores
                }
            else:
                top_idx = int(np.argmax(probabilities)) if len(probabilities) > 0 else None
                detected_themes_section = {
                    "Detected Themes": [],
                    "Confidence Scores": {},
                    "Highest Predicted": {
                        "Theme": self.qol_themes[top_idx] if top_idx is not None else "N/A",
                        "Score": float(probabilities[top_idx]) if top_idx is not None else 0.0
                    }
                }
        except Exception as e:
            logger.error(f"Error during thematic classification for patient {patient_id}: {e}")
            detected_themes_section = {
                "error": "Error during classification."
            }

        # --- SENTIMENT & EMOTION ANALYSIS ---
        sentiment_section = {}
        if 'sentiment_and_emotions' in patient_row:
            overall_sentiment = patient_row['sentiment_and_emotions'].get('overall_sentiment', {})
            emotions = patient_row['sentiment_and_emotions'].get('emotions', {})

            sentiment_section = {
                "Overall Sentiment": {
                    "Label": overall_sentiment.get('label', 'N/A'),
                    "Score": float(overall_sentiment.get('score', 0.0))
                },
                "Top Emotions": dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:4]) if emotions else {}
            }
        else:
            sentiment_section = {
                "error": "Sentiment and emotion analysis not available."
            }

        # --- MENTAL HEALTH ZERO-SHOT CLASSIFICATION ---
        zero_shot_section = {}
        if 'mental_health_linguistic_signals' in patient_row:
            mh_signals = patient_row['mental_health_linguistic_signals']
            if isinstance(mh_signals, dict) and 'details' in mh_signals:
                sorted_patterns = sorted(mh_signals['details'].items(), key=lambda item: item[1], reverse=True)[:3]
                zero_shot_section = {label.capitalize(): float(score) for label, score in sorted_patterns}
            else:
                zero_shot_section = {"message": "No specific mental health patterns detected."}
        else:
            zero_shot_section = {"message": "Mental health linguistic signals not available."}

        # --- RAG SUMMARY ---
        rag_summary_text = ""
        if rag_system:
            rag_query = f"""
            Generate a comprehensive Quality of Life summary for Patient ID: {patient_id}.
            Consider the following context:
            - Age: {metadata.get('Age', 'N/A')}
            - Gender: {metadata.get('Gender', 'N/A')}
            - Occupation: {metadata.get('Job_Title', 'N/A')}
            - Previous Hernia Repairs: {len(metadata.get('Previous_Hernia_Repairs', []))} reported.

            Based on the patient's combined narrative, provide a structured summary addressing the five QoL themes:
            -   **Physical Symptoms and Functional Limitations**: What pain, discomfort, or limitations in daily activities are mentioned?
            -   **Body Image Concerns**: Detail any self-consciousness, embarrassment, or concerns about appearance?
            -   **Mental Health Challenges**: What emotional states, anxieties, or mood issues are evident?
            -   **Impact on Interpersonal Relationships**: How are social, family, or intimate relationships affected?
            -   **Employment and Financial Concerns**: What impact is there on work ability, job, or financial stability?

            If a theme is not explicitly mentioned in the patient's narrative, clearly state 'Not explicitly mentioned in narrative for this theme.'

            Patient's Combined Narrative: {patient_row['combined_narrative']}
            """
            try:
                rag_summary_text = rag_system.get_rag_summary_for_patient(patient_id, rag_query, all_documents)
            except Exception as e:
                logger.error(f"Error during RAG summary generation for patient {patient_id}: {e}")
                rag_summary_text = "Error: Unable to generate RAG summary."
        else:
            rag_summary_text = "RAG system not initialized."

        # --- FINAL JSON STRUCTURE ---
        report_json = {
            "metadata": metadata_section,
            "detected_themes": detected_themes_section,
            "sentiment_and_emotion_analysis": sentiment_section,
            "zero_shot_classification": zero_shot_section,
            "qol_summary": rag_summary_text
        }

        output_folder = "reports"
        os.makedirs(output_folder, exist_ok=True)  # create folder if it doesn't exist

        # Generate unique filename using UUID + Patient ID
        unique_id = str(uuid.uuid4())
        safe_patient_id = str(patient_id).replace(" ", "_")
        filename = f"{unique_id}_{safe_patient_id}.json"
        file_path = os.path.join(output_folder, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report_json, f, indent=4, ensure_ascii=False)
            logger.info(f"Report saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write report to file for {patient_id}: {e}")
        
        # emotions = plot_emotions(report_json["sentiment_and_emotion_analysis"]["Top Emotions"], unique_id, patient_id)
        # hernia_repairs = plot_hernia_repairs(report_json["metadata"]["Previous Hernia Repairs"], unique_id, patient_id)
        # wordcloud = plot_wordcloud(report_json["qol_summary"], unique_id, patient_id)

        # def encode_image_to_base64(image_path):
        #     if not image_path:
        #         return None
        #     with open(image_path, "rb") as img:
        #         return base64.b64encode(img.read()).decode("utf-8")

        # encoded_emotions = encode_image_to_base64(emotions)
        # encoded_hernia = encode_image_to_base64(hernia_repairs)
        # encoded_wordcloud = encode_image_to_base64(wordcloud)

        # # 3. Add encoded images to report_json
        # report_json["visualizations"] = {
        #     "emotions_chart_base64": encoded_emotions,
        #     "hernia_timeline_base64": encoded_hernia,
        #     "wordcloud_base64": encoded_wordcloud
        # }

        return report_json


    # def generate_comprehensive_patient_report(self, patient_row: pd.Series, rag_system: Any) -> str:
    #     """Generates a comprehensive QoL report for a given patient row (as pd.Series)."""

    #     if patient_row is None or patient_row.empty:
    #         logger.error(f"Patient row is empty or invalid.")
    #         return "Error: Patient data not available."

    #     report_parts = []

    #     # Extract metadata dictionary safely
    #     metadata = patient_row.get('Metadata', {})
    #     if isinstance(metadata, pd.Series):
    #         metadata = metadata.to_dict()

    #     patient_id = patient_row.get('Patient_ID', 'Unknown')

    #     report_parts.append("# Patient Quality of Life Report 📄")
    #     report_parts.append(f"### Basic Information (Patient ID: **{patient_id}**)\n")

    #     report_parts.append(f"- **Patient ID:** {patient_id}")
    #     report_parts.append(f"- **Name:** {patient_id}")
    #     report_parts.append(f"- **Age:** {metadata.get('Age', 'N/A')}")
    #     report_parts.append(f"- **Gender:** {metadata.get('Gender', 'N/A')}")
    #     report_parts.append(f"- **Ethnicity:** {metadata.get('Ethnicity', 'N/A')}")
    #     report_parts.append(f"- **Marital Status:** {metadata.get('Marital_Status', 'N/A')}")
    #     report_parts.append(f"- **Occupation:** {metadata.get('Job_Title', 'N/A')} ({metadata.get('Occupation_Category', 'N/A')})\n")

    #     report_parts.append(f"- **Previous Hernia Repairs:**")
    #     if metadata.get('Previous_Hernia_Repairs'):
    #         for repair in metadata['Previous_Hernia_Repairs']:
    #             report_parts.append(
    #                 f"   - Year: {repair.get('Year', 'N/A')}, Type: {repair.get('Type', 'N/A')}, "
    #                 f"Mesh: {repair.get('Mesh_Used', 'N/A')}, Wound Breakdown: {repair.get('Wound_Breakdown', 'N/A')}, "
    #                 f"Healing Time: {repair.get('Healing_Time', 'N/A')}"
    #             )
    #     else:
    #         report_parts.append("   - None reported.")

    #     report_parts.append("\n- **Medical History:**")
    #     medical_history = metadata.get('Medical_History', {})
    #     if isinstance(medical_history, pd.Series):
    #         medical_history = medical_history.to_dict()

    #     if medical_history:
    #         for condition, status in medical_history.items():
    #             if status not in ["No", "N/A", "", None] and condition != "Prior_Major_Surgeries":
    #                 report_parts.append(f"   - {condition.replace('_', ' ').title()}: {status}")
    #         if medical_history.get('Prior_Major_Surgeries'):
    #             report_parts.append(f"   - Prior Major Surgeries: {', '.join(medical_history['Prior_Major_Surgeries'])}\n")
    #         else:
    #             report_parts.append("   - No prior major surgeries reported.\n")
    #     else:
    #         report_parts.append("   - No medical history reported.\n")

    #     report_parts.append("\n- **Current Medications:**")
    #     medications = metadata.get('Medications', {})
    #     if isinstance(medications, pd.Series):
    #         medications = medications.to_dict()
    #     if medications:
    #         for med, dosage in medications.items():
    #             report_parts.append(f"   - {med}: {dosage}")
    #     else:
    #         report_parts.append("   - No current medications reported.")

    #     report_parts.append(f"- **Able to lie flat comfortably:** {metadata.get('Able_to_lie_flat_comfortably', 'N/A')}")
    #     report_parts.append(f"- **QoL Areas Affected (Self-reported from Metadata):** {', '.join(metadata.get('QoL_Areas_Affected', ['None specified']))}\n")

    #     # AI-CLASSIFIED THEMES
    #     report_parts.append("## AI-Classified Quality of Life Themes 🤖\n")
    #     text_for_classification = patient_row.get('preprocessed_narrative', '')
    #     try:
    #         if not isinstance(text_for_classification, str):
    #             text_for_classification = str(text_for_classification)

    #         inputs = self.thematic_tokenizer(
    #             text_for_classification,
    #             return_tensors="pt",
    #             truncation=True,
    #             padding="max_length",
    #             max_length=Config.MAX_MODEL_INPUT_LENGTH
    #         )
    #         inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #         with torch.no_grad():
    #             logits = self.thematic_model(**inputs).logits

    #         probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    #         predicted_labels_indices = np.where(probabilities > 0.5)[0]
    #         predicted_theme_names = [self.qol_themes[idx] for idx in predicted_labels_indices if idx < len(self.qol_themes)]

    #         if predicted_theme_names:
    #             report_parts.append(f"- **Detected Themes:** {', '.join(predicted_theme_names)}")
    #             confidence_scores_str = ", ".join([f"{self.qol_themes[idx]}: {probabilities[idx]:.2f}" for idx in predicted_labels_indices])
    #             report_parts.append(f"   (Confidence Scores: {confidence_scores_str})")
    #         else:
    #             report_parts.append("- **Detected Themes:** No themes confidently classified (all scores < 0.5).\n")
    #             if len(probabilities) > 0:
    #                 top_idx = np.argmax(probabilities)
    #                 report_parts.append(f"   (Highest predicted: {self.qol_themes[top_idx]}: {probabilities[top_idx]:.2f})\n")
    #     except Exception as e:
    #         logger.error(f"Error during thematic classification for patient {patient_id}: {e}")
    #         report_parts.append("- **Detected Themes:** Error during classification.\n")

    #     # SENTIMENT & EMOTION
    #     report_parts.append("\n## Sentiment and Emotion Analysis 📊")
    #     sentiment_data = patient_row.get('sentiment_and_emotions', {})
    #     if isinstance(sentiment_data, pd.Series):
    #         sentiment_data = sentiment_data.to_dict()

    #     overall_sentiment = sentiment_data.get('overall_sentiment', {})
    #     emotions = sentiment_data.get('emotions', {})

    #     if overall_sentiment:
    #         report_parts.append(f"- *Overall Sentiment:* *{overall_sentiment.get('label', 'N/A').upper()}* (Score: {overall_sentiment.get('score', 0.0):.2f})")

    #     if emotions:
    #         top_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:4])
    #         emotion_str = ", ".join([f"**{label.capitalize()}**: {score:.2f}" for label, score in top_emotions.items()])
    #         report_parts.append(f"- *Key Emotions Detected (Average Intensity):* {emotion_str}")
    #     else:
    #         report_parts.append("- No specific emotions confidently detected.")

    #     # MENTAL HEALTH SIGNALS
    #     report_parts.append("\n### MENTAL HEALTH PATTERN DETECTION (ZERO-SHOT CLASSIFICATION)")
    #     mh_signals = patient_row.get('mental_health_linguistic_signals', {})
    #     if isinstance(mh_signals, pd.Series):
    #         mh_signals = mh_signals.to_dict()

    #     details = mh_signals.get('details', {})
    #     if details:
    #         sorted_patterns = sorted(details.items(), key=lambda item: item[1], reverse=True)[:3]
    #         pattern_str = ", ".join([f"*{label.capitalize()}*: {score:.2f}" for label, score in sorted_patterns])
    #         report_parts.append(f"- *Top Detected Mental Health Patterns:* {pattern_str}")
    #     else:
    #         report_parts.append("- No specific mental health linguistic signals detected.")

    #     # RAG SUMMARY
    #     report_parts.append("\n## AI-Generated Comprehensive QoL Summary (via RAG) 📝\n")
    #     if rag_system:
    #         rag_query = f"""
    #         Generate a comprehensive Quality of Life summary for Patient ID: {patient_id}.
    #         Context:
    #         - Age: {metadata.get('Age', 'N/A')}
    #         - Gender: {metadata.get('Gender', 'N/A')}
    #         - Occupation: {metadata.get('Job_Title', 'N/A')}
    #         - Previous Hernia Repairs: {len(metadata.get('Previous_Hernia_Repairs', []))}

    #         Based on the patient's combined narrative, provide a structured summary:
    #         - Physical Symptoms
    #         - Body Image
    #         - Mental Health
    #         - Relationships
    #         - Employment

    #         Patient Narrative: {patient_row['combined_narrative']}
    #         """
    #         try:
    #             rag_summary = rag_system.get_rag_summary_for_patient(patient_id, rag_query)
    #             report_parts.append(rag_summary)
    #         except Exception as e:
    #             logger.error(f"Error during RAG summary generation for patient {patient_id}: {e}")
    #             report_parts.append(f"Error: Unable to generate RAG summary.")
    #     else:
    #         logger.warning("RAG system not initialized.")
    #         report_parts.append("RAG system not initialized.")

    #     return "\n".join(report_parts)



