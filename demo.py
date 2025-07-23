import shutil
import os
import time
import pandas as pd
import json
import re
import nltk
import torch
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
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import math
import random

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

    OLLAMA_BASE_URL = "https://b37f03b3d465.ngrok-free.app"
    OLLAMA_MODEL_NAME = "llama3:8b-instruct-q8_0"

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

def main():
    logger.info("Starting main execution...")

    # --- 2. Data Loading and Preprocessing ---
    class PatientDataProcessor:
        def __init__(self, qol_themes: List[str], mlb_thematic_instance: MultiLabelBinarizer, tokenizer: AutoTokenizer):
            self.qol_themes = qol_themes
            self.mlb_thematic = mlb_thematic_instance
            self.tokenizer = tokenizer
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
            self.lemmatizer = nltk.stem.WordNetLemmatizer()

        def load_data_from_single_file(self, path: str) -> pd.DataFrame:
            logger.info(f"Loading data from {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                logger.info(f"Dataset loaded successfully! Total patients: {len(df)}")
                return df
            except FileNotFoundError as e:
                logger.error(f"Dataset file not found: {e}. Please check the path in Config.")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from dataset: {e}. Check file integrity.")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during data loading: {e}")
                raise

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
            existing_narrative_cols = [col for col in narrative_fields_keys if col in df.columns]
            df['combined_narrative'] = df[existing_narrative_cols].apply(
                lambda row: ' '.join(filter(None, row.values.astype(str))).strip(), axis=1
            )
            df['combined_narrative'] = df['combined_narrative'].str.replace(r'\\s+', ' ', regex=True).fillna('')
            logger.info("Combined narrative created from relevant fields.")
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

        def assign_robust_multi_labels(self, row: pd.Series) -> List[str]:
            labels_for_this_row = []
            narrative_lower = row['combined_narrative'].lower()

            symptom_keywords = [
                "pain", "ache", "discomfort", "pulling", "stiffness", "fragile", "limited", "movement",
                "bending", "lifting", "walking", "sitting", "standing", "sleep", "exhausting",
                "struggle", "strain", "wince", "jolt", "burden", "mobility", "physical", "fatigue",
                "nausea", "swelling", "shortness of breath", "cramp", "soreness", "weakness",
                "vomiting", "dizzy", "coughs", "breathing", "tightness", "headache", "migraine",
                "fever", "aching", "cramping", "stiff", "tired", "debilitating", "debilitated"
            ]
            body_image_keywords = [
                "self-conscious", "embarrassed", "looks", "bulge", "disfigured", "ruins my figure",
                "hiding something", "pregnant", "ashamed", "unattractive", "scrutinized", "appearance",
                "scar", "figure", "deformity", "visible", "unappealing", "misshapen", "ugly",
                "self image", "confidence", "blemish", "complexion", "features"
            ]
            mental_health_keywords = [
                "anxious", "worry", "fear", "dread", "nervous", "stressed", "panic", "hopeless",
                "pointless", "trapped", "despair", "low", "depressed", "sad", "down", "miserable",
                "unhappy", "frustrated", "irritable", "snap", "overwhelmed", "consumed", "draining",
                "mental health", "mood", "rumination", "isolated", "withdrawn", "guilt",
                "emotional eating", "distraction", "volatile", "anger", "stress", "crying", "agitated",
                "moody", "anxiety", "depression", "PTSD", "trauma", "coping", "therapy", "counselling",
                "depressive", "panic attacks", "insomnia", "suicidal", "self harm"
            ]
            relationship_keywords = [
                "social", "friends", "partner", "intimacy", "isolated", "withdrawn", "strain", "guilt",
                "awkward", "less desirable", "spontaneity", "relationships", "dating", "sexual",
                "family", "loved one", "connection", "communication", "support", "depend", "burden",
                "marriage", "divorce", "friendship", "social life", "close"
            ]
            employment_keywords = [
                "job", "work", "employment", "financial", "income", "bills", "livelihood", "unemployed",
                "shifts", "productive", "career", "salary", "pay", "money", "redundancy", "debt",
                "cost", "expenditure", "workplace", "colleague", "promotion", "leave", "disability",
                "benefits", "earning", "budget", "expenses"
            ]

            if any(kw in narrative_lower for kw in symptom_keywords) or \
            any(row[f] and row[f] != '' for f in ["Symptoms_Restrictions_Activities", "Symptoms_Adaptations_Face", "Symptoms_Affect_Movement", "Symptoms_Pain_Coping"]):
                labels_for_this_row.append("Symptoms and Function")
            if any(kw in narrative_lower for kw in body_image_keywords) or \
            any(row[f] and row[f] != '' for f in ["BodyImage_SelfConscious_Embarrassed", "BodyImage_Others_Noticing"]):
                labels_for_this_row.append("Body Image")
            if any(kw in narrative_lower for kw in mental_health_keywords) or \
            any(row[f] and row[f] != '' for f in ["MentalHealth_Affected_Elaborate", "MentalHealth_Coping_Strategies"]):
                labels_for_this_row.append("Mental Health")
            if any(kw in narrative_lower for kw in relationship_keywords) or \
            any(row[f] and row[f] != '' for f in ["Relationships_Social_Affected", "Relationships_Intimate_Affected"]):
                labels_for_this_row.append("Interpersonal Relationships")
            if any(kw in narrative_lower for kw in employment_keywords) or \
            any(row[f] and row[f] != '' for f in ["Employment_What_Do_You_Do", "Employment_Ability_To_Work_Affected", "Employment_Changed_Work", "Employment_Financial_Affected"]):
                labels_for_this_row.append("Employment/Financial Concerns")

            if not labels_for_this_row and narrative_lower.strip() != '':
                labels_for_this_row.append("Symptoms and Function")
            return labels_for_this_row

        def _prepare_dataframe_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:
            df['preprocessed_narrative'] = df['combined_narrative'].apply(self.preprocess_narrative_text)
            df['labels_as_names'] = df.apply(self.assign_robust_multi_labels, axis=1)
            df['labels_encoded'] = list(self.mlb_thematic.transform(df['labels_as_names']))
            df['labels_as_indices'] = df['labels_as_names'].apply(lambda names: [self.qol_themes.index(name) for name in names])
            df['dominant_label_id'] = df.apply(
                lambda row: random.choice(row['labels_as_indices']) if row['labels_as_indices'] else random.randint(0, len(self.qol_themes) - 1),
                axis=1
            )
            return df

        def prepare_data_for_classification(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dataset, Dataset, DataCollatorWithPadding]:
            df_processed = self._prepare_dataframe_for_classification(df.copy())
            logger.info("Narratives preprocessed and labels assigned for main dataset.")
            logger.info(f"Example multi-labels for first patient: {df_processed['labels_encoded'].iloc[0]}")
            logger.info(f"Class distribution of 'dominant_label_id' before oversampling: {Counter(df_processed['dominant_label_id'])}")

            text_splitter_for_classification = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE_CLASSIFICATION,
                chunk_overlap=Config.CHUNK_OVERLAP_CLASSIFICATION,
                length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=True)),
            )

            split_documents_for_classification = []
            for index, row in df_processed.iterrows():
                original_narrative = row['preprocessed_narrative']
                if not original_narrative.strip():
                    logger.debug(f"Skipping empty narrative for patient ID: {row['Patient_ID']}")
                    continue
                chunks = text_splitter_for_classification.split_text(original_narrative)
                if not chunks:
                    chunks = [original_narrative]
                for i, chunk in enumerate(chunks):
                    split_documents_for_classification.append({
                        'preprocessed_narrative': chunk,
                        'labels': row['labels_encoded'],
                        'patient_id': row['Patient_ID'],
                        'chunk_index': i,
                        'dominant_label_id': row['dominant_label_id']
                    })

            split_df_for_classification = pd.DataFrame(split_documents_for_classification)
            logger.info(f"Original number of narratives: {len(df_processed)}")
            logger.info(f"Number of chunks for classification after splitting: {len(split_df_for_classification)}")

            X_resample = split_df_for_classification['preprocessed_narrative'].tolist()
            y_resample = split_df_for_classification['dominant_label_id'].tolist()

            # ros = RandomOverSampler(random_state=Config.RANDOM_SEED)
            # resampled_indices_flat, y_resampled_labels_single = ros.fit_resample(
            #     np.arange(len(X_resample)).reshape(-1, 1),
            #     np.array(y_resample)
            # )
            # resampled_df = split_df_for_classification.iloc[resampled_indices_flat.flatten()].copy()

            # logger.info(f"Class distribution after oversampling: {Counter(y_resampled_labels_single)}")
            # logger.info(f"Number of samples after oversampling: {len(resampled_df)}")

            unique_classes = set(y_resample)
            logger.info(f"Unique dominant_label_ids found: {unique_classes}")

            if len(unique_classes) < 2:
                logger.warning("Only one class found in dominant_label_id. Skipping oversampling.")
                resampled_df = split_df_for_classification.copy()
            else:
                ros = RandomOverSampler(random_state=Config.RANDOM_SEED)
                resampled_indices_flat, y_resampled_labels_single = ros.fit_resample(
                    np.arange(len(X_resample)).reshape(-1, 1),
                    np.array(y_resample)
                )
                resampled_df = split_df_for_classification.iloc[resampled_indices_flat.flatten()].copy()
                logger.info(f"Class distribution after oversampling: {Counter(y_resampled_labels_single)}")
                logger.info(f"Number of samples after oversampling: {len(resampled_df)}")

            train_dataset_hf = Dataset.from_pandas(resampled_df[['preprocessed_narrative', 'labels']].astype({'labels': object}))
            eval_dataset_hf = Dataset.from_pandas(split_df_for_classification[['preprocessed_narrative', 'labels']].astype({'labels': object}))

            def tokenize_function(examples):
                return self.tokenizer(examples["preprocessed_narrative"], truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)

            train_dataset = train_dataset_hf.map(tokenize_function, batched=True, remove_columns=['preprocessed_narrative'], num_proc=os.cpu_count() if os.cpu_count() else 1)
            eval_dataset = eval_dataset_hf.map(tokenize_function, batched=True, remove_columns=['preprocessed_narrative'], num_proc=os.cpu_count() if os.cpu_count() else 1)
            logger.info("Datasets tokenized for classification.")

            class CustomDataCollatorWithPadding(DataCollatorWithPadding):
                def __call__(self, features):
                    labels = [feature.pop('labels') for feature in features]
                    batch = super().__call__(features)
                    batch['labels'] = torch.tensor(labels, dtype=torch.float32)
                    return batch

            data_collator = CustomDataCollatorWithPadding(tokenizer=self.tokenizer)
            logger.info("CustomDataCollatorWithPadding created.")
            return df_processed, train_dataset, eval_dataset, data_collator

    logger.info(f"Initializing tokenizer for thematic model: {Config.THEMATIC_MODEL_NAME}")
    thematic_tokenizer = AutoTokenizer.from_pretrained(Config.THEMATIC_MODEL_NAME)

    data_processor = PatientDataProcessor(Config.QOL_THEMES, mlb_thematic, thematic_tokenizer)
    patients_df_raw = data_processor.load_data_from_single_file(Config.DATASET_PATH_A)
    print(f"DEBUG: Loaded {len(patients_df_raw)} patients from dataset.")
    print(f"DEBUG: Columns in dataset: {patients_df_raw.columns.tolist()}")
    patients_df_extracted_initial = data_processor.extract_narrative_fields(patients_df_raw.copy())
    print(f"DEBUG: Extracted narratives for {len(patients_df_extracted_initial)} patients.")
    print(f"DEBUG: Columns after extraction: {patients_df_extracted_initial.columns.tolist()}")
    patients_df, train_dataset, eval_dataset, data_collator = data_processor.prepare_data_for_classification(patients_df_extracted_initial)
    logger.info(f"Final dataset prepared with {len(patients_df)} patients and {len(train_dataset)} training samples.")
    logger.info(f"Example preprocessed narrative for first patient: {patients_df['preprocessed_narrative'].iloc[0]}")

    logger.info("\nCombined Narrative for first patient (from consolidated data):")
    logger.info(patients_df['combined_narrative'].iloc[0])
    logger.info(f"Assigned Multi-Labels (names) for first patient: {patients_df['labels_as_names'].iloc[0]}")
    logger.info(f"Preprocessed Narrative for first patient: {patients_df['preprocessed_narrative'].iloc[0]}")

    # --- Load Pre-trained Thematic Classification Model ---
    logger.info(f"Loading fine-tuned thematic model from {Config.MODEL_SAVE_DIR} for inference.")

    try:
        thematic_model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_SAVE_DIR)
        logger.info("Thematic model loaded successfully for inference.")
        print("✅ Thematic model loaded successfully from saved directory.")

        thematic_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Thematic model moved to: {thematic_model.device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    except Exception as e:
        logger.error(f"Failed to load fine-tuned thematic model from {Config.MODEL_SAVE_DIR}: {e}")
        logger.info("Falling back to loading a fresh model from Hugging Face Hub (without fine-tuning) for this session.")

        thematic_model = AutoModelForSequenceClassification.from_pretrained(
            Config.THEMATIC_MODEL_NAME,
            num_labels=len(Config.QOL_THEMES),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        thematic_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Fresh model loaded and moved to: {thematic_model.device}")
        print("⚠️ Loaded a fresh model from Hugging Face Hub (fallback).")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    from sentence_transformers import SentenceTransformer

    try:
        sentence_embedder = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        logger.info("SentenceTransformer for keyword expansion loaded.")
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer for keyword expansion: {e}. Keyword expansion might be limited.")
        sentence_embedder = None

    known_good_emotion_model_name = Config.EMOTION_MODEL_NAME
    logger.info(f"Using emotion model name: {known_good_emotion_model_name}")

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

    nlp_analysis_service = NLPAnalysisService(
        thematic_tokenizer_instance=thematic_tokenizer,
        sentiment_model_name=Config.SENTIMENT_MODEL_NAME,
        emotion_model_name=Config.EMOTION_MODEL_NAME,
        sentence_embedder=sentence_embedder
    )

    logger.info("Applying sentiment and emotion analysis to narratives...")
    patients_df['sentiment_and_emotions'] = patients_df['preprocessed_narrative'].apply(
        nlp_analysis_service.analyze_sentiment_and_emotions_of_chunks
    )
    logger.info("Sentiment and emotion analysis completed.")
    print("\nSentiment and Emotions for first 3 patients:")
    print(patients_df[['Patient_ID', 'sentiment_and_emotions']].head(3))

    logger.info("Applying mental health linguistic signal detection (Zero-Shot Classification)....\n")
    patients_df['mental_health_linguistic_signals'] = patients_df['preprocessed_narrative'].apply(
        nlp_analysis_service.identify_mental_health_patterns_advanced
    )
    logger.info("Mental health linguistic signal detection completed.")
    print("\nMental Health Linguistic Signals (Zero-Shot Classification) for first 3 patients:")
    for i in range(min(3, len(patients_df))):
        patient_id = patients_df.iloc[i]['Patient_ID']
        zero_shot_result = patients_df.iloc[i]['mental_health_linguistic_signals']
        print(f"--- Patient ID: {patient_id} ---")
        if zero_shot_result and 'details' in zero_shot_result:
            sorted_patterns = sorted(zero_shot_result['details'].items(), key=lambda item: item[1], reverse=True)
            for label, score in sorted_patterns[:3]:
                print(f"  - {label.capitalize()}: {score:.2f}")
        else:
            print("  No significant mental health patterns detected or unexpected result format.")
    print("-" * 30)


    chroma_db_path = Config.CHROMA_DB_DIR
    print(f"Attempting to perform aggressive cleanup of ChromaDB directory: {chroma_db_path}")

    if os.path.exists(chroma_db_path):
        try:
            for i in range(3):
                shutil.rmtree(chroma_db_path)
                print(f"Attempt {i+1}: Existing ChromaDB directory removed. Waiting a bit...")
                time.sleep(2)
                if not os.path.exists(chroma_db_path):
                    print("Directory successfully removed and confirmed gone.")
                    break
            else:
                print("WARNING: Directory still exists after multiple removal attempts. It might be in use or have persistent permission issues.")
        except Exception as e:
            print(f"ERROR: Could not remove directory {chroma_db_path}: {e}")
            print("Please ensure your Google Drive is fully mounted. If the problem persists, delete the folder manually via the Google Drive web interface.")
    else:
        print("ChromaDB directory does not exist, no cleanup needed.")

    drive_root = "/content/drive/MyDrive"
    if not os.path.exists(drive_root):
        print("CRITICAL WARNING: Google Drive is not mounted or path is incorrect. Please ensure drive.mount('/content/drive') ran successfully.")
    if not os.access(drive_root, os.W_OK):
        print(f"CRITICAL WARNING: No write access to {drive_root}. This will prevent ChromaDB from being created. Check mount options.")

    print("ChromaDB cleanup script finished.")

    # --- RAGSystem Class with refined _initialize_vectorstore and patient-specific retrieval ---
    class RAGSystem:
        def __init__(self, embedding_model_name: str, chroma_db_dir: str, ollama_base_url: str, ollama_model_name: str, thematic_tokenizer_instance: AutoTokenizer, reranker_model_name: str):
            try:
                self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
                logger.info(f"Embedding function loaded: {embedding_model_name}")
            except Exception as e:
                logger.error(f"FATAL ERROR: Could not load embedding model {embedding_model_name}: {e}. Ensure model exists and network is stable.")
                self.embedding_function = None

            self.chroma_db_dir = chroma_db_dir
            self.ollama_llm = Ollama(model=ollama_model_name, base_url=ollama_base_url)
            self.thematic_tokenizer = thematic_tokenizer_instance
            self.vectorstore = None

            try:
                self.reranker = CrossEncoder(reranker_model_name)
                logger.info(f"Reranker model loaded: {reranker_model_name}")
            except Exception as e:
                logger.warning(f"Could not load reranker model {reranker_model_name}: {e}. Reranking will be skipped.")
                self.reranker = None

            self._initialize_vectorstore()

        def _initialize_vectorstore(self):
            """Initializes or loads the ChromaDB vector store."""
            os.makedirs(self.chroma_db_dir, exist_ok=True)

            if self.embedding_function is None:
                logger.error("ChromaDB initialization skipped: Embedding function failed to load.")
                self.vectorstore = None
                return

            try:
                self.vectorstore = Chroma(persist_directory=self.chroma_db_dir, embedding_function=self.embedding_function)

                if self.vectorstore._collection.count() == 0:
                    logger.warning("ChromaDB collection is empty at path or just created. It will be populated when documents are added.")
                else:
                    logger.info(f"Loaded existing ChromaDB from {self.chroma_db_dir} with {self.vectorstore._collection.count()} documents.")
            except Exception as e:
                logger.error(f"FATAL ERROR: Could not initialize or load ChromaDB at {self.chroma_db_dir}. Details: {e}")
                logger.error(f"This might be due to issues with the embedding model, FAISS, disk permissions, or corrupted existing database files.")
                self.vectorstore = None

            try:
                if self.ollama_llm:
                    logger.info(f"Attempting to connect to Ollama at: {self.ollama_llm.base_url}")
                    test_response = self.ollama_llm.invoke("Hi", temperature=0, timeout=10)
                    logger.info(f"Ollama connection successful. Test response: {test_response[:50]}...")
                else:
                    logger.warning("Ollama LLM object was not initialized, skipping connection test.")
            except Exception as e:
                logger.error(f"FAILED TO CONNECT TO OLLAMA at {self.ollama_llm.base_url if self.ollama_llm else 'N/A'}. Details: {e}")
                logger.error("Ensure Ollama is running, the model is pulled, and ngrok (or direct access) is correctly configured.")
                self.ollama_llm = None

        def populate_vectorstore(self, documents: List[Document]):
            """Adds documents to the vector store in batches."""
            if self.vectorstore is None:
                logger.error("Vectorstore not initialized. Cannot add documents. Check prior initialization errors.")
                return

            logger.info(f"Preparing to add {len(documents)} documents to ChromaDB.")
            if not documents:
                logger.warning("No documents provided to populate ChromaDB. Skipping population.")
                return

            try:
                if self.vectorstore._collection.count() > 0:
                    logger.info("Clearing existing ChromaDB collection before adding new documents.")
                    existing_ids = self.vectorstore._collection.get()['ids']
                    if existing_ids:
                        self.vectorstore._collection.delete(ids=existing_ids)
            except Exception as e:
                logger.warning(f"Could not clear existing ChromaDB collection (might be empty or corrupted): {e}")

            MAX_BATCH_SIZE = 5461

            num_documents = len(documents)
            if num_documents > MAX_BATCH_SIZE:
                logger.info(f"Documents exceed max batch size ({MAX_BATCH_SIZE}). Splitting into batches.")
                num_batches = math.ceil(num_documents / MAX_BATCH_SIZE)
                for i in range(num_batches):
                    start_index = i * MAX_BATCH_SIZE
                    end_index = min((i + 1) * MAX_BATCH_SIZE, num_documents)
                    batch = documents[start_index:end_index]
                    try:
                        self.vectorstore.add_documents(batch)
                        logger.info(f"Added batch {i+1}/{num_batches} ({len(batch)} documents).\n")
                    except Exception as e:
                        logger.error(f"Error adding batch {i+1} to ChromaDB: {e}")
                        logger.error("Vectorstore population failed for this batch. Check documents or vectorstore state.")
                        return
                logger.info(f"ChromaDB populated and persisted with all {num_documents} documents in batches.\n")
            else:
                try:
                    self.vectorstore.add_documents(documents)
                    logger.info(f"ChromaDB populated and persisted with {len(documents)} documents.\n")
                except Exception as e:
                    logger.error(f"Error adding documents to ChromaDB: {e}. Vectorstore population failed. This might indicate an issue with the documents themselves or the vectorstore state.\n")

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

        def _get_patient_specific_dense_docs(self, query_text: str, patient_id_filter: str) -> List[Document]:
            """
            Helper method to perform dense retrieval (similarity search) for a specific patient.
            This method is designed to be called by a RunnableLambda.
            """
            if self.vectorstore is None:
                logger.error("Vectorstore not initialized in _get_patient_specific_dense_docs.")
                return []
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query_text,
                    k=Config.RAG_TOP_K_RETRIEVAL,
                    filter={"patient_id": patient_id_filter}
                )
                return [doc for doc, _score in results_with_scores]
            except Exception as e:
                logger.error(f"Error during patient-specific dense retrieval for patient {patient_id_filter}: {e}")
                return []

        def get_rag_summary_for_patient(self, patient_id: str, query: str) -> str:
            """Generates a RAG summary for a specific patient by filtering retrieval."""
            if self.vectorstore is None:
                logger.error("Vectorstore not initialized. Cannot generate patient-specific RAG summary.")
                return "Error: RAG system not ready."
            if self.ollama_llm is None:
                logger.error("Ollama LLM not initialized or connected. Cannot generate patient-specific RAG summary.")
                return "Error: LLM not available."

            global documents_for_vectorstore
            patient_docs = [doc for doc in documents_for_vectorstore if doc.metadata.get("patient_id") == patient_id]

            if not patient_docs:
                logger.warning(f"No documents found for patient ID: {patient_id}. Cannot generate RAG summary.")
                return f"No relevant narrative excerpts found for patient {patient_id} to generate a RAG summary."

            patient_texts = [doc.page_content for doc in patient_docs]
            patient_metadatas = [doc.metadata for doc in patient_docs]

            bm25_retriever = BM25Retriever.from_texts(texts=patient_texts, metadatas=patient_metadatas)
            bm25_retriever.k = Config.RAG_TOP_K_RETRIEVAL

            dense_retriever_runnable = RunnableLambda(
                lambda q: self._get_patient_specific_dense_docs(q, patient_id)
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever_runnable],
                weights=[0.5, 0.5]
            )
            logger.info(f"Ensemble (Hybrid) Retriever configured for patient: {patient_id}.")

            def _rerank_documents_local(retrieved_docs: List[Document], current_query: str) -> List[Document]:
                if not self.reranker:
                    logger.warning("Reranker not available. Skipping re-ranking.")
                    return retrieved_docs[:Config.RAG_TOP_K_FINAL]

                if not retrieved_docs:
                    return []

                sentence_pairs = [(current_query, doc.page_content) for doc in retrieved_docs]
                scores = self.reranker.predict(sentence_pairs)

                ranked_indices = np.argsort(scores)[::-1]
                ranked_docs = [retrieved_docs[i] for i in ranked_indices]

                logger.info(f"Re-ranked {len(retrieved_docs)} documents, returning top {Config.RAG_TOP_K_FINAL}.")
                return ranked_docs[:Config.RAG_TOP_K_FINAL]

            rag_chain_for_patient = (
                {"context": ensemble_retriever, "question": RunnablePassthrough()}
                | RunnablePassthrough.assign(context=RunnableLambda(lambda x: _rerank_documents_local(x["context"], x["question"])))\
                | ChatPromptTemplate.from_template(self._get_rag_prompt_template())\
                | self.ollama_llm\
                | StrOutputParser()
            )

            try:
                response = rag_chain_for_patient.invoke(query)
                return response
            except Exception as e:
                logger.error(f"Error invoking RAG chain for patient {patient_id}: {e}")
                return f"Error generating summary for patient {patient_id}: {e}"


    # --- Execution for RAG System Setup ---
    logger.info("Preparing documents for vector store...")
    documents_for_vectorstore: List[Document] = []

    text_splitter_for_vectorstore = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE_CLASSIFICATION,
        chunk_overlap=Config.CHUNK_OVERLAP_CLASSIFICATION,
        length_function=lambda text: len(thematic_tokenizer.encode(text, add_special_tokens=True)),
    )

    for index, row in patients_df.iterrows():
        original_narrative = row['preprocessed_narrative']
        patient_id = row['Patient_ID']

        theme_names = row.get('labels_as_names', [])
        themes_for_metadata = ", ".join(theme_names) if theme_names else "None"

        sentiment_info = row.get('sentiment_and_emotions', {})
        overall_sentiment_label = sentiment_info.get('overall_sentiment', {}).get('label', 'N/A')
        overall_sentiment_score = round(sentiment_info.get('overall_sentiment', {}).get('score', 0.0), 2)
        emotions_detected = json.dumps(sentiment_info.get('emotions', {}))

        mh_signals_data = row.get('mental_health_linguistic_signals', {})
        mental_health_keywords = json.dumps(mh_signals_data)

        chunks = text_splitter_for_vectorstore.split_text(original_narrative)
        if not chunks and original_narrative.strip():
            chunks = [original_narrative]
        elif not chunks:
            continue

        for i, chunk in enumerate(chunks):
            documents_for_vectorstore.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "patient_id": patient_id,
                        "themes": themes_for_metadata,
                        "chunk_index": i,
                        "overall_sentiment_label": overall_sentiment_label,
                        "overall_sentiment_score": overall_sentiment_score,
                        "emotions_detected": emotions_detected,
                        "mental_health_keywords": mental_health_keywords
                    }
                )
            )

    logger.info(f"Number of documents for vectorstore: {len(documents_for_vectorstore)}")

    rag_system = RAGSystem(
        embedding_model_name=Config.EMBEDDING_MODEL_NAME,
        chroma_db_dir=Config.CHROMA_DB_DIR,
        ollama_base_url=Config.OLLAMA_BASE_URL,
        ollama_model_name=Config.OLLAMA_MODEL_NAME,
        thematic_tokenizer_instance=thematic_tokenizer,
        reranker_model_name=Config.RERANKER_MODEL_NAME
    )

    if documents_for_vectorstore:
        rag_system.populate_vectorstore(documents_for_vectorstore)
    else:
        logger.error("No documents generated for vectorstore. Cannot populate. Check data loading and processing in prior cells.")

    QOL_THEMES = Config.QOL_THEMES


    class PatientReportGenerator:
        def __init__(self, thematic_model: AutoModelForSequenceClassification, thematic_tokenizer: AutoTokenizer, nlp_analysis_service: Any, qol_themes: List[str], mlb_thematic_instance: MultiLabelBinarizer):
            self.thematic_model = thematic_model
            self.thematic_tokenizer = thematic_tokenizer
            self.nlp_analysis_service = nlp_analysis_service
            self.qol_themes = qol_themes
            self.mlb_thematic = mlb_thematic_instance
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.thematic_model.to(self.device)

        def generate_comprehensive_patient_report(self, patient_id: str, patients_df: pd.DataFrame, rag_system: Any) -> str:
            """Generates a comprehensive QoL report for a given patient ID."""
            patient_data = patients_df[patients_df['Patient_ID'] == patient_id]

            if patient_data.empty:
                logger.error(f"Patient ID '{patient_id}' not found in the dataset.")
                return f"Error: Patient ID '{patient_id}' not found."

            patient_row = patient_data.iloc[0]
            report_parts = []

            report_parts.append("# Patient Quality of Life Report 📄")
            report_parts.append(f"### Basic Information (Patient ID: **{patient_id}**)\\n")

            metadata = patient_row.get('Metadata', {})

            logger.info(f"DEBUG: 'Able_to_lie_flat_comfortably' from metadata for {patient_id}: {metadata.get('Able_to_lie_flat_comfortably', 'KEY_NOT_FOUND')}")

            report_parts.append(f"- **Name:** {patient_id}")
            report_parts.append(f"- **Age:** {metadata.get('Age', 'N/A')}")
            report_parts.append(f"- **Gender:** {metadata.get('Gender', 'N/A')}")
            report_parts.append(f"- **Ethnicity:** {metadata.get('Ethnicity', 'N/A')}")
            report_parts.append(f"- **Marital Status:** {metadata.get('Marital_Status', 'N/A')}")
            report_parts.append(f"- **Occupation:** {metadata.get('Job_Title', 'N/A')} ({metadata.get('Occupation_Category', 'N/A')})\\n")

            report_parts.append(f"- **Previous Hernia Repairs:**")
            if metadata.get('Previous_Hernia_Repairs'):
                for repair in metadata['Previous_Hernia_Repairs']:
                    report_parts.append(
                        f"   - Year: {repair.get('Year', 'N/A')}, Type: {repair.get('Type', 'N/A')}, "
                        f"Mesh: {repair.get('Mesh_Used', 'N/A')}, Wound Breakdown: {repair.get('Wound_Breakdown', 'N/A')}, "
                        f"Healing Time: {repair.get('Healing_Time', 'N/A')}"
                    )
            else:
                report_parts.append("   - None reported.")

            report_parts.append("\n- **Medical History:**")
            medical_history = metadata.get('Medical_History', {})
            if medical_history:
                for condition, status in medical_history.items():
                    if status not in ["No", "N/A", "", None] and condition != "Prior_Major_Surgeries":
                        report_parts.append(f"   - {condition.replace('_', ' ').title()}: {status}")
                if medical_history.get('Prior_Major_Surgeries'):
                    report_parts.append(f"   - Prior Major Surgeries: {', '.join(medical_history['Prior_Major_Surgeries'])}\n")
                else:
                    report_parts.append("   - No prior major surgeries reported.\n")
            else:
                report_parts.append("   - No medical history reported.\n")

            report_parts.append("\n- **Current Medications:**")
            medications = metadata.get('Medications', {})
            if medications:
                for med, dosage in medications.items():
                    report_parts.append(f"   - {med}: {dosage}")
            else:
                report_parts.append("   - No current medications reported.")

            report_parts.append(f"- **Able to lie flat comfortably:** {metadata.get('Able_to_lie_flat_comfortably', 'N/A')}")
            report_parts.append(f"- **QoL Areas Affected (Self-reported from Metadata):** {', '.join(metadata.get('QoL_Areas_Affected', ['None specified']))}\\n")

            # AI-CLASSIFIED QUALITY OF LIFE THEMES
            report_parts.append("## AI-Classified Quality of Life Themes 🤖\\n")
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
                    report_parts.append(f"- **Detected Themes:** {', '.join(predicted_theme_names)}")
                    confidence_scores_str = ", ".join([f"{self.qol_themes[idx]}: {probabilities[idx]:.2f}" for idx in predicted_labels_indices])
                    report_parts.append(f"   (Confidence Scores: {confidence_scores_str})")
                else:
                    report_parts.append("- **Detected Themes:** No specific themes confidently classified by AI (all scores below 0.5).\\n")
                    if len(probabilities) > 0:
                        top_idx = np.argmax(probabilities)
                        report_parts.append(f"   (Highest predicted: {self.qol_themes[top_idx]}: {probabilities[top_idx]:.2f})\\n")
                    else:
                        report_parts.append("   (No probabilities available.)\\n")
            except Exception as e:
                logger.error(f"Error during thematic classification for patient {patient_id}: {e}")
                report_parts.append("- **Detected Themes:** Error during classification.\\n")

            report_parts.append("\n## Sentiment and Emotion Analysis 📊")
            if 'sentiment_and_emotions' in patient_row:
                overall_sentiment = patient_row['sentiment_and_emotions'].get('overall_sentiment', {})
                emotions = patient_row['sentiment_and_emotions'].get('emotions', {})

                report_parts.append(f"- *Overall Sentiment:* *{overall_sentiment.get('label', 'N/A').upper()}* (Score: {overall_sentiment.get('score', 0.0):.2f})")

                if emotions:

                    top_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:4])
                    emotion_str = ", ".join([f"**{label.capitalize()}**: {score:.2f}" for label, score in top_emotions.items()])
                    report_parts.append(f"- *Key Emotions Detected (Average Intensity):* {emotion_str}")
                else:
                    report_parts.append("- No specific emotions confidently detected.")
            else:
                report_parts.append("- Sentiment and emotion analysis results not available.")


            report_parts.append("\n### MENTAL HEALTH PATTERN DETECTION (ZERO-SHOT CLASSIFICATION)")

            if 'mental_health_linguistic_signals' in patient_row:
                mh_signals = patient_row['mental_health_linguistic_signals']
                if mh_signals and 'details' in mh_signals:
                    sorted_patterns = sorted(mh_signals['details'].items(), key=lambda item: item[1], reverse=True)[:3]
                    pattern_str = ", ".join([f"*{label.capitalize()}*: {score:.2f}\n" for label, score in sorted_patterns])
                    report_parts.append(f"- *Top Detected Mental Health Patterns:* {pattern_str}")
                else:
                    report_parts.append("- No specific mental health linguistic signals detected.\n")
            else:
                report_parts.append("- Mental health linguistic signals not available.\n")


            report_parts.append("\n## AI-Generated Comprehensive QoL Summary (via RAG) 📝\n")

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

                    rag_summary = rag_system.get_rag_summary_for_patient(patient_id, rag_query)
                    report_parts.append(rag_summary)
                except Exception as e:
                    logger.error(f"Error during RAG summary generation for patient {patient_id}: {e}")
                    report_parts.append(f"Error: Unable to generate RAG summary for {patient_id}.")
            else:
                logger.warning("RAG system not initialized. Cannot generate comprehensive summary.")
                report_parts.append("RAG system not initialized. Cannot generate comprehensive summary.")

            return "\n".join(report_parts)

    logger.info("\n--- Generating a Sample Comprehensive QoL Report ---\n")


    sample_patient_id = patients_df['Patient_ID'].iloc[0]

    report_generator = PatientReportGenerator(
        thematic_model=thematic_model,
        thematic_tokenizer=thematic_tokenizer,
        nlp_analysis_service=nlp_analysis_service,
        qol_themes=QOL_THEMES,
        mlb_thematic_instance=mlb_thematic
    )

    full_report = report_generator.generate_comprehensive_patient_report(
        sample_patient_id,
        patients_df,
        rag_system)

    print(full_report)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()