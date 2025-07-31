import os
import re
import uuid
import json
import nltk
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from sklearn.preprocessing import MultiLabelBinarizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, DataCollatorWithPadding

# from langchain_ollama import OllamaLLM as Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from visulize import (
    plot_emotions, 
    plot_hernia_repairs,
    plot_wordcloud,
)

# --- Configuration Management ---
class Config:
    DATASET_PATH_A = "data/patients_100a.json"
    MODEL_SAVE_DIR = "qol_classifier_fine_tuned"
    CHROMA_DB_DIR = "chroma_db"
    LOGGING_DIR = "logs_thematic"
    RESULTS_DIR = "results_thematic"
    HYPEROPT_LOG_DIR = "hyperopt_logs"

    # THEMATIC_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    # SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
    # EMOTION_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'
    # EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    # RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    THEMATIC_MODEL_NAME = r"C:\AI Project\RAG\offline_models\pubmed_bert"
    SENTIMENT_MODEL_NAME = r"C:\AI Project\RAG\offline_models\finbert"
    EMOTION_MODEL_NAME = r"C:\AI Project\RAG\offline_models\twitter_emotion"
    EMBEDDING_MODEL_NAME = r"C:\AI Project\RAG\offline_models\bge_embedding"
    RERANKER_MODEL_NAME = r"C:\AI Project\RAG\offline_models\msmarco_cross_encoder"

    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL_NAME = "llama3:8b-instruct-q8_0"

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

        # logger.info(f"Loading zero-shot classification model: facebook/bart-large-mnli on device: {self.device}")
        # self.mental_health_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=self.device)
        # self.mental_health_candidate_labels = [
        #     "anxiety", "hopelessness", "social withdrawal", "frustration or irritability",
        #     "low mood or depression", "feeling overwhelmed", "rumination"
        # ]

        logger.info(f"Loading zero-shot classification model: LOCAL - bart_large_mnli on device: {self.device}")

        # Local path to the model directory
        local_bart_path = r"C:\AI Project\RAG\offline_models\bart_large_mnli"

        # Load model and tokenizer from local path
        self.mental_health_classifier = pipeline(
            "zero-shot-classification",
            model=AutoModelForSequenceClassification.from_pretrained(local_bart_path),
            tokenizer=AutoTokenizer.from_pretrained(local_bart_path),
            device=self.device
        )

        # Candidate labels remain the same
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
            "Patient_ID": patient_id,
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

        print(f"\n--- Patient ID: {patient_id} ---")
        print("Narrative:", text_for_classification[:200]) 
        print("QOL Themes:", self.qol_themes)

        try:
            # print("Logits:", logits)
            # print("Probabilities:", probabilities)
            # print("Predicted Indices:", predicted_labels_indices)
            # print("Predicted Themes:", predicted_theme_names)

            inputs = self.thematic_tokenizer(text_for_classification, return_tensors="pt", truncation=True, padding="max_length", max_length=Config.MAX_MODEL_INPUT_LENGTH)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.thematic_model(**inputs).logits

            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            predicted_labels_indices = np.where(probabilities > 0.5)[0]

            print(f"[{patient_id}] Logits: {logits}")
            print(f"[{patient_id}] Probabilities: {probabilities}")

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

        wordcloud_data = plot_wordcloud(rag_summary_text)
        top_words = wordcloud_data["top_words"]

        # Now construct report_json
        report_json = {
            "metadata": metadata_section,
            "detected_themes": detected_themes_section,
            "sentiment_and_emotion_analysis": sentiment_section,
            "zero_shot_classification": zero_shot_section,
            "qol_summary": rag_summary_text,
            "wordcloud": top_words
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