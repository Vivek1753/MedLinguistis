import os
import json
import torch
import shutil
import uvicorn
import logging
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
# Prepare RAG docs
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline import (
    Config,
    set_seed,
    PatientDataProcessor,
    NLPAnalysisService,
    RAGSystem,
    PatientReportGenerator,
    MultiLabelBinarizer
)

# --- Setup ---
app = FastAPI()
set_seed(Config.RANDOM_SEED)

# Logging
logger = logging.getLogger("uvicorn")
logger.info("FastAPI server initialized.")

# Tokenizer & Model Load
thematic_tokenizer = AutoTokenizer.from_pretrained(Config.THEMATIC_MODEL_NAME)
mlb_thematic = MultiLabelBinarizer(classes=Config.QOL_THEMES)
mlb_thematic.fit([Config.QOL_THEMES])

try:
    thematic_model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_SAVE_DIR)
    thematic_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Thematic model loaded from {Config.MODEL_SAVE_DIR}")
except Exception:
    thematic_model = AutoModelForSequenceClassification.from_pretrained(
        Config.THEMATIC_MODEL_NAME,
        num_labels=len(Config.QOL_THEMES),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )
    thematic_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Thematic model loaded from HuggingFace: {Config.THEMATIC_MODEL_NAME}")

try:
    sentence_embedder = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    logger.info("SentenceTransformer for keyword expansion loaded.")
except Exception as e:
    logger.warning(f"Could not load SentenceTransformer for keyword expansion: {e}. Keyword expansion might be limited.")
    sentence_embedder = None

# Initialize NLP services and RAG system once
nlp_analysis_service = NLPAnalysisService(
    thematic_tokenizer_instance=thematic_tokenizer,
    sentiment_model_name=Config.SENTIMENT_MODEL_NAME,
    emotion_model_name=Config.EMOTION_MODEL_NAME,
    sentence_embedder=sentence_embedder
)

rag_system = RAGSystem(
    embedding_model_name=Config.EMBEDDING_MODEL_NAME,
    chroma_db_dir=Config.CHROMA_DB_DIR,
    ollama_base_url=Config.OLLAMA_BASE_URL,
    ollama_model_name=Config.OLLAMA_MODEL_NAME,
    thematic_tokenizer_instance=thematic_tokenizer,
    reranker_model_name=Config.RERANKER_MODEL_NAME
)

@app.post("/generate_report/")
async def generate_qol_report(request: Request):
    try:
        patient_data = await request.json()

        # Convert to DataFrame (1 row per patient)
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, list):
            df = pd.DataFrame(patient_data)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported JSON format."})

        data_processor = PatientDataProcessor(Config.QOL_THEMES, mlb_thematic, thematic_tokenizer)
        print(f"DEBUG: Loaded {len(df)} patients from dataset.")
        print(f"DEBUG: Columns in dataset: {df.columns.tolist()}")
        df_extracted = data_processor.extract_narrative_fields(df.copy())
        print(f"DEBUG: Extracted narratives for {len(df_extracted)} patients.")
        print(f"DEBUG: Columns after extraction: {df_extracted.columns.tolist()}")
        print(df_extracted)  # Display first few rows of extracted data
        df_prepared, _, _ = data_processor.prepare_data_for_classification(df_extracted)
        logger.info(f"Final dataset prepared with {len(df_prepared)} patients and {len(df_prepared['preprocessed_narrative'])} narratives.")
        logger.info(f"Example preprocessed narrative for first patient: {df_prepared['preprocessed_narrative']}")

        # Apply sentiment and emotion analysis
        logger.info("Applying sentiment and emotion analysis to narrative...")
        df_prepared['sentiment_and_emotions'] = df_prepared['preprocessed_narrative'].apply(
            nlp_analysis_service.analyze_sentiment_and_emotions_of_chunks
        )
        logger.info("Sentiment and emotion analysis completed.")

        # Print result for the single patient
        patient_row = df_prepared.iloc[0]
        print(f"\n--- Sentiment and Emotion Analysis for Patient: {patient_row['Patient_ID']} ---")
        sentiment = patient_row['sentiment_and_emotions'].get('overall_sentiment', {})
        emotions = patient_row['sentiment_and_emotions'].get('emotions', {})

        print(f"  Sentiment: {sentiment.get('label', 'N/A')} (Score: {sentiment.get('score', 0.0):.2f})")
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emo, score in sorted_emotions[:3]:
                print(f"  - {emo.capitalize()}: {score:.2f}")
        else:
            print("  No dominant emotions detected.")

        # Apply mental health pattern detection
        logger.info("Applying mental health linguistic signal detection (Zero-Shot Classification)...")
        df_prepared['mental_health_linguistic_signals'] = df_prepared['preprocessed_narrative'].apply(
            nlp_analysis_service.identify_mental_health_patterns_advanced
        )
        logger.info("Mental health linguistic signal detection completed.")

        # Print zero-shot result
        mh_result = df_prepared.iloc[0]['mental_health_linguistic_signals']
        print(f"\n--- Mental Health Linguistic Patterns for Patient: {patient_row['Patient_ID']} ---")
        if mh_result and 'details' in mh_result:
            sorted_mh = sorted(mh_result['details'].items(), key=lambda x: x[1], reverse=True)
            for label, score in sorted_mh[:3]:
                print(f"  - {label.capitalize()}: {score:.2f}")
        else:
            print("  No significant mental health patterns detected.")
        print("-" * 30)

        chroma_db_path = Config.CHROMA_DB_DIR
        logger.info(f"Checking and cleaning ChromaDB directory at: {chroma_db_path}")

        if os.path.exists(chroma_db_path):
            try:
                shutil.rmtree(chroma_db_path)
                logger.info("✅ Existing ChromaDB directory removed.")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove ChromaDB directory: {e}")
        else:
            logger.info("ℹ️ No existing ChromaDB directory found. Skipping cleanup.")


        documents_for_vectorstore = []
        text_splitter_for_vectorstore = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE_CLASSIFICATION,
            chunk_overlap=Config.CHUNK_OVERLAP_CLASSIFICATION,
            length_function=lambda text: len(thematic_tokenizer.encode(text, add_special_tokens=True)),
        )

        for index, row in df_prepared.iterrows():
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

        if documents_for_vectorstore:
            rag_system.populate_vectorstore(documents_for_vectorstore)
        else:
            logger.error("No documents generated for vectorstore. Cannot populate. Check data loading and processing in prior cells.")

        report_generator = PatientReportGenerator(
            thematic_model=thematic_model,
            thematic_tokenizer=thematic_tokenizer,
            nlp_analysis_service=nlp_analysis_service,
            qol_themes=Config.QOL_THEMES,
            mlb_thematic_instance=mlb_thematic
        )

        report = report_generator.generate_comprehensive_patient_report(
            # patient_id=sample_patient_id,
            patients_df=df_prepared,
            rag_system=rag_system,
            all_documents=documents_for_vectorstore
        )

        print(f"patient full report: {report}")

        return JSONResponse(
            status_code=200,
            content={
                "Patient_ID": df_prepared['Patient_ID'].iloc[0],
                "report": report
            }
        )
    
    except Exception as e:
        logger.error(f"Invalid JSON input: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON input."})

if __name__ == "__main__":
    uvicorn.run(app, host="10.122.203.150", port=8000, reload=True)

# uvicorn main:app --reload

