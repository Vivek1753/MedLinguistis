# 🧠 Quality of Life (QoL) Patient Report Generation Pipeline

This project is an end-to-end offline-ready NLP pipeline designed to analyze patient narratives and medical metadata to generate comprehensive Quality of Life (QoL) reports. It integrates transformer-based classification, sentiment analysis, vector search via LangChain, and FastAPI-based API endpoints for interaction.

---

## 🔧 Tech Stack

- **Python Version:** 3.13.5
- **Frameworks & Libraries:**
  - Transformers (Hugging Face)
  - LangChain
  - ChromaDB
  - FastAPI
  - scikit-learn
  - PyTorch
  - Uvicorn
  - TextBlob
  - Matplotlib, Wordcloud

---

## 🚀 Getting Started

### 1. 📥 Clone the Repository

```bash
cd your-repo-name
git clone https://github.com/Vivek1753/MedLinguistis.git
```
### 2. Create and Activate a Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```
### 4. Download and Setup Models
You have two options:
  
Option A: Offline Setup
- If you're working in a fully offline environment:
```bash
python model.py
```
- This will download and configure all models from Hugging Face
- Note: These directories are excluded from GitHub using `.gitignore`

Option B: Download Models from Google Drive
- Download the required models from the following link and place them in the appropriate folders: https://drive.google.com/drive/folders/1rgA-oLbE2fkktZGeBsPFjGqiMPFYl5Vi?usp=sharing

### 5. Run the FastAPI Application
``` bash
uvicorn main:app --reload
```
Default: http://127.0.0.1:8000

## API Usage (via Postman)

- **Method:** `POST`
- **Endpoint:** `http://127.0.0.1:8000/generate_report/`
- **Headers:**  `Content-Type: application/json`
- **Body Type:**  `raw` → `JSON`
- **Example JSON Input:**  Located inside the `testing_data/` folder.

## Project Structure
``` bash
├── chroma_db/                    # Chroma vector store 
├── data/                         # Raw or processed data
├── offline_models/               # ⚠️ [Ignored] Local model storage
├── qol_classifier_fine_tuned/    # ⚠️ [Ignored] Fine-tuned models
├── reports/                      # Generated patient reports
├── report_images/                # Images/visuals for reports
├── testing_data/                 # Sample patient data in JSON format
├── pipeline.py                   # Pipeline configuration & orchestration
├── model.py                      # Loads models & sets up offline mode
├── main.py                       # FastAPI app entry point
├── visualize.py                  # Additional utility scripts
├── requirements.txt              # Dependency list
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```





