# 🧠 Quality of Life (QoL) Patient Report Generation Pipeline

This project is a complete, offline-capable NLP system that processes patient narratives and medical metadata to produce structured Quality of Life (QoL) reports. It combines transformer-based thematic classification, sentiment and emotion analysis, and vector search using LangChain for context-aware insights. A FastAPI backend provides easy API access, enabling seamless integration with healthcare applications and local deployments without reliance on external APIs.

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






