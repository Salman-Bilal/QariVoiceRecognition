# 🎙️ Qari Voice Recognition System

> **AI-Powered Quranic Recitation Analysis & Voice Identification**

An intelligent system that identifies which famous Qari (Quran reciter) a person's voice most closely resembles and provides detailed feedback on recitation quality including timing, melody, and breathing patterns.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This Final Year Project implements a complete machine learning pipeline for Quranic recitation analysis. The system can:

1. **Identify Qari Voice Similarity** - Upload your recitation and discover which of 12 famous Qaris your voice most resembles
2. **Analyze Recitation Quality** - Get detailed scores on timing, melody (pitch), and breathing patterns
3. **Compare Against All Qaris** - See a comprehensive breakdown of your similarity to every Qari in the database

### **Main Purpose**

To provide aspiring Quran reciters with an AI-powered tool that:
- Helps them understand their vocal characteristics
- Provides objective feedback on recitation quality
- Identifies areas for improvement through comparison with professional Qaris

---

## ✨ Features

### **Core Functionality**

- 🎤 **Voice Fingerprinting** - Uses ECAPA-TDNN deep learning model to extract unique vocal characteristics
- 📊 **Multi-Dimensional Analysis** - Evaluates timing (DTW), melody (pitch contour), and breathing (pause detection)
- 🎯 **High Accuracy** - Achieves 99.83% test accuracy on Qari identification
- 🌐 **Web Interface** - Modern, responsive UI with drag-and-drop upload
- 🎙️ **Live Recording** - Record directly in browser (no file upload needed)
- 📈 **Interactive Visualizations** - Charts and graphs powered by Chart.js
- 🔄 **Dynamic Qari Selection** - Compare against any of the 12 Qaris

### **Technical Features**

- RESTful API built with FastAPI
- Real-time audio processing with librosa
- Cosine similarity-based voice matching
- Dynamic Time Warping for timing analysis
- Adaptive breath detection with configurable thresholds

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JS + Chart.js
│   (Browser)     │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   FastAPI       │  REST API Server
│   Backend       │  (Port 8000)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬─────────────┐
    ▼         ▼          ▼             ▼
┌────────┐ ┌─────┐  ┌────────┐  ┌──────────┐
│Matching│ │Audio│  │Analysis│  │  Models  │
│ Engine │ │Proc.│  │ Metrics│  │ (ECAPA)  │
└────────┘ └─────┘  └────────┘  └──────────┘
```

### **Pipeline Overview**

1. **Data Collection** - Gathered recordings from 12 Qaris × 8 Surahs
2. **Preprocessing** - Normalization, denoising, silence removal
3. **Segmentation** - Split into 4-second windows with 2-second overlap
4. **Feature Extraction** - ECAPA-TDNN embeddings (192-dim) + MFCC features
5. **Model Training** - Neural network classifiers (Track A & B)
6. **Deployment** - FastAPI backend + Web frontend

---

## 🚀 Installation

### **Prerequisites**

- Python 3.11 or higher
- pip package manager
- 8GB+ RAM recommended
- (Optional) CUDA-capable GPU for faster processing

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/QariVoiceRecognition.git
cd QariVoiceRecognition
```

### **Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch or TensorFlow, install manually:

```bash
# PyTorch (CPU version)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# PyTorch (GPU version - CUDA 11.8)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### **Step 4: Download Pre-trained Models**

Models are automatically downloaded on first run. Alternatively, download manually:

```bash
python -c "from matching.similarity import _load_classifier; _load_classifier()"
```

### **Step 5: Prepare Dataset (Optional - for training)**

If you want to retrain models:

1. Place raw audio files in `dataset/raw/{Qari_Name}/`
2. Run preprocessing pipeline:

```bash
python run_pipeline.py
```

---

## 💻 Usage

### **Quick Start - Web Interface**

1. **Start the API server:**

```bash
python api/main.py
```

Server will start at `http://localhost:8000`

2. **Open your browser:**

```
http://localhost:8000/static/index.html
```

3. **Upload or record audio** and analyze!

### **Using the API Directly**

#### **Health Check**

```bash
curl http://localhost:8000/health
```

#### **Identify Qari**

```bash
curl -X POST http://localhost:8000/api/identify-qari \
  -F "audio_file=@my_recitation.wav" \
  -F "top_k=5"
```

**Response:**
```json
{
  "success": true,
  "top_match": {
    "qari": "Mishary Al-Fasay",
    "similarity": 87.32,
    "confidence": "high"
  },
  "all_matches": [...]
}
```

#### **Analyze Recitation**

```bash
curl -X POST http://localhost:8000/api/analyze-recitation \
  -F "audio_file=@my_recitation.wav" \
  -F "surah_name=1_Surah_Fatiha" \
  -F "reference_qari=Abdul Basit Abdul Samad"
```

**Response:**
```json
{
  "success": true,
  "report": {
    "overall_score": 82.5,
    "timing": {"timing_score": 78.3, ...},
    "melody": {"melody_score": 85.1, ...},
    "breath": {"breath_score": 84.1, ...}
  }
}
```

### **Command-Line Usage**

#### **Test Your Voice**

```bash
# Edit analysis/test_my_voice.py with your audio path
python analysis/test_my_voice.py
```

#### **Calibrate System**

```bash
python analysis/calibrate.py
```

#### **Evaluate Models**

```bash
# Evaluate similarity engine
python matching/evaluate_similarity.py

# Check breath detection
python analysis/diagnose_breathe.py
```

---

## 📚 API Documentation

### **Base URL:** `http://localhost:8000`

### **Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check + model status |
| GET | `/api/list-qaris` | Get available Qaris |
| GET | `/api/available-surahs` | Get available Surahs |
| POST | `/api/identify-qari` | Identify which Qari user sounds like |
| POST | `/api/analyze-recitation` | Analyze recitation quality |
| POST | `/api/compare-all-qaris` | Compare against all 12 Qaris |

### **Interactive API Docs**

Once server is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 📂 Project Structure

```
QariVoiceRecognition/
│
├── api/                          # FastAPI backend
│   └── main.py                   # API server with all endpoints
│
├── frontend/                     # Web interface
│   ├── index.html                # Main HTML page
│   ├── style.css                 # Styling
│   └── app.js                    # JavaScript logic
│
├── analysis/                     # Analysis modules
│   ├── aggregate.py              # Main report generator
│   ├── timing.py                 # DTW-based timing analysis
│   ├── melody.py                 # Pitch contour analysis
│   ├── breath.py                 # Breathing/pause detection
│   └── config.py                 # Configuration constants
│
├── matching/                     # Voice matching engine
│   ├── similarity.py             # Cosine similarity matching
│   ├── build_reference_embeddings.py
│   ├── evaluate_similarity.py
│   └── reference_embeddings.pkl  # Pre-computed Qari embeddings
│
├── models/                       # ML models
│   ├── train_baseline_nn.py      # Track B (MFCC) training
│   ├── train_embedding_classifier.py  # Track A (embedding) training
│   └── checkpoints/              # Saved model weights
│
├── feature_extraction/           # Audio feature extraction
│   ├── embeddings.py             # ECAPA-TDNN embedding extraction
│   ├── mfcc_pooled.py            # MFCC feature extraction
│   └── scaler.pkl                # Feature normalizer
│
├── preprocessing/                # Data preprocessing
│   ├── pipeline.py               # Main preprocessing pipeline
│   ├── normalize.py              # Audio normalization
│   ├── segment.py                # Audio segmentation
│   └── noise.py                  # Noise reduction
│
├── dataset/                      # Audio dataset
│   ├── raw/                      # Original recordings
│   ├── processed/                # Preprocessed audio
│   │   ├── normalized/           # Cleaned audio files
│   │   ├── chunks/               # 4-sec segments
│   │   └── embeddings/           # Extracted features
│   └── processed/
│       ├── manifest.csv          # File metadata
│       └── splits.csv            # Train/val/test splits
│
├── evaluation/                   # Model evaluation
│   ├── metrics.py                # Evaluation metrics
│   └── reports/                  # Performance reports
│       ├── track_a_metrics.json  # Embedding model results
│       ├── track_b_metrics.json  # MFCC model results
│       └── similarity_engine_metrics.json
│
├── logs/                         # Application logs
│   └── api.log                   # API request logs
│
├── run_pipeline.py               # Main preprocessing script
├── config.yaml                   # System configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📊 Model Performance

### **Track A - Voice Embeddings (ECAPA-TDNN)**

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 98.39% | **99.83%** |
| Precision | - | 99.82% |
| Recall | - | 99.84% |
| F1-Score | - | 99.83% |

**Key Achievement:** Only 5 misclassifications out of 2,939 test clips!

### **Track B - MFCC Features**

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 95.86% | **97.14%** |
| Precision | - | 96.90% |
| Recall | - | 97.18% |
| F1-Score | - | 96.96% |

### **Similarity Engine**

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **99.73%** |
| Top-5 Accuracy | **100%** |

**Interpretation:** The system correctly identifies the Qari as the #1 match 99.73% of the time, and the correct Qari is always within the top 5 predictions.

### **Dataset**

- **Qaris:** 12 world-famous reciters
- **Surahs:** 8 chapters per Qari
- **Total Audio Clips:** 2,939 (after segmentation)
- **Audio Format:** 16kHz, mono, WAV
- **Segment Length:** 4 seconds (2-second overlap)

---

## 🛠️ Technologies Used

### **Machine Learning**
- **Deep Learning:** PyTorch, TensorFlow
- **Speech Processing:** SpeechBrain, librosa
- **Feature Extraction:** ECAPA-TDNN, MFCC
- **Similarity Metrics:** Cosine similarity, DTW (Dynamic Time Warping)

### **Backend**
- **API Framework:** FastAPI
- **Server:** Uvicorn
- **Audio Processing:** librosa, soundfile, noisereduce

### **Frontend**
- **UI:** HTML5, CSS3, JavaScript (ES6+)
- **Visualization:** Chart.js
- **Icons:** Font Awesome

### **Data Science**
- **Data Manipulation:** pandas, NumPy
- **ML Library:** scikit-learn
- **Evaluation:** confusion matrix, precision/recall

---

## ⚠️ Limitations

### **Current Limitations**

1. **No Tajweed/Pronunciation Analysis**
   - System only evaluates acoustic properties (voice timbre, rhythm, pitch)
   - Does NOT check pronunciation correctness or Tajweed rules
   - Cannot detect mispronounced Arabic letters

2. **Limited Dataset**
   - Only 12 Qaris (many famous reciters not included)
   - Only 8 Surahs per Qari
   - May not generalize to Qaris outside training set

3. **Breathing Detection Sensitivity**
   - May miss very short breaths
   - Can be affected by background noise
   - Requires relatively clean audio

4. **Language Limitation**
   - Specifically designed for Quranic Arabic recitation
   - Not applicable to other languages or speech types

5. **Recording Quality Requirements**
   - Best results with clear, studio-quality audio
   - Performance degrades with poor quality recordings
   - Sensitive to background noise

### **Known Issues**

- Breath detection may return 0 score if audio is very quiet
- Timing score magic numbers need calibration per Surah length
- No user authentication system yet
- No database for storing user history

---

## 🔮 Future Improvements

### **High Priority**

- [ ] **Pronunciation Analysis** - Implement Arabic phoneme recognition
- [ ] **Tajweed Rule Checking** - Detect makharij, ghunnah, qalqalah errors
- [ ] **Expand Dataset** - Add more Qaris (target: 30+) and Surahs (target: 20+)
- [ ] **Mobile App** - iOS/Android native applications
- [ ] **User Accounts** - Save history, track progress over time

### **Medium Priority**

- [ ] **Real-time Processing** - Live feedback during recitation
- [ ] **Confidence Intervals** - Statistical significance for scores
- [ ] **Audio Augmentation** - Improve model robustness
- [ ] **Multi-language UI** - Arabic, Urdu, English interfaces
- [ ] **Offline Mode** - Desktop app without internet required

### **Low Priority**

- [ ] **Social Features** - Share results, leaderboards
- [ ] **Certification System** - Generate certificates for high scores
- [ ] **Advanced Visualizations** - Spectrograms, waveform displays
- [ ] **API Rate Limiting** - Production-ready security
- [ ] **Docker Container** - Easy deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Development Setup**

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **[Your Name]** - *Initial work* - Final Year Project

---

## 🙏 Acknowledgments

- **SpeechBrain Team** - For the excellent ECAPA-TDNN model
- **All Qaris** - Whose beautiful recitations made this project possible
- **Supervisors & Mentors** - For guidance throughout the project
- **Open Source Community** - For the amazing tools and libraries

---

## 📞 Contact

For questions or support, please contact:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/QariVoiceRecognition/issues)

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{qarivoicerecognition2026,
  title={Qari Voice Recognition System: AI-Powered Quranic Recitation Analysis},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/QariVoiceRecognition}}
}
```

---

<div align="center">

**Made with ❤️ for the Quran recitation community**

[⬆ Back to Top](#-qari-voice-recognition-system)

</div>
