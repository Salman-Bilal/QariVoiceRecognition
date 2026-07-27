# 🚀 Quick Start Guide

Get the Qari Voice Recognition System running in 3 simple steps!

## ⚡ Quick Installation

### **Windows**

```bash
# 1. Install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the application
python app.py
```

### **Linux / Mac**

```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the application
python app.py
```

## 🌐 Access the Application

The browser will automatically open. If not, go to:

- **Web Interface:** http://localhost:8000/static/index.html
- **API Docs:** http://localhost:8000/docs

## 📝 Usage Instructions

### **Option 1: Identify Which Qari You Sound Like**

1. Click **"Identify Qari"** tab
2. Upload or record your Quran recitation (any Surah, any length)
3. Click **"Identify Qari"**
4. See results showing which Qari your voice matches!

### **Option 2: Analyze Your Recitation Quality**

1. Click **"Analyze Recitation"** tab
2. Upload your audio file
3. Select the Surah you recited
4. Choose a reference Qari (optional, defaults to Abdul Basit)
5. Click **"Analyze Recitation"**
6. Get detailed scores on timing, melody, and breathing!

### **Option 3: Compare Against All Qaris**

1. Click **"Compare All Qaris"** tab
2. Upload your recitation
3. Click **"Compare Against All Qaris"**
4. See a full breakdown comparing you to all 12 Qaris!

## 🎤 Recording Tips

- **Duration:** Record at least 10-15 seconds for best results
- **Quality:** Use a good microphone, record in a quiet room
- **Distance:** Speak clearly, not too far from the mic
- **Format:** WAV recommended, but MP3, M4A, FLAC also supported
- **Volume:** Avoid clipping (recording too loud)

## 🐛 Troubleshooting

### Server won't start

```bash
# Make sure port 8000 is not in use
# On Windows:
netstat -ano | findstr :8000

# On Linux/Mac:
lsof -i :8000
```

### "Module not found" error

```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Audio not being processed

- Check file format (WAV, MP3, M4A, FLAC only)
- Check file size (must be under 50MB)
- Check audio duration (must be 1-10 minutes)
- Make sure audio is not silent

### Models downloading slowly

The ECAPA-TDNN model (~80MB) will download automatically on first run. This may take a few minutes depending on your internet connection.

## 📖 For More Information

See [README.md](README.md) for complete documentation.

## 💡 Example Commands

### Using API with curl

```bash
# Identify Qari
curl -X POST http://localhost:8000/api/identify-qari \
  -F "audio_file=@my_recitation.wav" \
  -F "top_k=5"

# Analyze Recitation
curl -X POST http://localhost:8000/api/analyze-recitation \
  -F "audio_file=@my_recitation.wav" \
  -F "surah_name=1_Surah_Fatiha" \
  -F "reference_qari=Mishary Al-Fasay"
```

### Using Python requests

```python
import requests

# Identify Qari
with open('my_recitation.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/identify-qari',
        files={'audio_file': f},
        data={'top_k': 5}
    )
    print(response.json())
```

## 🎯 Next Steps

1. Try the three different analysis modes
2. Experiment with different Qaris as reference
3. Record multiple times to see consistency
4. Check the API documentation at `/docs` for advanced usage

---

**Enjoy analyzing your Quranic recitations!** 🎙️✨
