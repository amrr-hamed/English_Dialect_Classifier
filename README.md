---
title: English Dialect Classifier
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Predicting English Dialect Using Speech Brain and Streamlit
license: apache-2.0
---
# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).


🎤 English Accent Analyzer
Streamlit App
PyTorch

A tool to identify English accents from audio/video sources with optimized processing for large files.

# 🎤 English Accent Analyzer  
A **Streamlit-based** tool powered by **PyTorch** to identify English accents from audio and video sources.

---

## 🚀 Key Features  
- 🎧 Upload local audio/video files or provide media URLs (e.g., Loom links)  
- 📦 Automatic splitting of large files into 1-minute chunks  
- ⚡ Early stopping for efficient analysis  
- 📊 Confidence-based predictions  
- 🖥️ Intuitive interactive Streamlit dashboard  

---

## ⚙️ Installation  

1. **Clone the repository:**
```bash
git clone https://github.com/amrr-hamed/English_Dialect_Classifier.git
cd English_Dialect_Classifier
```

2. **Install required Python packages:**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg (needed for audio processing):**

- **Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

- **macOS (using Homebrew):**
```bash
brew install ffmpeg
```

---

## 🖥️ How to Use  

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Access the app in your browser** at:  
[http://localhost:8501](http://localhost:8501)

3. **Inside the app:**
   - Upload an audio/video file **OR** paste a media URL  
   - Click **Analyze**  
   - View predicted dialects with confidence scores  
   - Use interactive UI controls to explore results


📥 Input Options
1. Upload a file
Supported formats:

Video: .mp4, .webm, .avi, .mov, .mkv, .m4v

Audio: .mp3, .wav, .m4a, .aac, .ogg, .flac

2. Provide a URL
Loom videos: https://www.loom.com/share/...

Direct media links: https://example.com/video.mp4

🔧 Optimizations for Large Files
The system automatically handles large files using these techniques:

Diagram
Code















Chunk Processing:

Audio is split into 1-minute segments

Only segments >10 seconds are processed

Enables parallel processing (future implementation)

Early Stopping:

Stops processing when 3 consecutive chunks agree with high confidence

Saves processing time for long files

Efficient Extraction:

Uses FFmpeg for fast audio extraction

Torchaudio fallback for compatibility

Direct streaming for URL sources

Confidence Threshold:

Only predictions >60% confidence are considered

Reduces false positives from noisy segments

📊 Example Output
Example Dashboard

The dashboard shows:

Predicted accent with confidence percentage

Confidence scores per minute

Accent distribution charts

Processing time metrics
