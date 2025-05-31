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

🚀 Features
Supports local files, direct media URLs, and Loom videos

Automatically splits large files into 1-minute chunks

Early stopping for faster analysis

Confidence-based predictions

Interactive Streamlit dashboard

⚙️ Installation
Clone the repository:

bash
git clone https://github.com/your-username/accent-analyzer.git
cd accent-analyzer
Install dependencies:

bash
pip install -r requirements.txt
Install FFmpeg (required for audio processing):

bash
# On Ubuntu/Debian
sudo apt install ffmpeg

# On macOS
brew install ffmpeg
🖥️ Usage
Run the Streamlit app:

bash
streamlit run app.py
The app will open in your browser at http://localhost:8501

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