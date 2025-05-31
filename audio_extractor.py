import os
import sys
import tempfile
import warnings
import time
import shutil
import random

import torch
import torchaudio
import yt_dlp
from contextlib import contextmanager

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def extract_audio_from_video_url(video_url):
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio[abr<=64]',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }

    with suppress_stdout_stderr():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    for file in os.listdir(temp_dir):
        if file.endswith('.wav'):
            end_time = time.time()
            print(f"[⏱️] Audio extraction took {end_time - start_time:.2f} seconds.")
            return os.path.join(temp_dir, file)
    raise Exception("Failed to extract audio in WAV format")



def smart_chunk_audio(waveform, sample_rate, duration_minutes):
    """Smart chunking based on video duration"""
    total_duration = waveform.size(1) / sample_rate
    print(f"📏 Video duration: {total_duration/60:.1f} minutes")
    
    if duration_minutes <= 1:
        # Short videos: smaller chunks, process all
        chunk_length_sec = 10
        return chunk_audio_all(waveform, sample_rate, chunk_length_sec)
    
    elif duration_minutes <= 5:
        # Medium videos: normal chunks, skip some randomly
        chunk_length_sec = 20
        all_chunks = chunk_audio_all(waveform, sample_rate, chunk_length_sec)
        # Keep 70% of chunks randomly
        keep_ratio = 0.7
        num_keep = max(1, int(len(all_chunks) * keep_ratio))
        selected_chunks = random.sample(all_chunks, num_keep)
        print(f"📦 Selected {len(selected_chunks)} out of {len(all_chunks)} chunks")
        return selected_chunks
    
    else:
        # Long videos: strategic sampling from beginning, middle, end
        chunk_length_sec = 25
        return chunk_audio_strategic(waveform, sample_rate, chunk_length_sec)

def chunk_audio_all(waveform, sample_rate, chunk_length_sec=20):
    """Create all chunks from audio"""
    chunk_samples = chunk_length_sec * sample_rate
    total_samples = waveform.size(1)
    chunks = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.size(1) > sample_rate * 3:  # ignore very short chunks (3 sec minimum)
            chunks.append(chunk)
    return chunks

def chunk_audio_strategic(waveform, sample_rate, chunk_length_sec=25):
    """Strategic chunking for long videos - sample from beginning, middle, end"""
    total_samples = waveform.size(1)
    chunk_samples = chunk_length_sec * sample_rate
    
    chunks = []
    
    # Beginning: 2-3 chunks
    beginning_chunks = min(3, total_samples // chunk_samples)
    for i in range(beginning_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.size(1) > sample_rate * 3:
            chunks.append(chunk)
    
    # Middle: 2-3 chunks
    middle_start = total_samples // 2 - chunk_samples
    middle_chunks = min(3, 2)
    for i in range(middle_chunks):
        start = middle_start + (i * chunk_samples)
        end = min(start + chunk_samples, total_samples)
        if start >= 0 and start < total_samples:
            chunk = waveform[:, start:end]
            if chunk.size(1) > sample_rate * 3:
                chunks.append(chunk)
    
    # End: 2-3 chunks
    end_start = total_samples - (3 * chunk_samples)
    end_chunks = min(3, 3)
    for i in range(end_chunks):
        start = max(0, end_start + (i * chunk_samples))
        end = min(start + chunk_samples, total_samples)
        if start < total_samples:
            chunk = waveform[:, start:end]
            if chunk.size(1) > sample_rate * 3:
                chunks.append(chunk)
    
    print(f"📦 Strategic sampling: {len(chunks)} chunks from long video")
    return chunks

def prepare_audio(video_url):
    """Main function to extract and prepare audio chunks"""
    try:
        print(f"🎵 Extracting audio from video...")
        audio_path = extract_audio_from_video_url(video_url)
        print(f"✅ Audio extracted to: {audio_path}")

        print(f"🎯 Loading and preparing audio...")
        start = time.time()
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        end = time.time()
        print(f"[⏱️] Audio preparation took {end - start:.2f} seconds.")

        # # Apply simple VAD
        # print(f"🎤 Applying Voice Activity Detection...")
        # start = time.time()
        # waveform = simple_vad(waveform, sample_rate)
        # end = time.time()
        # print(f"[⏱️] VAD took {end - start:.2f} seconds.")

        # Calculate duration and apply smart chunking
        duration_minutes = waveform.size(1) / sample_rate / 60
        
        print(f"🧩 Smart chunking based on duration...")
        start = time.time()
        chunks = smart_chunk_audio(waveform, sample_rate, duration_minutes)
        end = time.time()
        print(f"[⏱️] Smart chunking took {end - start:.2f} seconds. Total chunks: {len(chunks)}")

        return {
            "success": True,
            "chunks": chunks,
            "audio_path": audio_path,
            "duration_minutes": duration_minutes,
            "total_chunks": len(chunks)
        }

    except Exception as e:
        print(f"❌ Error in audio preparation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "chunks": [],
            "audio_path": None
        }