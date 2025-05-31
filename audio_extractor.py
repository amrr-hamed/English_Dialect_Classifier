import os
import sys
import tempfile
import warnings
import time
import shutil
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path

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

class SimpleAudioExtractor:
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v']
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    def extract_audio_from_source(self, source):
        """Extract audio from file path, direct media URL, or Loom URL"""
        start_time = time.time()
        
        # Check if source is a file path
        if self._is_file_path(source):
            print(f"📁 Processing uploaded file: {source}")
            return self._process_local_file(source, start_time)
        
        # Check if source is a direct media URL
        if self._is_direct_media_url(source):
            print(f"🔗 Processing direct media URL: {source}")
            return self._download_direct_media(source, start_time)
        
        # Check if source is a Loom URL
        if self._is_loom_url(source):
            print(f"🎥 Processing Loom URL: {source}")
            return self._extract_from_loom(source, start_time)
        
        raise Exception("Unsupported URL format. Please use Loom URLs or direct media links.")

    def _is_file_path(self, source):
        """Check if source is a local file path"""
        try:
            path = Path(source)
            return path.exists() and path.is_file()
        except:
            return False

    def _is_direct_media_url(self, url):
        """Check if URL points directly to a media file"""
        try:
            parsed = urlparse(url.lower())
            path = unquote(parsed.path)
            return any(path.endswith(ext) for ext in self.supported_video_formats + self.supported_audio_formats)
        except:
            return False

    def _is_loom_url(self, url):
        """Check if URL is a Loom video"""
        return 'loom.com' in url.lower()

    def _process_local_file(self, file_path, start_time):
        """Process a local file (uploaded file)"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # If it's already an audio file, convert to WAV if needed
            if file_ext in self.supported_audio_formats:
                if file_ext == '.wav':
                    end_time = time.time()
                    print(f"[⏱️] Audio file processing took {end_time - start_time:.2f} seconds.")
                    return file_path
                else:
                    return self._convert_to_wav(file_path, start_time)
            
            # If it's a video file, extract audio
            elif file_ext in self.supported_video_formats:
                return self._extract_audio_from_video_file(file_path, start_time)
            
            else:
                raise Exception(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            raise Exception(f"Failed to process local file: {str(e)}")

    def _download_direct_media(self, url, start_time):
        """Download direct media URL"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Determine file extension from URL or content type
            parsed_url = urlparse(url)
            url_ext = Path(parsed_url.path).suffix.lower()
            
            if url_ext in self.supported_video_formats + self.supported_audio_formats:
                ext = url_ext
            else:
                # Try to get from content type
                content_type = response.headers.get('content-type', '').lower()
                if 'video' in content_type:
                    ext = '.mp4'
                elif 'audio' in content_type:
                    ext = '.mp3'
                else:
                    ext = '.mp4'  # default
            
            downloaded_file = os.path.join(temp_dir, f'downloaded{ext}')
            
            with open(downloaded_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✅ Downloaded {os.path.getsize(downloaded_file) / 1024 / 1024:.1f}MB")
            
            # Process the downloaded file
            if ext in self.supported_audio_formats:
                if ext == '.wav':
                    end_time = time.time()
                    print(f"[⏱️] Direct audio download took {end_time - start_time:.2f} seconds.")
                    return downloaded_file
                else:
                    return self._convert_to_wav(downloaded_file, start_time)
            else:
                return self._extract_audio_from_video_file(downloaded_file, start_time)
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Failed to download direct media: {str(e)}")

    def _extract_from_loom(self, url, start_time):
        """Extract audio from Loom URL using yt-dlp"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, 'loom_audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'http_headers': {
                    'User-Agent': self.user_agent,
                },
            }

            with suppress_stdout_stderr():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            # Find the extracted audio file
            for file in os.listdir(temp_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(temp_dir, file)
                    end_time = time.time()
                    print(f"[⏱️] Loom audio extraction took {end_time - start_time:.2f} seconds.")
                    return audio_path

            raise Exception("Audio file not found after Loom extraction")
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Failed to extract from Loom: {str(e)}")

    def _extract_audio_from_video_file(self, video_file, start_time):
        """Extract audio from video file using FFmpeg or torchaudio"""
        temp_dir = tempfile.mkdtemp()
        output_audio = os.path.join(temp_dir, 'extracted_audio.wav')
        
        try:
            # Try FFmpeg first
            import subprocess
            
            cmd = [
                'ffmpeg', '-i', video_file,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',  # uncompressed WAV
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # mono
                '-y',  # overwrite output file
                output_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_audio):
                end_time = time.time()
                print(f"[⏱️] Audio extraction from video took {end_time - start_time:.2f} seconds.")
                return output_audio
            else:
                raise Exception("FFmpeg failed, trying torchaudio...")
                
        except (FileNotFoundError, Exception):
            # Fallback to torchaudio
            return self._convert_to_wav(video_file, start_time)

    def _convert_to_wav(self, audio_file, start_time):
        """Convert audio file to WAV format using torchaudio"""
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            
            # Save as WAV
            temp_dir = tempfile.mkdtemp()
            output_wav = os.path.join(temp_dir, 'converted_audio.wav')
            torchaudio.save(output_wav, waveform, 16000)
            
            end_time = time.time()
            print(f"[⏱️] Audio conversion took {end_time - start_time:.2f} seconds.")
            return output_wav
            
        except Exception as e:
            raise Exception(f"Failed to convert audio to WAV: {str(e)}")

def chunk_audio_1min(waveform, sample_rate):
    """Create 1-minute chunks from audio"""
    chunk_length_sec = 60  # 1 minute chunks
    chunk_samples = chunk_length_sec * sample_rate
    total_samples = waveform.size(1)
    chunks = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        # Only include chunks that are at least 10 seconds long
        if chunk.size(1) > sample_rate * 10:
            chunks.append(chunk)
    
    print(f"📦 Created {len(chunks)} 1-minute chunks")
    return chunks

def prepare_audio(video_source):
    """Main function to extract and prepare 1-minute audio chunks"""
    try:
        print(f"🎵 Extracting audio from source...")
        extractor = SimpleAudioExtractor()
        audio_path = extractor.extract_audio_from_source(video_source)
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

        # Calculate duration and create 1-minute chunks
        duration_minutes = waveform.size(1) / sample_rate / 60
        
        print(f"🧩 Creating 1-minute chunks...")
        start = time.time()
        chunks = chunk_audio_1min(waveform, sample_rate)
        end = time.time()
        print(f"[⏱️] Chunking took {end - start:.2f} seconds. Total chunks: {len(chunks)}")

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