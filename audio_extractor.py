import os
import sys
import tempfile
import warnings
import time
import shutil
import random
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

class RobustAudioExtractor:
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v', '.3gp']
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    def extract_audio_from_source(self, source):
        """
        Extract audio from various sources:
        - File path (uploaded file)
        - Direct media URL (MP4, etc.)
        - Loom URL
        - Other video hosting URLs
        """
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
        
        # Try with yt-dlp for other platforms (with robust error handling)
        print(f"🌐 Processing URL with yt-dlp: {source}")
        return self._extract_with_ytdlp_robust(source, start_time)

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
            
            # If it's already an audio file, just return it
            if file_ext in self.supported_audio_formats:
                if file_ext == '.wav':
                    end_time = time.time()
                    print(f"[⏱️] Audio file processing took {end_time - start_time:.2f} seconds.")
                    return file_path
                else:
                    # Convert to WAV
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
                'User-Agent': random.choice(self.user_agents),
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '').lower()
            if 'video' in content_type:
                if 'mp4' in content_type:
                    ext = '.mp4'
                elif 'webm' in content_type:
                    ext = '.webm'
                else:
                    ext = '.mp4'  # default
            elif 'audio' in content_type:
                if 'mpeg' in content_type or 'mp3' in content_type:
                    ext = '.mp3'
                elif 'wav' in content_type:
                    ext = '.wav'
                else:
                    ext = '.mp3'  # default
            else:
                # Try to get from URL
                parsed_url = urlparse(url)
                url_ext = Path(parsed_url.path).suffix.lower()
                ext = url_ext if url_ext in self.supported_video_formats + self.supported_audio_formats else '.mp4'
            
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
        """Extract audio from Loom with multiple strategies"""
        strategies = [
            self._loom_strategy_basic,
            self._loom_strategy_embed,
            self._loom_strategy_api,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"Trying Loom strategy {i+1}...")
                result = strategy(url, start_time)
                if result:
                    return result
                time.sleep(1)  # Brief delay between strategies
            except Exception as e:
                print(f"Loom strategy {i+1} failed: {str(e)}")
                continue
        
        raise Exception("Failed to extract audio from Loom URL with all strategies")

    def _loom_strategy_basic(self, url, start_time):
        """Basic Loom extraction using yt-dlp"""
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio[abr<=128]/best[height<=720]',
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
                'User-Agent': random.choice(self.user_agents)
            }
        }

        with suppress_stdout_stderr():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        return self._find_audio_file(temp_dir, start_time)

    def _loom_strategy_embed(self, url, start_time):
        """Try Loom embed URL format"""
        # Extract video ID from Loom URL
        import re
        loom_id_match = re.search(r'loom\.com/share/([a-zA-Z0-9]+)', url)
        if loom_id_match:
            video_id = loom_id_match.group(1)
            embed_url = f"https://www.loom.com/embed/{video_id}"
            return self._loom_strategy_basic(embed_url, start_time)
        return None

    def _loom_strategy_api(self, url, start_time):
        """Try to get direct video URL from Loom"""
        # This is a placeholder for a more sophisticated approach
        # You might need to inspect Loom's network requests to find direct video URLs
        return None

    def _extract_with_ytdlp_robust(self, url, start_time):
        """Robust yt-dlp extraction with multiple strategies"""
        strategies = [
            self._ytdlp_strategy_basic,
            self._ytdlp_strategy_with_headers,
            self._ytdlp_strategy_low_quality,
            self._ytdlp_strategy_audio_only,
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"Trying yt-dlp strategy {i+1}...")
                result = strategy(url, start_time)
                if result:
                    return result
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"yt-dlp strategy {i+1} failed: {str(e)}")
                continue
        
        raise Exception("Failed to extract audio with all yt-dlp strategies")

    def _ytdlp_strategy_basic(self, url, start_time):
        """Basic yt-dlp strategy"""
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio[abr<=64]/worst',
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
                ydl.download([url])

        return self._find_audio_file(temp_dir, start_time)

    def _ytdlp_strategy_with_headers(self, url, start_time):
        """yt-dlp with browser-like headers"""
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio[abr<=64]/worst',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
            'sleep_interval': 1,
            'max_sleep_interval': 3,
        }

        with suppress_stdout_stderr():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        return self._find_audio_file(temp_dir, start_time)

    def _ytdlp_strategy_low_quality(self, url, start_time):
        """yt-dlp with lowest quality to avoid detection"""
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'worstaudio/worst',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '128',
            }],
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'sleep_interval': 2,
            'max_sleep_interval': 5,
        }

        with suppress_stdout_stderr():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        return self._find_audio_file(temp_dir, start_time)

    def _ytdlp_strategy_audio_only(self, url, start_time):
        """yt-dlp targeting audio-only streams"""
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'prefer_ffmpeg': True,
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
        }

        with suppress_stdout_stderr():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        return self._find_audio_file(temp_dir, start_time)

    def _extract_audio_from_video_file(self, video_file, start_time):
        """Extract audio from video file using FFmpeg"""
        temp_dir = tempfile.mkdtemp()
        output_audio = os.path.join(temp_dir, 'extracted_audio.wav')
        
        try:
            import subprocess
            
            # Use FFmpeg to extract audio
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
                raise Exception(f"FFmpeg failed: {result.stderr}")
                
        except FileNotFoundError:
            # Fallback to torchaudio if FFmpeg not available
            return self._convert_to_wav(video_file, start_time)
        except Exception as e:
            raise Exception(f"Failed to extract audio from video: {str(e)}")

    def _convert_to_wav(self, audio_file, start_time):
        """Convert audio file to WAV format"""
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

    def _find_audio_file(self, directory, start_time):
        """Find the extracted audio file"""
        audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.aac']
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(directory, file)
                
                # Convert to WAV if not already
                if not file.lower().endswith('.wav'):
                    return self._convert_to_wav(audio_path, start_time)
                
                end_time = time.time()
                print(f"[⏱️] Audio extraction took {end_time - start_time:.2f} seconds.")
                return audio_path
        
        raise Exception("No audio file found after extraction")

# Update the main function to use the new extractor
def extract_audio_from_video_url(video_source):
    """
    Main function that handles all types of video sources:
    - File paths (uploaded files)
    - Direct media URLs
    - Loom URLs
    - Other video platform URLs
    """
    extractor = RobustAudioExtractor()
    return extractor.extract_audio_from_source(video_source)

# Keep the existing chunking functions unchanged
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

def prepare_audio(video_source):
    """Main function to extract and prepare audio chunks"""
    try:
        print(f"🎵 Extracting audio from source...")
        audio_path = extract_audio_from_video_url(video_source)
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