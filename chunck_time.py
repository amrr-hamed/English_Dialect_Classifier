import os
import sys
import warnings
import time
import statistics
from collections import Counter

import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

from audio_extractor import extract_audio_from_video_url

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def create_chunks_by_size(waveform, sample_rate, chunk_length_sec):
    """Create chunks of specific size"""
    chunk_samples = chunk_length_sec * sample_rate
    total_samples = waveform.size(1)
    chunks = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.size(1) > sample_rate * 2:  # minimum 2 seconds
            chunks.append(chunk)
    return chunks

def predict_chunks_timing(chunks, classifier):
    """Time the prediction process for chunks"""
    if not chunks:
        return [], 0.0
    
    start_time = time.time()
    
    # Pad to same length
    max_len = max(chunk.size(1) for chunk in chunks)
    padded_chunks = [torch.nn.functional.pad(chunk, (0, max_len - chunk.size(1))) for chunk in chunks]
    batch = torch.cat(padded_chunks, dim=0).unsqueeze(1)
    batch = batch.squeeze(1)

    out_prob, score, index, text_lab = classifier.classify_batch(batch)
    
    end_time = time.time()
    prediction_time = end_time - start_time
    
    results = []
    for i in range(len(chunks)):
        results.append({
            "accent": text_lab[i],
            "confidence": score[i].item(),
        })
    
    return results, prediction_time

def analyze_chunk_size_performance(video_url, chunk_sizes=[10, 15, 20, 30, 60]):
    """Analyze performance for different chunk sizes"""
    print("🔍 Starting Chunk Size Performance Analysis")
    print("=" * 60)
    
    # Extract and prepare audio once
    print("🎵 Extracting and preparing audio...")
    audio_start = time.time()
    
    audio_path = extract_audio_from_video_url(video_url)
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # # Apply VAD
    # waveform = simple_vad(waveform, sample_rate)
    
    audio_end = time.time()
    audio_prep_time = audio_end - audio_start
    
    duration_minutes = waveform.size(1) / sample_rate / 60
    print(f"✅ Audio prepared in {audio_prep_time:.2f}s | Duration: {duration_minutes:.1f} minutes")
    
    # Load model once
    print("🧠 Loading model...")
    model_start = time.time()
    classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa")
    model_end = time.time()
    model_load_time = model_end - model_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("📊 CHUNK SIZE ANALYSIS RESULTS")
    print("=" * 60)
    
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"\n🧩 Testing {chunk_size}-second chunks...")
        
        # Create chunks
        chunk_start = time.time()
        chunks = create_chunks_by_size(waveform, sample_rate, chunk_size)
        chunk_end = time.time()
        chunking_time = chunk_end - chunk_start
        
        if not chunks:
            print(f"❌ No valid chunks created for {chunk_size}s size")
            continue
        
        # Predict
        predictions, prediction_time = predict_chunks_timing(chunks, classifier)
        
        # Calculate statistics
        confidences = [p["confidence"] for p in predictions]
        accents = [p["accent"] for p in predictions]
        
        avg_confidence = statistics.mean(confidences) if confidences else 0
        max_confidence = max(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # Most common accent
        accent_counts = Counter(accents)
        most_common_accent = accent_counts.most_common(1)[0] if accent_counts else ("Unknown", 0)
        
        # Calculate processing rates
        total_processing_time = chunking_time + prediction_time
        chunks_per_second = len(chunks) / total_processing_time if total_processing_time > 0 else 0
        seconds_per_chunk = total_processing_time / len(chunks) if len(chunks) > 0 else 0
        
        result = {
            "chunk_size": chunk_size,
            "num_chunks": len(chunks),
            "chunking_time": chunking_time,
            "prediction_time": prediction_time,
            "total_time": total_processing_time,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "min_confidence": min_confidence,
            "std_confidence": std_confidence,
            "most_common_accent": most_common_accent[0],
            "accent_occurrence": most_common_accent[1],
            "chunks_per_second": chunks_per_second,
            "seconds_per_chunk": seconds_per_chunk,
            "confidence_consistency": 1 - (std_confidence / avg_confidence) if avg_confidence > 0 else 0
        }
        
        results.append(result)
        
        # Print results for this chunk size
        print(f"  📦 Chunks created: {len(chunks)}")
        print(f"  ⏱️  Chunking time: {chunking_time:.3f}s")
        print(f"  🧠 Prediction time: {prediction_time:.3f}s")
        print(f"  🔄 Total processing: {total_processing_time:.3f}s")
        print(f"  ⚡ Processing rate: {chunks_per_second:.1f} chunks/sec")
        print(f"  📈 Avg confidence: {avg_confidence:.3f}")
        print(f"  🎯 Most common: {most_common_accent[0]} ({most_common_accent[1]} times)")
        print(f"  📊 Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")
    
    # Print summary comparison
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"{'Size':<6} {'Chunks':<8} {'Total Time':<12} {'Rate':<12} {'Avg Conf':<10} {'Consistency':<12} {'Winner'}")
        print("-" * 80)
        
        for r in results:
            consistency = f"{r['confidence_consistency']:.2f}"
            print(f"{r['chunk_size']:<6} {r['num_chunks']:<8} {r['total_time']:<12.3f} {r['chunks_per_second']:<12.1f} {r['avg_confidence']:<10.3f} {consistency:<12} {r['most_common_accent']}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("🏆 RECOMMENDATIONS")
    print("=" * 60)
    
    if results:
        # Find best for speed
        fastest = min(results, key=lambda x: x['total_time'])
        print(f"⚡ Fastest processing: {fastest['chunk_size']}s chunks ({fastest['total_time']:.2f}s total)")
        
        # Find best for accuracy (highest average confidence)
        most_accurate = max(results, key=lambda x: x['avg_confidence'])
        print(f"🎯 Highest accuracy: {most_accurate['chunk_size']}s chunks ({most_accurate['avg_confidence']:.3f} avg confidence)")
        
        # Find most consistent
        most_consistent = max(results, key=lambda x: x['confidence_consistency'])
        print(f"📊 Most consistent: {most_consistent['chunk_size']}s chunks ({most_consistent['confidence_consistency']:.3f} consistency)")
        
        # Find best balance (speed + accuracy)
        for r in results:
            r['balance_score'] = (r['chunks_per_second'] * 0.4) + (r['avg_confidence'] * 100 * 0.6)
        
        best_balance = max(results, key=lambda x: x['balance_score'])
        print(f"⚖️  Best balance: {best_balance['chunk_size']}s chunks (score: {best_balance['balance_score']:.1f})")
    
    return results

def quick_test_multiple_videos(video_urls, chunk_sizes=[10, 15, 20, 30]):
    """Quick test on multiple videos to get average performance"""
    print("🔍 MULTI-VIDEO CHUNK SIZE ANALYSIS")
    print("=" * 60)
    
    all_results = {size: [] for size in chunk_sizes}
    
    for i, video_url in enumerate(video_urls, 1):
        print(f"\n📹 Testing Video {i}/{len(video_urls)}")
        try:
            video_results = analyze_chunk_size_performance(video_url, chunk_sizes)
            for result in video_results:
                all_results[result['chunk_size']].append(result)
        except Exception as e:
            print(f"❌ Error with video {i}: {str(e)}")
            continue
    
    # Calculate averages
    print("\n" + "=" * 60)
    print("📊 AVERAGE PERFORMANCE ACROSS ALL VIDEOS")
    print("=" * 60)
    
    avg_results = []
    for chunk_size in chunk_sizes:
        if all_results[chunk_size]:
            results = all_results[chunk_size]
            avg_result = {
                'chunk_size': chunk_size,
                'avg_total_time': statistics.mean([r['total_time'] for r in results]),
                'avg_chunks_per_sec': statistics.mean([r['chunks_per_second'] for r in results]),
                'avg_confidence': statistics.mean([r['avg_confidence'] for r in results]),
                'avg_consistency': statistics.mean([r['confidence_consistency'] for r in results]),
                'sample_count': len(results)
            }
            avg_results.append(avg_result)
    
    if avg_results:
        print(f"{'Size':<6} {'Samples':<8} {'Avg Time':<10} {'Avg Rate':<10} {'Avg Conf':<10} {'Consistency'}")
        print("-" * 60)
        for r in avg_results:
            print(f"{r['chunk_size']:<6} {r['sample_count']:<8} {r['avg_total_time']:<10.2f} {r['avg_chunks_per_sec']:<10.1f} {r['avg_confidence']:<10.3f} {r['avg_consistency']:.3f}")
    
    return avg_results

if __name__ == "__main__":
    # Test with single video
    video_url = "https://www.youtube.com/watch?v=-JTq1BFBwmo&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=2"
    
    print("🚀 Starting Single Video Analysis...")
    results = analyze_chunk_size_performance(video_url)
    
    # Uncomment below to test multiple videos
    # print("\n" + "="*60)
    # print("🚀 Starting Multi-Video Analysis...")
    # video_urls = [
    #     "https://www.youtube.com/watch?v=VIDEO1",
    #     "https://www.youtube.com/watch?v=VIDEO2",
    #     # Add more video URLs here
    # ]
    # multi_results = quick_test_multiple_videos(video_urls)