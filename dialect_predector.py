import os
import sys
import warnings
import time
from collections import Counter

import torch
from speechbrain.inference.classifiers import EncoderClassifier

from audio_extractor import prepare_audio

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def predict_accent_from_chunks(chunks, classifier, early_stopping_threshold=3, confidence_threshold=0.6):
    """Predict accents for chunks iteratively with early stopping based on confident predictions only."""
    print(f"\n📦 Running prediction for up to {len(chunks)} chunks with early stopping (threshold={early_stopping_threshold}, confidence>{confidence_threshold*100}%)...")
    iterative_start_time = time.time()

    results = []
    consecutive_dialect_count = 0
    last_dialect = None
    
    processed_chunks_count_in_func = 0 # Renamed to avoid clash if this func is nested

    for i, chunk_tensor in enumerate(chunks):
        processed_chunks_count_in_func += 1
        
        current_chunk_for_batch = chunk_tensor
        if current_chunk_for_batch.ndim == 1:
            current_chunk_for_batch = current_chunk_for_batch.unsqueeze(0) # Shape: [1, T]
        elif not (current_chunk_for_batch.ndim == 2 and current_chunk_for_batch.shape[0] == 1):
            print(f"Warning: Chunk {i+1} has unexpected shape {current_chunk_for_batch.shape}. Required [T] or [1,T]. Skipping.")
            continue

        # Perform prediction for the single chunk
        out_prob, score, index, text_lab = classifier.classify_batch(current_chunk_for_batch)

        accent = text_lab[0] # Batch of 1
        confidence = score[0].item()
        class_idx = index[0].item()
        
        # Determine if prediction is confident enough
        is_confident = confidence > confidence_threshold
        confidence_indicator = "✓" if is_confident else "✗"
        
        print(f"Chunk {i+1}/{len(chunks)}: {accent} | Confidence: {confidence:.2f} {confidence_indicator}")
        
        current_result = {
            "chunk_index_original": i + 1,
            "accent": accent,
            "confidence": confidence,
            "class_index": class_idx,
            "is_confident": is_confident
        }
        results.append(current_result)

        # Only consider confident predictions for early stopping
        if is_confident:
            if accent == last_dialect:
                consecutive_dialect_count += 1
            else:
                last_dialect = accent
                consecutive_dialect_count = 1

            if consecutive_dialect_count >= early_stopping_threshold:
                print(f"\n⚠️ Early stopping triggered after processing chunk {i+1}: "
                      f"{early_stopping_threshold} consecutive confident chunks predicted '{last_dialect}'.")
                break
        else:
            # Reset consecutive count if prediction is not confident
            consecutive_dialect_count = 0
            last_dialect = None
            
    iterative_end_time = time.time()
    num_actually_processed = len(results)
    confident_predictions = sum(1 for r in results if r["is_confident"])
    print(f"[⏱️] Prediction for {num_actually_processed} out of {len(chunks)} available chunks took {iterative_end_time - iterative_start_time:.2f} seconds.")
    print(f"[📊] {confident_predictions}/{num_actually_processed} predictions were confident (>{confidence_threshold*100}%).")
    
    # Add sequential "chunk" number for processed chunks
    for idx, res_item in enumerate(results):
        res_item["chunk"] = idx + 1

    return results

def get_final_verdict(chunk_results, confidence_threshold=0.6):
    """Determine final accent based on confident predictions only (confidence > threshold)."""
    if not chunk_results:
        return None, 0.0, {}, {}

    # Filter for confident predictions only
    confident_results = [r for r in chunk_results if r["confidence"] > confidence_threshold]
    
    if not confident_results:
        print(f"\n⚠️ No confident predictions found (confidence > {confidence_threshold*100}%). Using all predictions as fallback.")
        confident_results = chunk_results

    accent_confidence_sum = {}
    accent_counts = Counter()
    all_accent_counts = Counter()  # Track all predictions for reporting

    # Calculate stats for confident predictions
    for result in confident_results:
        accent = result["accent"]
        confidence = result["confidence"]
        accent_counts[accent] += 1
        accent_confidence_sum[accent] = accent_confidence_sum.get(accent, 0.0) + confidence

    # Calculate stats for all predictions (for reporting)
    for result in chunk_results:
        all_accent_counts[result["accent"]] += 1

    final_accent = max(accent_confidence_sum, key=accent_confidence_sum.get)
    final_confidence = accent_confidence_sum[final_accent] / accent_counts[final_accent]

    print(f"\n📊 Accent Analysis (based on {len(confident_results)} confident predictions out of {len(chunk_results)} total):")
    print(f"    Confident predictions (confidence > {confidence_threshold*100}%):")
    for accent in accent_counts:
        count = accent_counts[accent]
        total_conf = accent_confidence_sum[accent]
        avg_conf = total_conf / count
        print(f"      {accent}: {count} chunks, total confidence: {total_conf:.2f}, avg confidence: {avg_conf:.2f}")
    
    print(f"    All predictions (including low confidence):")
    for accent in all_accent_counts:
        count = all_accent_counts[accent]
        print(f"      {accent}: {count} chunks")

    return final_accent, final_confidence, accent_counts, all_accent_counts


def analyze_video_accent(video_url, confidence_threshold=0.6):
    """Main function to analyze video accent with confidence threshold"""
    total_start = time.time()
    
    try:
        audio_result = prepare_audio(video_url)
        
        if not audio_result["success"]:
            return {
                "success": False, "error": audio_result["error"], "predicted_accent": "Error",
                "confidence_score": 0.0, "confidence_percentage": "0.0%", "video_url": video_url,
                "processing_time": time.time() - total_start
            }
        
        chunks = audio_result["chunks"]
        available_chunks_count = len(chunks)
        
        if not chunks:
            return {
                "success": False, "error": "No valid audio chunks found", "predicted_accent": "Error",
                "confidence_score": 0.0, "confidence_percentage": "0.0%", "video_url": video_url,
                "available_chunks_count": 0, "processed_chunks_count": 0,
                "processing_time": time.time() - total_start
            }

        print(f"🧠 Loading accent classification model...")
        load_model_start = time.time()
        classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa")
        load_model_end = time.time()
        print(f"[⏱️] Model loading took {load_model_end - load_model_start:.2f} seconds.")

        chunk_results = predict_accent_from_chunks(chunks, classifier, confidence_threshold=confidence_threshold)
        processed_chunks_count = len(chunk_results)
        
        final_accent, final_confidence, confident_accent_counts, all_accent_counts = get_final_verdict(chunk_results, confidence_threshold)
        
        if final_accent is None:
             return {
                "success": False, "error": "Could not determine accent (no chunks processed or no consensus)",
                "predicted_accent": "Unknown", "confidence_score": 0.0, "confidence_percentage": "0.0%",
                "video_url": video_url, "available_chunks_count": available_chunks_count,
                "processed_chunks_count": processed_chunks_count, "chunk_results": chunk_results,
                "processing_time": time.time() - total_start
            }

        # Calculate statistics
        confident_chunks = [r for r in chunk_results if r["confidence"] > confidence_threshold]
        confident_chunks_count = len(confident_chunks)
        
        avg_conf_processed_chunks = 0.0
        if processed_chunks_count > 0:
            avg_conf_processed_chunks = sum(r["confidence"] for r in chunk_results) / processed_chunks_count
            
        avg_conf_confident_chunks = 0.0
        if confident_chunks_count > 0:
            avg_conf_confident_chunks = sum(r["confidence"] for r in confident_chunks) / confident_chunks_count
        
        total_end = time.time()
        total_processing_time = total_end - total_start
        print(f"\n[⏱️] 🔁 Total pipeline time: {total_processing_time:.2f} seconds.")
        
        winning_chunks_for_final_accent = confident_accent_counts.get(final_accent, 0)
        early_stopped = processed_chunks_count < available_chunks_count

        print(f"\n✅ Final Verdict: {final_accent}")
        print(f"📈 Final Confidence (for '{final_accent}'): {final_confidence:.2f}")
        print(f"🎯 Based on {winning_chunks_for_final_accent} confident occurrences out of {confident_chunks_count} confident chunks.")
        print(f"   ({confident_chunks_count}/{processed_chunks_count} chunks were confident, threshold: {confidence_threshold*100}%)")
        if early_stopped:
            print(f"   (Early stopping occurred. {available_chunks_count} chunks were available in total).")
        print(f"📊 Average Confidence Across All Processed Chunks: {avg_conf_processed_chunks:.2f}")
        print(f"📊 Average Confidence Across Confident Chunks: {avg_conf_confident_chunks:.2f}")

        return {
            "success": True,
            "predicted_accent": final_accent,
            "confidence_score": final_confidence,
            "confidence_percentage": f"{final_confidence * 100:.1f}%",
            "confidence_threshold": confidence_threshold,
            "average_confidence_processed_chunks": avg_conf_processed_chunks,
            "average_confidence_confident_chunks": avg_conf_confident_chunks,
            "confident_accent_counts": dict(confident_accent_counts),
            "all_accent_counts": dict(all_accent_counts),
            "processed_chunks_count": processed_chunks_count,
            "confident_chunks_count": confident_chunks_count,
            "available_chunks_count": available_chunks_count,
            "winning_chunks_for_final_accent": winning_chunks_for_final_accent,
            "audio_file": audio_result.get("audio_path"),
            "video_url": video_url,
            "duration_minutes": audio_result.get("duration_minutes"),
            "chunk_results": chunk_results,
            "processing_time": total_processing_time,
            "early_stopped": early_stopped
        }

    except Exception as e:
        total_end = time.time()
        processing_time_before_error = total_end - total_start
        print(f"❌ Error: {str(e)}")
        print(f"[⏱️] Total time before error: {processing_time_before_error:.2f} seconds.")
        
        return {
            "success": False, "error": str(e), "predicted_accent": "Error",
            "confidence_score": 0.0, "confidence_percentage": "0.0%", "video_url": video_url,
            "processing_time": processing_time_before_error
        }

if __name__ == "__main__":
    video_url = "https://www.youtube.com/shorts/sWUvKMC2450"
    result = analyze_video_accent(video_url, confidence_threshold=0.6)

    if result["success"]:
        print(f"\n🎤 Final Predicted Accent: {result['predicted_accent']}")
        print(f"🔢 Confidence Score: {result['confidence_score']:.4f}")
        print(f"📊 Confidence Percentage: {result['confidence_percentage']}")
        print(f"🎯 Based on {result['confident_chunks_count']} confident chunks out of {result['processed_chunks_count']} total")
    else:
        print(f"❌ Error: {result['error']}")
        print(f"⏱️ Processing Time: {result.get('processing_time', 0):.2f} seconds")