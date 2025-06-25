# hanabi.py

import librosa
import numpy as np
import json
import os
import datetime
import concurrent.futures
import argparse
import glob

# --- Configuration (Adjust these!) ---
RMS_THRESHOLD_FACTOR = 0.3  # Lowered from 0.5 to generate more cues
SPECTRAL_CENTROID_THRESHOLD_FACTOR = 1.0
MIN_TIME_BETWEEN_CUES = 0.2  # Reduced from 0.3 to allow more cues
FIREWORK_TYPES = {
    "low_mid": "burst",
    "high": "sparkle",
    "kick": "boom"
}
HOP_LENGTH_FEATURES = 512

# --- Timestamped Logger ---
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

# --- Audio File Discovery ---
def find_audio_file(custom_path=None):
    """Find audio file - either custom path or first one in ../audio directory"""
    if custom_path:
        custom_path = os.path.expanduser(custom_path)
        if os.path.exists(custom_path):
            log_message(f"Using specified audio file: {custom_path}")
            return custom_path
        else:
            log_message(f"Specified file not found: {custom_path}")
            return None
    
    # Look for audio files in ../audio directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "..", "audio")
    
    if not os.path.exists(audio_dir):
        log_message(f"Audio directory not found: {audio_dir}")
        return None
    
    # Common audio file extensions
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.aac']
    
    for ext in audio_extensions:
        pattern = os.path.join(audio_dir, ext)
        files = glob.glob(pattern)
        if files:
            audio_file = files[0]  # Use the first file found
            log_message(f"Found audio file: {audio_file}")
            return audio_file
    
    log_message(f"No audio files found in {audio_dir}")
    return None

# --- Output Directory Setup ---
def ensure_output_directory():
    """Ensure ../output directory exists"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created output directory: {output_dir}")
    
    return output_dir

# --- Feature Extraction Helpers (for threading) ---
def _extract_beats(y, sr):
    log_message("Starting beat tracking...")
    tempo_from_librosa, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Prepare tempo for logging
    tempo_for_logging = 0.0
    if isinstance(tempo_from_librosa, np.ndarray):
        if tempo_from_librosa.size == 1:
            tempo_for_logging = tempo_from_librosa.item()
        elif tempo_from_librosa.size > 1:
            log_message(f"Warning: Beat tracking returned an array of tempos: {tempo_from_librosa}. Logging the first one.")
            try:
                tempo_for_logging = float(tempo_from_librosa.flat[0])
            except (TypeError, ValueError):
                 log_message(f"Warning: Could not convert first element of tempo array to float. Logging 0.0 BPM.")
        else:
            log_message("Warning: Beat tracking returned an empty tempo array. Logging 0.0 BPM.")
    elif tempo_from_librosa is not None:
        try:
            tempo_for_logging = float(tempo_from_librosa)
        except (TypeError, ValueError):
            log_message(f"Warning: Could not convert tempo value to float. Logging 0.0 BPM.")
    else:
        log_message("Warning: Beat tracking returned None for tempo. Logging 0.0 BPM.")
    
    log_message(f"Beat tracking complete. Tempo: {tempo_for_logging:.2f} BPM, Beats: {len(beat_times)}")
    return tempo_from_librosa, beat_times

def _extract_rms(y, sr, hop_length):
    log_message("Starting RMS extraction...")
    rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    log_message(f"RMS extraction complete. Frames: {len(rms_frames)}")
    return rms_frames

def _extract_spectral_centroid(y, sr, hop_length):
    log_message("Starting spectral centroid extraction...")
    spectral_centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    log_message(f"Spectral centroid extraction complete. Frames: {len(spectral_centroid_frames)}")
    return spectral_centroid_frames

def _extract_onsets(y, sr):
    log_message("Starting onset detection...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    log_message(f"Onset detection complete. Onsets: {len(onset_times)}")
    return onset_times

# --- Core Functions ---
def analyze_audio(audio_path):
    log_message(f"Loading audio: {audio_path}...")
    try:
        y, sr = librosa.load(audio_path)
        log_message(f"Audio loaded successfully. Duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")
    except Exception as e:
        log_message(f"Error loading audio file: {e}")
        log_message("Please ensure you have ffmpeg installed and in your PATH if using MP3, or try a WAV file.")
        return None

    log_message("Starting parallel feature extraction...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        future_beats = executor.submit(_extract_beats, y, sr)
        future_rms = executor.submit(_extract_rms, y, sr, HOP_LENGTH_FEATURES)
        future_sc = executor.submit(_extract_spectral_centroid, y, sr, HOP_LENGTH_FEATURES)
        future_onsets = executor.submit(_extract_onsets, y, sr)

        results = {}
        try:
            log_message("Waiting for feature extraction tasks to complete...")
            results['tempo'], results['beat_times'] = future_beats.result()
            results['rms_frames'] = future_rms.result()
            results['spectral_centroid_frames'] = future_sc.result()
            results['onset_times'] = future_onsets.result()
            log_message("All feature extraction tasks completed.")
        except Exception as e:
            log_message(f"Error during threaded feature extraction: {e}")
            if future_beats.done() and future_beats.exception():
                 log_message(f"Beat extraction exception: {future_beats.exception()}")
            if future_rms.done() and future_rms.exception():
                 log_message(f"RMS extraction exception: {future_rms.exception()}")
            if future_sc.done() and future_sc.exception():
                 log_message(f"Spectral centroid extraction exception: {future_sc.exception()}")
            if future_onsets.done() and future_onsets.exception():
                 log_message(f"Onset extraction exception: {future_onsets.exception()}")
            return None

    return y, sr, results['tempo'], results['beat_times'], results['rms_frames'], results['spectral_centroid_frames'], results['onset_times']

def generate_cues(sr, beat_times, rms_frames, spectral_centroid_frames, onset_times, hop_length_features):
    cues = []
    last_cue_time = -1000.0

    if not rms_frames.size or not spectral_centroid_frames.size:
        log_message("Warning: RMS or Spectral Centroid data is empty. Cannot generate cues effectively.")
        return cues

    try:
        median_rms = np.median(rms_frames)
        # Use a more generous threshold calculation
        rms_trigger_threshold = median_rms + (np.std(rms_frames) * RMS_THRESHOLD_FACTOR)
        log_message(f"RMS trigger threshold: {rms_trigger_threshold:.6f} (median RMS: {median_rms:.6f}, std: {np.std(rms_frames):.6f})")

        median_spectral_centroid = np.median(spectral_centroid_frames)
        spectral_centroid_split_threshold = median_spectral_centroid * SPECTRAL_CENTROID_THRESHOLD_FACTOR
        log_message(f"Spectral centroid split threshold: {spectral_centroid_split_threshold:.2f} (median: {median_spectral_centroid:.2f})")
    except Exception as e:
        log_message(f"Error calculating medians or thresholds: {e}. RMS len: {len(rms_frames)}, SC len: {len(spectral_centroid_frames)}")
        return cues

    log_message("Generating cues from beats...")
    beats_processed = 0
    beats_above_threshold = 0
    
    for beat_time in beat_times:
        beats_processed += 1
        
        if beat_time - last_cue_time < MIN_TIME_BETWEEN_CUES:
            continue

        feature_frame_index = librosa.time_to_frames(beat_time, sr=sr, hop_length=hop_length_features)

        if feature_frame_index >= len(rms_frames) or feature_frame_index >= len(spectral_centroid_frames):
            continue
            
        current_rms = rms_frames[feature_frame_index]
        current_centroid = spectral_centroid_frames[feature_frame_index]
        
        # Debug logging for first few beats
        if beats_processed <= 5:
            log_message(f"Beat {beats_processed}: time={beat_time:.2f}s, RMS={current_rms:.6f} (threshold={rms_trigger_threshold:.6f}), centroid={current_centroid:.2f}")
        
        firework_type = None

        if current_rms > rms_trigger_threshold:
            beats_above_threshold += 1
            if current_centroid > spectral_centroid_split_threshold:
                firework_type = FIREWORK_TYPES["high"]
            else:
                firework_type = FIREWORK_TYPES["low_mid"]
            
        if firework_type:
            cues.append({
                "timestamp": round(beat_time, 2),
                "firework": firework_type,
            })
            last_cue_time = beat_time

    # Also generate cues from strong onsets
    log_message("Generating additional cues from onsets...")
    for onset_time in onset_times:
        if onset_time - last_cue_time < MIN_TIME_BETWEEN_CUES:
            continue
            
        feature_frame_index = librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length_features)
        
        if feature_frame_index >= len(rms_frames):
            continue
            
        current_rms = rms_frames[feature_frame_index]
        
        # Use a lower threshold for onsets since they're already detected events
        if current_rms > rms_trigger_threshold * 0.7:
            cues.append({
                "timestamp": round(onset_time, 2),
                "firework": FIREWORK_TYPES["kick"],
            })
            last_cue_time = onset_time

    cues.sort(key=lambda x: x["timestamp"])
    log_message(f"Processed {beats_processed} beats, {beats_above_threshold} above RMS threshold")
    log_message(f"Generated {len(cues)} total cues (beats + onsets).")
    return cues

def main():
    parser = argparse.ArgumentParser(description='HanabiSync - Generate firework cues from audio')
    parser.add_argument('-f', '--file', help='Path to audio file (if not specified, uses first file in ../audio)')
    args = parser.parse_args()
    
    # Find audio file
    audio_file_path = find_audio_file(args.file)
    if not audio_file_path:
        log_message("No audio file found. Please provide a file with -f or place audio files in ../audio directory.")
        return

    # Ensure output directory exists
    output_dir = ensure_output_directory()

    # Analyze audio
    analysis_results = analyze_audio(audio_file_path)
    
    if analysis_results is None:
        log_message("Audio analysis failed. Exiting.")
        return

    _y, sr, _tempo, beat_times, rms_frames, spectral_centroid_frames, onset_times = analysis_results
    
    # Generate cues
    firework_cues = generate_cues(sr, beat_times, rms_frames, spectral_centroid_frames, onset_times, HOP_LENGTH_FEATURES)

    # Prepare output filename
    output_filename_base = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_filename = os.path.join(output_dir, f"{output_filename_base}_cues.json")
    
    log_message(f"\n--- Generated Cues ---")
    if firework_cues:
        for line in json.dumps(firework_cues, indent=2).splitlines():
            log_message(line)
    else:
        log_message("[] (No cues generated)")

    # Save cues to output directory
    try:
        with open(output_filename, 'w') as f:
            json.dump(firework_cues, f, indent=2)
        log_message(f"\nCue map saved to {output_filename}")
    except IOError as e:
        log_message(f"Error saving cue map to {output_filename}: {e}")
    
    log_message("🎆🔊 HanabiSync PoC complete! Remember to tune thresholds in hanabi.py for better results. 🔊🎆")

if __name__ == "__main__":
    main()