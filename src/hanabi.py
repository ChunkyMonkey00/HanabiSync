import librosa
import numpy as np
import json
import os
import datetime
import concurrent.futures
import argparse
import glob

RMS_THRESHOLD_FACTOR_BEAT = 0.55
SPECTRAL_CENTROID_THRESHOLD_FACTOR = 0.8
ONSET_RMS_MULTIPLIER = 0.65
BASS_CENTROID_LOW_FACTOR = 0.75

SPECTRAL_FLUX_THRESHOLD_FACTOR = 1.75
ONSET_RMS_SHARP_ACCENT_MULTIPLIER = 0.75

MIN_TIME_BETWEEN_SAME_TYPE_CUES = 0.15
MIN_TIME_BETWEEN_ANY_PARTICLE_CUES = 0.08

FIREWORK_TYPES = {
    "mid_energy": "mid_burst",
    "high_freq": "high_sparkle",
    "strong_kick": "kick_boom",
    "deep_bass": "bass_pulse",
    "sharp_accent": "comet_tail"
}
HOP_LENGTH_FEATURES = 512
PARTICLE_FIREWORK_TYPES = [FIREWORK_TYPES["mid_energy"], FIREWORK_TYPES["high_freq"], FIREWORK_TYPES["strong_kick"], FIREWORK_TYPES["sharp_accent"]]


def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def find_audio_file(custom_path=None):
    if custom_path:
        custom_path = os.path.expanduser(custom_path)
        if os.path.exists(custom_path):
            log_message(f"Using specified audio file: {custom_path}")
            return custom_path
        log_message(f"Specified file not found: {custom_path}")
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "..", "audio")
    if not os.path.exists(audio_dir):
        log_message(f"Audio directory not found: {audio_dir}")
        return None
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.aac']
    for ext in audio_extensions:
        files = glob.glob(os.path.join(audio_dir, ext))
        if files:
            log_message(f"Found audio file: {files[0]}")
            return files[0]
    log_message(f"No audio files found in {audio_dir}")
    return None

def ensure_output_directory():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created output directory: {output_dir}")
    return output_dir

def _extract_beats(y, sr):
    log_message("Starting beat tracking...")
    tempo_from_librosa, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    tempo_for_logging_str = "N/A"
    if isinstance(tempo_from_librosa, np.ndarray):
        if tempo_from_librosa.size == 1:
            try:
                tempo_for_logging_str = f"{tempo_from_librosa.item():.2f}"
            except:
                tempo_for_logging_str = str(tempo_from_librosa.item())
        elif tempo_from_librosa.size > 1:
            try:
                mean_tempo = np.mean(tempo_from_librosa)
                tempo_for_logging_str = f"{mean_tempo:.2f} (mean of {len(tempo_from_librosa)} tempos)"
            except:
                tempo_for_logging_str = f"Array of {len(tempo_from_librosa)} tempos"
            log_message(f"Note: Beat tracking returned an array of tempos: {tempo_from_librosa}")
        else:
            tempo_for_logging_str = "Empty array"
            log_message("Warning: Beat tracking returned an empty tempo array.")
    elif tempo_from_librosa is not None:
        try:
            tempo_for_logging_str = f"{float(tempo_from_librosa):.2f}"
        except (TypeError, ValueError):
            tempo_for_logging_str = str(tempo_from_librosa)
            log_message(f"Warning: Could not convert tempo value {tempo_from_librosa} to float for formatted logging.")
    else:
        log_message("Warning: Beat tracking returned None for tempo.")
    
    log_message(f"Beat tracking complete. Tempo: {tempo_for_logging_str} BPM, Beats: {len(beat_times)}")
    return tempo_from_librosa, beat_times


def _extract_rms(y, sr, hop_length):
    log_message("Starting RMS extraction...")
    rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    log_message(f"RMS extraction complete. Frames: {len(rms_frames)}")
    return rms_frames

def _extract_spectral_centroid(y, sr, hop_length):
    log_message("Starting spectral centroid extraction...")
    sc_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    log_message(f"Spectral centroid extraction complete. Frames: {len(sc_frames)}")
    return sc_frames

def _extract_onsets(y, sr):
    log_message("Starting onset detection...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames', backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    log_message(f"Onset detection complete. Onsets: {len(onset_times)}")
    return onset_times

def _extract_spectral_flux(y, sr, hop_length):
    log_message("Starting spectral flux (onset strength) extraction...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    log_message(f"Spectral flux extraction complete. Frames: {len(onset_env)}")
    return onset_env

def analyze_audio(audio_path):
    log_message(f"Loading audio: {audio_path}...")
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        log_message(f"Error loading audio: {e}. Ensure ffmpeg is installed for MP3s.")
        return None
    log_message(f"Audio loaded: {len(y)/sr:.2f}s, SR: {sr}Hz")

    log_message("Starting parallel feature extraction...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            'beats': executor.submit(_extract_beats, y, sr),
            'rms': executor.submit(_extract_rms, y, sr, HOP_LENGTH_FEATURES),
            'sc': executor.submit(_extract_spectral_centroid, y, sr, HOP_LENGTH_FEATURES),
            'onsets': executor.submit(_extract_onsets, y, sr),
            'flux': executor.submit(_extract_spectral_flux, y, sr, HOP_LENGTH_FEATURES)
        }
        results = {}
        try:
            results['tempo_info'], results['beat_times'] = futures['beats'].result()
            results['rms_frames'] = futures['rms'].result()
            results['spectral_centroid_frames'] = futures['sc'].result()
            results['onset_times'] = futures['onsets'].result()
            results['spectral_flux_frames'] = futures['flux'].result()
            log_message("All feature extraction tasks completed.")
        except Exception as e:
            log_message(f"Error during threaded feature extraction: {e}")
            for name, future in futures.items():
                if future.done() and future.exception():
                    log_message(f"{name.capitalize()} extraction exception: {future.exception()}")
            return None
    return y, sr, results

def generate_cues(sr, features, hop_length):
    potential_cues = []
    
    rms_frames = features['rms_frames']
    sc_frames = features['spectral_centroid_frames']
    flux_frames = features['spectral_flux_frames']

    if not all(hasattr(arr, 'size') and arr.size > 0 for arr in [rms_frames, sc_frames, flux_frames]):
        log_message("Warning: One or more feature arrays are empty or invalid. Cannot generate cues effectively.")
        return []

    valid_rms = rms_frames[rms_frames > 0]
    median_rms = np.median(valid_rms) if valid_rms.size > 0 else 0.01 
    std_rms = np.std(valid_rms) if valid_rms.size > 0 else 0.01
    
    rms_trigger_threshold_beat = median_rms + (std_rms * RMS_THRESHOLD_FACTOR_BEAT)
    
    valid_sc = sc_frames[sc_frames > 0]
    median_sc = np.median(valid_sc) if valid_sc.size > 0 else 1000 
    sc_beat_split_thresh = median_sc * SPECTRAL_CENTROID_THRESHOLD_FACTOR
    
    onset_rms_trigger_threshold = median_rms + (std_rms * ONSET_RMS_MULTIPLIER) 
    bass_sc_thresh = median_sc * BASS_CENTROID_LOW_FACTOR
    
    valid_flux = flux_frames[flux_frames > 0]
    median_flux = np.median(valid_flux) if valid_flux.size > 0 else 0.1
    flux_trigger_threshold = median_flux * SPECTRAL_FLUX_THRESHOLD_FACTOR
    sharp_onset_rms_thresh = median_rms + (std_rms * ONSET_RMS_SHARP_ACCENT_MULTIPLIER)


    log_message(f"RMS Beat Trigger: {rms_trigger_threshold_beat:.4f}, Onset RMS Trigger: {onset_rms_trigger_threshold:.4f}, Sharp Onset RMS: {sharp_onset_rms_thresh:.4f}")
    log_message(f"SC Beat Split: {sc_beat_split_thresh:.2f}, Bass SC: {bass_sc_thresh:.2f}")
    log_message(f"Flux Trigger: {flux_trigger_threshold:.4f} (Median Flux: {median_flux:.4f})")

    for onset_time in features['onset_times']:
        frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length)
        if not (0 <= frame_idx < len(rms_frames) and 0 <= frame_idx < len(sc_frames) and 0 <= frame_idx < len(flux_frames)):
            continue

        current_rms = rms_frames[frame_idx]
        current_sc = sc_frames[frame_idx]
        current_flux = flux_frames[frame_idx]

        if current_rms > onset_rms_trigger_threshold and current_sc < bass_sc_thresh:
            potential_cues.append({
                "timestamp": round(onset_time, 3), "firework": FIREWORK_TYPES["deep_bass"], "source": "onset_bass"
            })

        if current_rms > sharp_onset_rms_thresh and current_flux > flux_trigger_threshold:
            potential_cues.append({
                "timestamp": round(onset_time, 3), "firework": FIREWORK_TYPES["sharp_accent"], "source": "onset_flux"
            })
        elif current_rms > onset_rms_trigger_threshold : 
            is_already_bass = any(
                pc["timestamp"] == round(onset_time, 3) and pc["firework"] == FIREWORK_TYPES["deep_bass"]
                for pc in potential_cues[-2:]
            )
            if not is_already_bass:
                 potential_cues.append({
                    "timestamp": round(onset_time, 3), "firework": FIREWORK_TYPES["strong_kick"], "source": "onset_kick"
                })
    log_message(f"Generated {len(potential_cues)} potential cues from onsets.")
    
    onset_cues_count = len(potential_cues)

    for beat_time in features['beat_times']:
        frame_idx = librosa.time_to_frames(beat_time, sr=sr, hop_length=hop_length)
        if not (0 <= frame_idx < len(rms_frames) and 0 <= frame_idx < len(sc_frames)):
            continue
        
        current_rms = rms_frames[frame_idx]
        current_sc = sc_frames[frame_idx]

        if current_rms > rms_trigger_threshold_beat:
            firework_type = FIREWORK_TYPES["high_freq"] if current_sc > sc_beat_split_thresh else FIREWORK_TYPES["mid_energy"]
            potential_cues.append({
                "timestamp": round(beat_time, 3), "firework": firework_type, "source": "beat"
            })
    log_message(f"Generated {len(potential_cues) - onset_cues_count} potential cues from beats.")

    potential_cues.sort(key=lambda x: (x["timestamp"], x["firework"]))
    log_message(f"Total potential cues before filtering: {len(potential_cues)}")

    final_cues = []
    last_cue_times_by_type = {fw_type: -float('inf') for fw_type in FIREWORK_TYPES.values()}
    last_particle_cue_time = -float('inf')
    
    dropped_same_type = 0
    dropped_particle_proximity = 0

    for cue in potential_cues:
        ts = cue["timestamp"]
        fw_type = cue["firework"]

        if ts - last_cue_times_by_type[fw_type] < MIN_TIME_BETWEEN_SAME_TYPE_CUES:
            dropped_same_type +=1
            continue
        
        is_particle_cue = fw_type in PARTICLE_FIREWORK_TYPES
        if is_particle_cue:
            if ts - last_particle_cue_time < MIN_TIME_BETWEEN_ANY_PARTICLE_CUES:
                can_override_proximity = False
                if final_cues and final_cues[-1]["timestamp"] == ts and final_cues[-1]["firework"] in PARTICLE_FIREWORK_TYPES and final_cues[-1]["firework"] != fw_type:
                    can_override_proximity = True
                
                if not can_override_proximity:
                    dropped_particle_proximity +=1
                    continue
            
            last_particle_cue_time = ts

        final_cues.append(cue)
        last_cue_times_by_type[fw_type] = ts

    log_message(f"Dropped {dropped_same_type} cues due to same-type cooldown.")
    log_message(f"Dropped {dropped_particle_proximity} cues due to particle proximity cooldown.")
    log_message(f"Generated {len(final_cues)} final cues.")
    return final_cues


def main():
    parser = argparse.ArgumentParser(description='HanabiSync - Advanced Cues')
    parser.add_argument('-f', '--file', help='Path to audio file')
    args = parser.parse_args()

    audio_file_path = find_audio_file(args.file)
    if not audio_file_path: return

    output_dir = ensure_output_directory()
    analysis_data = analyze_audio(audio_file_path)
    if not analysis_data: return

    _y, sr, features_dict = analysis_data
    firework_cues = generate_cues(sr, features_dict, HOP_LENGTH_FEATURES)

    output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_cues.json")
    
    log_message(f"\n--- Generated Cues ({len(firework_cues)}) ---")
    for cue in firework_cues[:10]:
        log_message(f"  {cue['timestamp']:.3f}s - {cue['firework']:<12} (from {cue['source']})")
    if len(firework_cues) > 10:
        log_message(f"  ... and {len(firework_cues) - 10} more cues.")

    try:
        cues_to_save = [{"timestamp": c["timestamp"], "firework": c["firework"]} for c in firework_cues]
        with open(output_filename, 'w') as f: json.dump(cues_to_save, f, indent=2)
        log_message(f"\nCue map saved to {output_filename}")
    except IOError as e:
        log_message(f"Error saving cue map: {e}")
    
    log_message("🎆🔊 HanabiSync Advanced PoC complete! Tune thresholds and cooldowns for best results. 🔊🎆")
    log_message("Future idea: Explore librosa.segment.recurrence_matrix for detecting truly repeating song patterns.")

if __name__ == "__main__":
    main()