import librosa
import numpy as np
import json
import os
import datetime
import concurrent.futures
import argparse
import glob

MIN_TIME_BETWEEN_SAME_TYPE_CUES = 0.15
MIN_TIME_BETWEEN_ANY_PARTICLE_CUES = 0.08
QUIET_DURATION_FOR_BOOST = 5.0

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

def calculate_feature_profile(feature_array, feature_name="Feature", filter_zeros=True):
    profile = {}
    percentiles_to_define = sorted(list(set([
        10, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
    ])))


    def get_default_profile():
        default_val = 0.0
        default_p = {
            'min': default_val, 'max': default_val, 'mean': default_val, 'std': default_val,
            'count': 0
        }
        for p_key_val in percentiles_to_define:
            default_p[f'p{p_key_val}'] = default_val
        default_p['median'] = default_p.get('p50', default_val)
        return default_p

    if not (hasattr(feature_array, 'size') and feature_array.size > 0):
        log_message(f"Warning: Empty or invalid array for {feature_name} profile. Returning default profile.")
        return get_default_profile()

    if filter_zeros:
        values = feature_array.astype(float)[feature_array > 1e-6] 
    else:
        values = feature_array

    if values.size == 0:
        log_message(f"Warning: No valid values (after filtering) in {feature_name} for profile. Checking original or returning default.")
        if filter_zeros and feature_array.size > 0: 
            values = feature_array
            log_message(f"Using original (unfiltered) array for {feature_name} profile as filtered was empty.")
        if values.size == 0:
            log_message(f"Warning: Array for {feature_name} is effectively empty. Returning default profile.")
            return get_default_profile()

    profile['min'] = np.min(values)
    profile['max'] = np.max(values)
    profile['mean'] = np.mean(values)
    profile['std'] = np.std(values)
    profile['count'] = values.size
    
    calculated_percentiles_values = np.percentile(values, percentiles_to_define)
    
    for p_key_val, p_numeric_val in zip(percentiles_to_define, calculated_percentiles_values):
        profile[f'p{p_key_val}'] = p_numeric_val
    
    profile['median'] = profile.get('p50', np.median(values))

    log_message(
        f"{feature_name} Profile (elements: {profile['count']}): "
        f"Min={profile['min']:.2f}, Max={profile['max']:.2f}, Mean={profile['mean']:.2f}, "
        f"Median={profile['median']:.2f} (P50), Std={profile['std']:.2f}, "
        f"P25={profile.get('p25', 'N/A'):.2f}, P75={profile.get('p75', 'N/A'):.2f}, P90={profile.get('p90', 'N/A'):.2f}"
    )
    return profile

class DynamicThresholds:
    def __init__(self, rms_profile, sc_profile, flux_profile, sensitivity_boost=False):
        self.rms_profile = rms_profile
        self.sc_profile = sc_profile
        self.flux_profile = flux_profile
        self.boost = sensitivity_boost

        key_bass_sc = 'p45'
        key_onset_rms_general = 'p55'

        key_rms_beat = 'p65'
        key_sc_beat_split = 'p70'

        key_flux_sharp = 'p85'
        key_rms_sharp = 'p80'

        if self.boost:
            key_bass_sc = 'p45' 
            key_onset_rms_general = 'p40' 
            key_rms_beat = 'p50'
            key_sc_beat_split = 'p60' 
            key_flux_sharp = 'p75'
            key_rms_sharp = 'p70'
        
        def get_profile_value(profile, key, feature_type_for_fallback="generic"):
            default_key = 'p50'
            if feature_type_for_fallback == "sc_bass" and 'p25' in profile :
                 default_key = 'p25' if 'p25' in profile else 'p10' if 'p10' in profile else 'p50'
            
            if key not in profile:
                log_message(f"Warning: Percentile key '{key}' not found in {feature_type_for_fallback} profile. Falling back to '{default_key}'. Ensure calculate_feature_profile computes all necessary percentiles.")
                chosen_key = default_key
                if chosen_key not in profile:
                    log_message(f"Critical Warning: Fallback key '{chosen_key}' also not found. Using raw median if possible.")
                    return profile.get('median', 0)
            else:
                chosen_key = key
            return profile[chosen_key]

        self.rms_trigger_threshold_beat = get_profile_value(self.rms_profile, key_rms_beat, "RMS")
        self.sc_beat_split_thresh = get_profile_value(self.sc_profile, key_sc_beat_split, "SC")

        self.bass_sc_thresh = get_profile_value(self.sc_profile, key_bass_sc, "sc_bass")
        self.onset_rms_trigger_threshold = get_profile_value(self.rms_profile, key_onset_rms_general, "RMS")

        self.flux_trigger_threshold = get_profile_value(self.flux_profile, key_flux_sharp, "Flux")
        self.sharp_onset_rms_thresh = get_profile_value(self.rms_profile, key_rms_sharp, "RMS")
        
        log_message(f"--- Dynamically Determined Thresholds (Boost: {self.boost}) ---")
        log_message(f"  RMS Beat Trigger ({key_rms_beat}): {self.rms_trigger_threshold_beat:.4f}")
        log_message(f"  SC Beat Split ({key_sc_beat_split} for high_freq): {self.sc_beat_split_thresh:.2f}")
        log_message(f"  Onset RMS General ({key_onset_rms_general} for kick/bass): {self.onset_rms_trigger_threshold:.4f}")
        log_message(f"  Bass SC ({key_bass_sc}): {self.bass_sc_thresh:.2f}")
        log_message(f"  Sharp Accent Flux ({key_flux_sharp}): {self.flux_trigger_threshold:.4f}")
        log_message(f"  Sharp Accent RMS ({key_rms_sharp}): {self.sharp_onset_rms_thresh:.4f}")
        log_message("----------------------------------------------------")


def generate_cues(sr, features, hop_length):
    potential_cues = []
    
    rms_frames = features['rms_frames']
    sc_frames = features['spectral_centroid_frames']
    flux_frames = features['spectral_flux_frames']

    if not all(hasattr(arr, 'size') and arr.size > 0 for arr in [rms_frames, sc_frames, flux_frames]):
        log_message("Warning: One or more feature arrays are empty. Cannot generate cues effectively.")
        return []

    rms_profile = calculate_feature_profile(rms_frames, "RMS")
    sc_profile = calculate_feature_profile(sc_frames, "Spectral Centroid")
    flux_profile = calculate_feature_profile(flux_frames, "Spectral Flux", filter_zeros=False)

    combined_events = []
    for onset_time in features.get('onset_times', []):
        combined_events.append({'time': onset_time, 'type': 'onset'})
    for beat_time in features.get('beat_times', []):
        combined_events.append({'time': beat_time, 'type': 'beat'})
    
    combined_events.sort(key=lambda x: x['time'])

    time_of_last_generated_cue_overall = -float('inf')
    sensitivity_boost_active = False
    current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, sensitivity_boost=False)

    log_message(f"Processing {len(combined_events)} time-sorted onsets/beats...")

    for event_data in combined_events:
        event_time = event_data['time']
        event_type = event_data['type']

        if not sensitivity_boost_active and (event_time - time_of_last_generated_cue_overall > QUIET_DURATION_FOR_BOOST):
            sensitivity_boost_active = True
            current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, sensitivity_boost=True)
            log_message(f"Sensitivity Boost ACTIVATED at {event_time:.2f}s (Gap: {event_time - time_of_last_generated_cue_overall:.2f}s)")
        
        frame_idx = librosa.time_to_frames(event_time, sr=sr, hop_length=hop_length)
        if not (0 <= frame_idx < len(rms_frames) and \
                0 <= frame_idx < len(sc_frames) and \
                (event_type != 'onset' or 0 <= frame_idx < len(flux_frames))):
            continue
        
        current_rms = rms_frames[frame_idx]
        current_sc = sc_frames[frame_idx]
        cue_generated_this_event = False

        if event_type == 'onset':
            current_flux = flux_frames[frame_idx]
            if current_rms > current_thresholds.onset_rms_trigger_threshold and current_sc < current_thresholds.bass_sc_thresh:
                potential_cues.append({
                    "timestamp": round(event_time, 3), "firework": FIREWORK_TYPES["deep_bass"], "source": "onset_bass"
                })
                cue_generated_this_event = True
            if current_rms > current_thresholds.sharp_onset_rms_thresh and current_flux > current_thresholds.flux_trigger_threshold:
                potential_cues.append({
                    "timestamp": round(event_time, 3), "firework": FIREWORK_TYPES["sharp_accent"], "source": "onset_flux"
                })
                cue_generated_this_event = True
            elif current_rms > current_thresholds.onset_rms_trigger_threshold:
                is_already_bass_or_sharp_at_this_onset = any(
                    pc["timestamp"] == round(event_time, 3) and 
                    (pc["firework"] == FIREWORK_TYPES["deep_bass"] or pc["firework"] == FIREWORK_TYPES["sharp_accent"])
                    for pc in potential_cues[-2:]
                )
                if not is_already_bass_or_sharp_at_this_onset:
                    potential_cues.append({
                        "timestamp": round(event_time, 3), "firework": FIREWORK_TYPES["strong_kick"], "source": "onset_kick"
                    })
                    cue_generated_this_event = True
        
        elif event_type == 'beat':
            if current_rms > current_thresholds.rms_trigger_threshold_beat:
                firework_type = FIREWORK_TYPES["high_freq"] if current_sc > current_thresholds.sc_beat_split_thresh else FIREWORK_TYPES["mid_energy"]
                potential_cues.append({
                    "timestamp": round(event_time, 3), "firework": firework_type, "source": "beat"
                })
                cue_generated_this_event = True

        if cue_generated_this_event:
            time_of_last_generated_cue_overall = round(event_time, 3)
            if sensitivity_boost_active:
                sensitivity_boost_active = False
                current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, sensitivity_boost=False)
                log_message(f"Sensitivity Boost DEACTIVATED at {event_time:.2f}s due to new cue.")

    log_message(f"Generated {len(potential_cues)} potential cues from combined event stream.")
    
    potential_cues.sort(key=lambda x: (x["timestamp"], x["firework"]))
    
    deduplicated_cues = []
    if potential_cues:
        deduplicated_cues.append(potential_cues[0])
        for i in range(1, len(potential_cues)):
            if not (potential_cues[i]["timestamp"] == potential_cues[i-1]["timestamp"] and \
                    potential_cues[i]["firework"] == potential_cues[i-1]["firework"]):
                deduplicated_cues.append(potential_cues[i])
    log_message(f"Potential cues after basic deduplication: {len(deduplicated_cues)}")

    final_cues = []
    last_cue_times_by_type = {fw_type: -float('inf') for fw_type in FIREWORK_TYPES.values()}
    last_particle_cue_time = -float('inf')
    
    dropped_same_type = 0
    dropped_particle_proximity = 0

    for cue in deduplicated_cues:
        ts = cue["timestamp"]
        fw_type = cue["firework"]

        if ts - last_cue_times_by_type[fw_type] < MIN_TIME_BETWEEN_SAME_TYPE_CUES:
            dropped_same_type +=1
            continue
        
        is_particle_cue = fw_type in PARTICLE_FIREWORK_TYPES
        if is_particle_cue:
            if ts - last_particle_cue_time < MIN_TIME_BETWEEN_ANY_PARTICLE_CUES:
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
    parser = argparse.ArgumentParser(description='HanabiSync - Auto-Thresholds & Adaptive Sensitivity')
    parser.add_argument('-f', '--file', help='Path to audio file')
    args = parser.parse_args()

    audio_file_path = find_audio_file(args.file)
    if not audio_file_path: return

    output_dir = ensure_output_directory()
    analysis_data = analyze_audio(audio_file_path)
    if not analysis_data: return

    _y, sr, features_dict = analysis_data
    firework_cues = generate_cues(sr, features_dict, HOP_LENGTH_FEATURES)

    output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_cues_adaptive.json")
    
    log_message(f"\n--- Generated Cues ({len(firework_cues)}) ---")
    for cue in firework_cues[:15]:
        log_message(f"  {cue['timestamp']:.3f}s - {cue['firework']:<12} (from {cue['source']})")
    if len(firework_cues) > 15:
        log_message(f"  ... and {len(firework_cues) - 15} more cues.")

    try:
        cues_to_save = [{"timestamp": c["timestamp"], "firework": c["firework"]} for c in firework_cues]
        with open(output_filename, 'w') as f: json.dump(cues_to_save, f, indent=2)
        log_message(f"\nCue map saved to {output_filename}")
    except IOError as e:
        log_message(f"Error saving cue map: {e}")
    
    log_message("🎆🔊 HanabiSync Adaptive PoC complete! Tune percentiles in DynamicThresholds & QUIET_DURATION_FOR_BOOST. 🔊🎆")

if __name__ == "__main__":
    main()