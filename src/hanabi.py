import librosa
import numpy as np
import json
import os
import datetime
import concurrent.futures
import argparse
import glob
import uuid

# --- Global Configuration ---
FIREWORK_TYPES = {
    "mid_energy": "mid_burst",
    "high_freq": "high_sparkle",
    "strong_kick": "kick_boom",
    "deep_bass": "bass_pulse",
    "sharp_accent": "comet_tail"
}
HOP_LENGTH_FEATURES = 512
PARTICLE_FIREWORK_TYPES = [
    FIREWORK_TYPES["mid_energy"], FIREWORK_TYPES["high_freq"],
    FIREWORK_TYPES["strong_kick"], FIREWORK_TYPES["sharp_accent"]
]

# --- Profile Definitions ---
PROFILES = {
    "balanced": {
        "dynamic_threshold_keys": {
            "key_bass_sc": 'p45', "key_onset_rms_general": 'p55', "key_rms_beat": 'p65',
            "key_sc_beat_split": 'p70', "key_flux_sharp": 'p85', "key_rms_sharp": 'p80',
        },
        "dynamic_threshold_keys_boost": {
            "key_bass_sc": 'p45', "key_onset_rms_general": 'p40', "key_rms_beat": 'p50',
            "key_sc_beat_split": 'p60', "key_flux_sharp": 'p75', "key_rms_sharp": 'p70',
        },
        "min_time_between_same_type_cues": 0.15,
        "min_time_between_any_particle_cues": 0.08,
        "quiet_duration_for_boost": 5.0,
        "cue_generation_logic_tweaks": {"allow_kick_with_others_on_onset": False}
    },
    "noisy": {
        "dynamic_threshold_keys": {
            "key_bass_sc": 'p35', "key_onset_rms_general": 'p40', "key_rms_beat": 'p50',
            "key_sc_beat_split": 'p60', "key_flux_sharp": 'p70', "key_rms_sharp": 'p65',
        },
        "dynamic_threshold_keys_boost": {
            "key_bass_sc": 'p30', "key_onset_rms_general": 'p30', "key_rms_beat": 'p40',
            "key_sc_beat_split": 'p50', "key_flux_sharp": 'p60', "key_rms_sharp": 'p55',
        },
        "min_time_between_same_type_cues": 0.08,
        "min_time_between_any_particle_cues": 0.04,
        "quiet_duration_for_boost": 3.0,
        "cue_generation_logic_tweaks": {"allow_kick_with_others_on_onset": True}
    },
    "quiet": {
        "dynamic_threshold_keys": {
            "key_bass_sc": 'p55', "key_onset_rms_general": 'p70', "key_rms_beat": 'p75',
            "key_sc_beat_split": 'p80', "key_flux_sharp": 'p90', "key_rms_sharp": 'p85',
        },
        "dynamic_threshold_keys_boost": {
            "key_bass_sc": 'p50', "key_onset_rms_general": 'p60', "key_rms_beat": 'p65',
            "key_sc_beat_split": 'p70', "key_flux_sharp": 'p80', "key_rms_sharp": 'p75',
        },
        "min_time_between_same_type_cues": 0.30,
        "min_time_between_any_particle_cues": 0.15,
        "quiet_duration_for_boost": 8.0,
        "cue_generation_logic_tweaks": {"allow_kick_with_others_on_onset": False}
    }
}


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
            try: tempo_for_logging_str = f"{tempo_from_librosa.item():.2f}"
            except: tempo_for_logging_str = str(tempo_from_librosa.item())
        elif tempo_from_librosa.size > 1:
            try: mean_tempo = np.mean(tempo_from_librosa); tempo_for_logging_str = f"{mean_tempo:.2f} (mean of {len(tempo_from_librosa)} tempos)"
            except: tempo_for_logging_str = f"Array of {len(tempo_from_librosa)} tempos"
            log_message(f"Note: Beat tracking returned an array of tempos: {tempo_from_librosa}")
        else: tempo_for_logging_str = "Empty array"; log_message("Warning: Beat tracking returned an empty tempo array.")
    elif tempo_from_librosa is not None:
        try: tempo_for_logging_str = f"{float(tempo_from_librosa):.2f}"
        except (TypeError, ValueError): tempo_for_logging_str = str(tempo_from_librosa); log_message(f"Warning: Could not convert tempo value {tempo_from_librosa} to float for formatted logging.")
    else: log_message("Warning: Beat tracking returned None for tempo.")
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
    try: y, sr = librosa.load(audio_path)
    except Exception as e: log_message(f"Error loading audio: {e}. Ensure ffmpeg is installed for MP3s."); return None
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
            for name, future_item in futures.items():
                if future_item.done() and future_item.exception():
                    log_message(f"{name.capitalize()} extraction exception: {future_item.exception()}")
            return None
    return y, sr, results

def calculate_feature_profile(feature_array, feature_name="Feature", filter_zeros=True):
    profile = {}
    percentiles_to_define = sorted(list(set([
        10, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95
    ])))

    def get_default_profile_dict():
        default_val = 0.0
        default_p_dict = {
            'min': default_val, 'max': default_val, 'mean': default_val, 'std': default_val, 'count': 0
        }
        for p_key_val in percentiles_to_define: default_p_dict[f'p{p_key_val}'] = default_val
        default_p_dict['median'] = default_p_dict.get('p50', default_val)
        return default_p_dict

    if not (hasattr(feature_array, 'size') and feature_array.size > 0):
        log_message(f"Warning: Empty or invalid array for {feature_name} profile. Returning default profile.")
        return get_default_profile_dict()

    values = feature_array.astype(float)[feature_array > 1e-6] if filter_zeros else feature_array.astype(float)

    if values.size == 0:
        log_message(f"Warning: No valid values (after filtering) in {feature_name} for profile. Checking original or returning default.")
        if filter_zeros and feature_array.size > 0: values = feature_array.astype(float)
        if values.size == 0: log_message(f"Warning: Array for {feature_name} is effectively empty. Returning default profile."); return get_default_profile_dict()

    profile['min'] = np.min(values); profile['max'] = np.max(values); profile['mean'] = np.mean(values)
    profile['std'] = np.std(values); profile['count'] = values.size
    
    calculated_percentiles_values = np.percentile(values, percentiles_to_define)
    for p_key_val, p_numeric_val in zip(percentiles_to_define, calculated_percentiles_values): profile[f'p{p_key_val}'] = p_numeric_val
    profile['median'] = profile.get('p50', np.median(values))

    log_message(
        f"{feature_name} Profile (elements: {profile['count']}): "
        f"Min={profile['min']:.2f}, Max={profile['max']:.2f}, Mean={profile['mean']:.2f}, "
        f"Median={profile.get('median', 'N/A'):.2f}, Std={profile['std']:.2f}, "
        f"P25={profile.get('p25', 'N/A'):.2f}, P75={profile.get('p75', 'N/A'):.2f}, P90={profile.get('p90', 'N/A'):.2f}"
    )
    return profile

class DynamicThresholds:
    def __init__(self, rms_profile, sc_profile, flux_profile, threshold_keys_config):
        self.rms_profile = rms_profile
        self.sc_profile = sc_profile
        self.flux_profile = flux_profile

        key_bass_sc = threshold_keys_config['key_bass_sc']
        key_onset_rms_general = threshold_keys_config['key_onset_rms_general']
        key_rms_beat = threshold_keys_config['key_rms_beat']
        key_sc_beat_split = threshold_keys_config['key_sc_beat_split']
        key_flux_sharp = threshold_keys_config['key_flux_sharp']
        key_rms_sharp = threshold_keys_config['key_rms_sharp']
        
        def get_profile_value(profile, key, feature_type_for_fallback="generic"):
            default_key = 'p50'
            if feature_type_for_fallback == "sc_bass" and 'p25' in profile:
                 default_key = 'p25' if 'p25' in profile else 'p10' if 'p10' in profile else 'p50'
            
            if key not in profile:
                log_message(f"Warning: Percentile key '{key}' not found in {feature_type_for_fallback} profile. Falling back to '{default_key}'.")
                chosen_key = default_key
                if chosen_key not in profile:
                    log_message(f"Critical Warning: Fallback key '{chosen_key}' also not found for {feature_type_for_fallback}. Using raw median if possible, else 0.")
                    return profile.get('median', 0.0) 
            else:
                chosen_key = key
            return profile[chosen_key]

        self.rms_trigger_threshold_beat = get_profile_value(self.rms_profile, key_rms_beat, "RMS")
        self.sc_beat_split_thresh = get_profile_value(self.sc_profile, key_sc_beat_split, "SC")
        self.bass_sc_thresh = get_profile_value(self.sc_profile, key_bass_sc, "sc_bass")
        self.onset_rms_trigger_threshold = get_profile_value(self.rms_profile, key_onset_rms_general, "RMS")
        self.flux_trigger_threshold = get_profile_value(self.flux_profile, key_flux_sharp, "Flux")
        self.sharp_onset_rms_thresh = get_profile_value(self.rms_profile, key_rms_sharp, "RMS")

def generate_cues(sr, features, hop_length, profile_settings):
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
    for onset_time in features.get('onset_times', []): combined_events.append({'time': onset_time, 'type': 'onset'})
    for beat_time in features.get('beat_times', []): combined_events.append({'time': beat_time, 'type': 'beat'})
    combined_events.sort(key=lambda x: x['time'])

    time_of_last_generated_cue_overall = -float('inf')
    sensitivity_boost_active = False
    
    initial_threshold_keys = profile_settings['dynamic_threshold_keys']
    current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, initial_threshold_keys)
    log_message(f"--- Dynamically Determined Thresholds for Profile '{profile_settings['name']}' (Boost: False) ---")
    log_message(f"  RMS Beat Trigger ({initial_threshold_keys['key_rms_beat']}): {current_thresholds.rms_trigger_threshold_beat:.4f}")
    log_message(f"  SC Beat Split ({initial_threshold_keys['key_sc_beat_split']}): {current_thresholds.sc_beat_split_thresh:.4f}")
    log_message(f"  Bass SC ({initial_threshold_keys['key_bass_sc']}): {current_thresholds.bass_sc_thresh:.4f}")
    log_message(f"  Onset RMS General ({initial_threshold_keys['key_onset_rms_general']}): {current_thresholds.onset_rms_trigger_threshold:.4f}")
    log_message(f"  Flux Sharp ({initial_threshold_keys['key_flux_sharp']}): {current_thresholds.flux_trigger_threshold:.4f}")
    log_message(f"  RMS Sharp ({initial_threshold_keys['key_rms_sharp']}): {current_thresholds.sharp_onset_rms_thresh:.4f}")
    log_message("----------------------------------------------------")


    log_message(f"Processing {len(combined_events)} time-sorted onsets/beats...")
    quiet_duration_for_boost = profile_settings['quiet_duration_for_boost']

    def _add_potential_cue(timestamp, firework_type, source_description):
        potential_cues.append({
            "id": str(uuid.uuid4()),
            "timestamp": round(timestamp, 3),
            "firework": firework_type,
            "source": source_description
        })

    for event_data in combined_events:
        event_time = event_data['time']
        event_type = event_data['type']

        if not sensitivity_boost_active and (event_time - time_of_last_generated_cue_overall > quiet_duration_for_boost):
            if quiet_duration_for_boost != float('inf'): 
                sensitivity_boost_active = True
                boost_threshold_keys = profile_settings['dynamic_threshold_keys_boost']
                current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, boost_threshold_keys)
                log_message(f"Sensitivity Boost ACTIVATED at {event_time:.2f}s for profile '{profile_settings['name']}'.")
                log_message(f"  Boosted RMS Beat Trigger ({boost_threshold_keys['key_rms_beat']}): {current_thresholds.rms_trigger_threshold_beat:.4f}")
                log_message(f"  Boosted SC Beat Split ({boost_threshold_keys['key_sc_beat_split']}): {current_thresholds.sc_beat_split_thresh:.4f}")
                log_message(f"  Boosted Bass SC ({boost_threshold_keys['key_bass_sc']}): {current_thresholds.bass_sc_thresh:.4f}")
                log_message(f"  Boosted Onset RMS General ({boost_threshold_keys['key_onset_rms_general']}): {current_thresholds.onset_rms_trigger_threshold:.4f}")
                log_message(f"  Boosted Flux Sharp ({boost_threshold_keys['key_flux_sharp']}): {current_thresholds.flux_trigger_threshold:.4f}")
                log_message(f"  Boosted RMS Sharp ({boost_threshold_keys['key_rms_sharp']}): {current_thresholds.sharp_onset_rms_thresh:.4f}")

        frame_idx = librosa.time_to_frames(event_time, sr=sr, hop_length=hop_length)
        if not (0 <= frame_idx < len(rms_frames) and 0 <= frame_idx < len(sc_frames) and \
                (event_type != 'onset' or 0 <= frame_idx < len(flux_frames))):
            continue
        
        current_rms = rms_frames[frame_idx]
        current_sc = sc_frames[frame_idx]
        generated_types_this_event = []

        if event_type == 'onset':
            current_flux = flux_frames[frame_idx]
            if current_rms > current_thresholds.onset_rms_trigger_threshold and current_sc < current_thresholds.bass_sc_thresh:
                _add_potential_cue(event_time, FIREWORK_TYPES["deep_bass"], "onset_bass")
                generated_types_this_event.append(FIREWORK_TYPES["deep_bass"])
            if current_rms > current_thresholds.sharp_onset_rms_thresh and current_flux > current_thresholds.flux_trigger_threshold:
                _add_potential_cue(event_time, FIREWORK_TYPES["sharp_accent"], "onset_flux")
                generated_types_this_event.append(FIREWORK_TYPES["sharp_accent"])
            
            if current_rms > current_thresholds.onset_rms_trigger_threshold: 
                allow_kick_with_others = profile_settings.get("cue_generation_logic_tweaks", {}).get("allow_kick_with_others_on_onset", False)
                can_trigger_kick = False
                if allow_kick_with_others: can_trigger_kick = True
                else:
                    if not any(fw_type in generated_types_this_event for fw_type in [FIREWORK_TYPES["deep_bass"], FIREWORK_TYPES["sharp_accent"]]):
                        can_trigger_kick = True
                if can_trigger_kick:
                    _add_potential_cue(event_time, FIREWORK_TYPES["strong_kick"], "onset_kick")
                    generated_types_this_event.append(FIREWORK_TYPES["strong_kick"])
        
        elif event_type == 'beat':
            if current_rms > current_thresholds.rms_trigger_threshold_beat:
                firework_type = FIREWORK_TYPES["high_freq"] if current_sc > current_thresholds.sc_beat_split_thresh else FIREWORK_TYPES["mid_energy"]
                _add_potential_cue(event_time, firework_type, "beat")
                generated_types_this_event.append(firework_type)

        if generated_types_this_event:
            time_of_last_generated_cue_overall = round(event_time, 3)
            if sensitivity_boost_active:
                sensitivity_boost_active = False
                normal_threshold_keys = profile_settings['dynamic_threshold_keys']
                current_thresholds = DynamicThresholds(rms_profile, sc_profile, flux_profile, normal_threshold_keys)
                log_message(f"Sensitivity Boost DEACTIVATED at {event_time:.2f}s. Reverted to normal thresholds for profile '{profile_settings['name']}'.")

    log_message(f"Generated {len(potential_cues)} potential cues from combined event stream.")
    potential_cues.sort(key=lambda x: (x["timestamp"], x["firework"])) 
    
    deduplicated_cues = []
    if potential_cues:
        deduplicated_cues.append(potential_cues[0])
        for i in range(1, len(potential_cues)):
            if not (potential_cues[i]["timestamp"] == potential_cues[i-1]["timestamp"] and \
                    potential_cues[i]["firework"] == potential_cues[i-1]["firework"]):
                deduplicated_cues.append(potential_cues[i])
    log_message(f"Potential cues after basic content deduplication: {len(deduplicated_cues)}")

    final_cues = []
    last_cue_times_by_type = {fw_type: -float('inf') for fw_type in FIREWORK_TYPES.values()}
    last_particle_cue_time = -float('inf')
    
    min_same_type_cooldown = profile_settings['min_time_between_same_type_cues']
    min_any_particle_cooldown = profile_settings['min_time_between_any_particle_cues']
    dropped_same_type, dropped_particle_proximity = 0, 0

    for cue in deduplicated_cues:
        ts, fw_type = cue["timestamp"], cue["firework"]
        if ts - last_cue_times_by_type[fw_type] < min_same_type_cooldown: dropped_same_type +=1; continue
        if fw_type in PARTICLE_FIREWORK_TYPES:
            if ts - last_particle_cue_time < min_any_particle_cooldown: dropped_particle_proximity +=1; continue
            last_particle_cue_time = ts
        final_cues.append(cue)
        last_cue_times_by_type[fw_type] = ts

    log_message(f"Dropped {dropped_same_type} cues (same-type cooldown: {min_same_type_cooldown}s).")
    log_message(f"Dropped {dropped_particle_proximity} cues (particle proximity cooldown: {min_any_particle_cooldown}s).")
    log_message(f"Generated {len(final_cues)} final cues using profile '{profile_settings['name']}'.")
    return final_cues


def main():
    parser = argparse.ArgumentParser(description='HanabiSync - Audio to Firework Cue Generator with Profiles')
    parser.add_argument('-f', '--file', help='Path to audio file')
    parser.add_argument('--profile', choices=['noisy', 'balanced', 'quiet'], default='balanced', help='Generation profile (default: balanced).')
    args = parser.parse_args()

    audio_file_path = find_audio_file(args.file)
    if not audio_file_path: return

    output_dir = ensure_output_directory()
    analysis_data = analyze_audio(audio_file_path)
    if not analysis_data: return

    y_audio, sr, features_dict = analysis_data
    
    selected_profile_name = args.profile
    profile_settings = PROFILES[selected_profile_name].copy() 
    profile_settings["name"] = selected_profile_name 
    log_message(f"Using profile: {selected_profile_name}")

    firework_cues = generate_cues(sr, features_dict, HOP_LENGTH_FEATURES, profile_settings)
    output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_cues_{selected_profile_name}.json") 
    
    log_message(f"\n--- Generated Cues ({len(firework_cues)}) for profile '{selected_profile_name}' ---")
    for cue in firework_cues[:15]: log_message(f"  {cue['timestamp']:.3f}s - {cue['firework']:<12} (from {cue['source']}) ID: {cue['id'][:8]}...")
    if len(firework_cues) > 15: log_message(f"  ... and {len(firework_cues) - 15} more cues.")

    # Prepare data for JSON export - REMOVED audio_features
    output_data = {
        "cues": firework_cues,
        "audio_metadata": {
            "filename": os.path.basename(audio_file_path),
            "sample_rate": sr,
            "hop_length_features": HOP_LENGTH_FEATURES,
            "duration": librosa.get_duration(y=y_audio, sr=sr)
        }
        # "audio_features" section has been removed
    }

    try:
        with open(output_filename, 'w') as f: json.dump(output_data, f, indent=2)
        log_message(f"\nCue map and minimal audio metadata saved to {output_filename}")
    except IOError as e: log_message(f"Error saving cue map: {e}")
    
    log_message(f"🎆🔊 HanabiSync ({selected_profile_name} profile) complete! 🔊🎆")

if __name__ == "__main__":
    main()