# HanabiSync: The Deep Dive

The primary tool is [Librosa](https://librosa.org/), a powerful Python library for music and audio analysis.  

### Audio Loading & Preprocessing

*   **Loading:** The script first loads the audio file using `librosa.load()`. This converts the audio waveform into a numerical array (`y`) and provides the sample rate (`sr`).
*   **Dependencies:** For compatibility with various compressed audio formats (like MP3s), `librosa` often relies on `ffmpeg`. If you encounter loading errors, ensure `ffmpeg` is correctly installed and its executable is in your system's PATH.
*   **Framing:** Many audio features are extracted by breaking the audio into small, overlapping chunks called "frames." The `HOP_LENGTH_FEATURES` constant (defaulting to 512 samples) defines the number of samples between successive frames. A smaller hop length means more frames and finer temporal resolution, but also more computation.

### Parallel Feature Extraction

To efficiently analyze the audio, HanabiSync employs parallel processing for feature extraction. While the raw feature arrays themselves are extensive and *not* exported in the final JSON cue file (to keep output manageable), they are fundamental for internal analysis:

*   **Beat Tracking:** (`librosa.beat.beat_track`)
    *   **What it is:** Identifies the underlying tempo (BPM) and the exact timestamps of musical beats. This is often based on identifying periodic patterns in the audio's rhythmic content.
    *   **Why it's used:** Provides a fundamental rhythmic grid for placing `mid_burst` and `high_sparkle` cues, ensuring they align with the music's pulse.
*   **Onset Detection:** (`librosa.onset.onset_detect`)
    *   **What it is:** Pinpoints moments where there's a sudden, significant increase in energy or a rapid change in the sound's characteristics. Think of a drum hit, a sharp guitar strum, or a vocal consonant.
    *   **Why it's used:** Crucial for triggering percussive effects like `kick_boom`, `bass_pulse`, and `comet_tail` which often occur at sharp transients.
*   **RMS Energy Analysis (Root Mean Square):** (`librosa.feature.rms`)
    *   **What it is:** Measures the overall loudness or intensity of the audio within each frame. A higher RMS value indicates a louder sound.
    *   **Why it's used:** Acts as a primary indicator for how energetic or impactful a moment is. High RMS at a beat might trigger a `mid_burst` or `high_sparkle`, while high RMS at an onset suggests a `kick_boom`.
*   **Spectral Centroid:** (`librosa.feature.spectral_centroid`)
    *   **What it is:** Represents the "brightness" or "darkness" of a sound. It's the weighted mean of the frequencies present in a sound, where the weights are the magnitudes of the frequencies. A higher spectral centroid means more energy is concentrated in higher frequencies (brighter sound), while a lower one indicates more energy in lower frequencies (darker/bassier sound).
    *   **Why it's used:** Helps differentiate between high-frequency (e.g., cymbals, synths) and low-frequency (e.g., kick drums, basslines) events. Used to split beat-based cues (e.g., `high_sparkle` vs. `mid_burst`) and to identify bass-specific onsets.
*   **Spectral Flux (Onset Strength):** (`librosa.onset.onset_strength`)
    *   **What it is:** This is a key measure for detecting sharp transients. It quantifies how rapidly the *spectral content* (the distribution of frequencies) of the sound changes from one frame to the next. A high spectral flux indicates a sudden shift in the sound's "timbre" or "color," often associated with sharp attacks.
    *   **Why it's used:** Specifically targets events like quick cymbal hits, vocal plosives, or sharp instrumental accents, which are perfect for `comet_tail` cues. It's distinct from RMS, which only measures loudness change; spectral flux focuses on *timbral* change.

### Profile-Driven Cue Logic

HanabiSync doesn't rely on fixed thresholds, which would perform poorly across different genres or recording qualities. Instead, it uses a sophisticated, adaptive, and profile-driven system:

*   **Statistical Feature Profiling (`calculate_feature_profile` function):**
    *   For each extracted feature (RMS, Spectral Centroid, Spectral Flux), the script calculates a statistical profile, including minimum, maximum, mean, standard deviation, and critically, various **percentiles** (e.g., P25, P50, P75, P90).
    *   **Why Percentiles?** Percentiles are robust. Instead of saying "an RMS value of 0.05 is loud," which is arbitrary, we can say "an RMS value is louder than 75% of all other RMS values in *this specific song*." This makes the analysis adaptive to the unique dynamics of each audio track. For instance, `p55` refers to the value at which 55% of the feature data falls below it.
*   **Selectable Profiles (`balanced`, `noisy`, `quiet`):**
    *   Users choose a profile via a CLI argument. Each profile (`PROFILES` dictionary in `hanabi.py`) defines a set of **threshold keys** (e.g., `'key_bass_sc': 'p45'`). These keys dictate *which percentile* of a given feature should be used as a trigger threshold.
    *   **How they differ:**
        *   `balanced`: Uses mid-to-high percentiles for a moderate amount of cues.
        *   `noisy`: Uses lower percentiles, leading to more cues being generated, suitable for dense or highly dynamic tracks where you want to catch more events.
        *   `quiet`: Uses higher percentiles, resulting in fewer, more impactful cues, ideal for sparse arrangements or quieter tracks where you want to avoid triggering on subtle background noise.
*   **Dynamic Thresholds (`DynamicThresholds` class):**
    *   This class takes the statistical profiles of RMS, SC, and Flux, and the chosen profile's `threshold_keys`. It then dynamically sets the actual numerical trigger values based on the specified percentiles. For example, if `key_rms_beat` is `p65`, the `rms_trigger_threshold_beat` will be set to the 65th percentile value of the song's RMS profile.
*   **Sensitivity Boost:**
    *   During quiet sections of a song (defined by `quiet_duration_for_boost` seconds without a triggered cue), the script can temporarily activate a "sensitivity boost."
    *   **How it works:** It switches to a secondary set of `dynamic_threshold_keys_boost` defined in the profile, which typically use *lower* percentiles (meaning lower numerical thresholds). This allows the system to pick up more subtle events in quieter passages.
    *   **Deactivation:** Once a significant event is detected and a cue is generated, the boost is deactivated, and the system reverts to the normal (higher percentile) thresholds, preventing an overwhelming number of cues during louder sections.
*   **Cue Generation Rules:**
    *   A sophisticated set of if/else logic maps combinations of feature values at beats or onsets to specific firework types. Examples:
        *   `bass_pulse`: If an onset's RMS is above `onset_rms_trigger_threshold` AND its Spectral Centroid is *below* `bass_sc_thresh` (indicating low-frequency content).
        *   `comet_tail`: If an onset's RMS is above `sharp_onset_rms_thresh` AND its Spectral Flux is above `flux_trigger_threshold` (indicating a sharp timbral change).
        *   `high_sparkle` vs. `mid_burst`: If a beat's RMS is above `rms_trigger_threshold_beat`, then its Spectral Centroid (`sc_beat_split_thresh`) determines if it's a `high_sparkle` (brighter) or `mid_burst` (general).
    *   **`allow_kick_with_others_on_onset`:** A profile-specific tweak to allow `kick_boom` to fire even if a `bass_pulse` or `comet_tail` already triggered on the same onset. `noisy` profiles might enable this for more activity.
*   **Cooldown Mechanisms:**
    *   To prevent an overly dense or visually jarring sequence of fireworks, cooldowns are applied:
        *   `min_time_between_same_type_cues`: Prevents the same firework type from triggering too rapidly (e.g., no `high_sparkle` within 0.15s of the last one).
        *   `min_time_between_any_particle_cues`: A stricter cooldown for all particle-based fireworks (excluding `bass_pulse`, which is a background effect). This ensures a visually pleasing separation between distinct explosions.

### Defined Firework Types

The script maps detected audio events to one of the following symbolic firework types, each designed to capture a specific sonic characteristic:

*   `mid_burst`: Associated with general energy on musical beats with moderate frequency content.
*   `high_sparkle`: Triggered by energetic beats characterized by bright, high-frequency sounds.
*   `kick_boom`: Fired by strong, impactful percussive onsets, typically like kick drums or other powerful rhythmic hits.
*   `bass_pulse`: Identified by deep, resonant bass frequencies detected during onsets.
*   `comet_tail`: Generated by sharp, quick accents in the music that show a rapid change in spectral content (high spectral flux).

### JSON Output Structure

The final output is a JSON file, designed to be concise and actionable. It contains the generated cues and essential audio metadata.  
