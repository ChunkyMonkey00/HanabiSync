# HanabiSync 🎆🔊

**Turn your music into fireworks.**  
HanabiSync is an open-source project featuring a Python-based audio analysis tool that generates synchronized firework cue maps from any song or audio track, and an optional web-based visualizer to bring those cues to life.

Perfect for backyard firework shows, synced holiday displays, creating music videos, or just nerding out with music-reactive explosions.

---

## 🚀 What It Does

HanabiSync has two main components:

1.  **Python Audio Analyzer (`src/hanabi.py`):**
    *   Analyzes an audio file using techniques like beat tracking, onset detection, RMS energy, spectral centroid, and spectral flux analysis via the Librosa library.
    *   Utilizes selectable **Profiles** (`balanced`, `noisy`, `quiet`) to tailor the sensitivity and characteristics of cue generation to different audio environments.
    *   Generates a timestamped cue map identifying moments for specific firework effects, each with a unique ID and source.
    *   Outputs a JSON file containing the cues and essential audio metadata. Example structure:
      ```json
      {
        "cues": [
          { 
            "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            "timestamp": 2.531, 
            "firework": "mid_burst",
            "source": "beat" 
          },
          { 
            "id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
            "timestamp": 4.115, 
            "firework": "high_sparkle",
            "source": "beat"
          }
          // ... more cues
        ],
        "audio_metadata": {
          "filename": "your_song.mp3",
          "sample_rate": 44100,
          "hop_length_features": 512,
          "duration": 210.75
        }
      }
      ```
    *   This cue file can be used with the included web visualizer or fed into a microcontroller, Raspberry Pi, or custom control system.

2.  **Web Visualizer (`hanabiVisualizer/`):** (Optional)
    *   Plays your audio file and the generated JSON cue file in sync.
    *   Renders dynamic firework animations on an HTML5 Canvas, corresponding to the cue types.
    *   Includes playback controls (play, pause, stop, seek) and a log of triggered events.

---

## 🎶 Audio Analysis Engine (Python Core)

The `src/hanabi.py` script dives deep into your audio to find the perfect moments for fireworks:

*   **Audio Loading & Preprocessing:** Uses [Librosa](https://librosa.org/) to load various audio formats.
*   **Parallel Feature Extraction:** Efficiently analyzes audio by running multiple extraction processes concurrently. While the raw feature arrays are not exported in the final JSON, they are crucial for internal analysis:
    *   **Beat Tracking:** Identifies the song's tempo and the timing of individual beats.
    *   **Onset Detection:** Pinpoints sudden increases in energy or changes in sound.
    *   **RMS Energy Analysis:** Gauges the loudness and intensity.
    *   **Spectral Centroid:** Determines the "brightness" or "darkness" of sounds.
    *   **Spectral Flux (Onset Strength):** Measures spectral change, effective for sharp transients.
*   **Profile-Driven Cue Logic:**
    *   **Selectable Profiles:** Choose between `balanced`, `noisy`, or `quiet` profiles (via CLI argument) to adjust the analysis sensitivity. Each profile has pre-defined dynamic threshold configurations.
    *   **Dynamic Thresholds:** Thresholds for triggering cues are not fixed but are dynamically calculated based on the statistical profile of the audio's own features (RMS, spectral centroid, spectral flux).
    *   **Sensitivity Boost:** For quieter sections, a sensitivity boost can be temporarily activated to catch more subtle events, then deactivated when significant events occur. This is configurable per profile.
    *   **Cue Generation Rules:** A sophisticated set of rules maps specific sonic events (combinations of feature values at onsets or beats) to firework types. For example:
        *   A strong beat with high spectral centroid might trigger a `high_sparkle`.
        *   An intense onset with low spectral centroid could become a `bass_pulse`.
        *   A sharp transient with high spectral flux might be a `comet_tail`.
    *   **Cooldown Mechanisms:** Configurable per profile, `min_time_between_same_type_cues` and `min_time_between_any_particle_cues` prevent cues from firing too rapidly, ensuring a more visually pleasing and realistic show.
*   **Defined Firework Types:** The script maps detected audio events to one of the following firework types:
    *   `mid_burst`: General energy, often on beats with moderate frequency content.
    *   `high_sparkle`: Bright, high-frequency events, typically on energetic beats.
    *   `kick_boom`: Strong, impactful onsets, like kick drums.
    *   `bass_pulse`: Deep, resonant bass hits identified by onsets with low spectral content.
    *   `comet_tail`: Sharp, quick accents with significant spectral change.

---

## ✨ Web Visualizer (`hanabiVisualizer/`)

Bring your generated cues to life! The web visualizer (open `hanabiVisualizer/index.html` in a browser) offers:

*   **Easy File Loading:** Pick your original audio file and the `_cues_<profile_name>.json` file generated by the Python script.
*   **Synchronized Playback:** Audio playback is tightly synced with firework animations.
*   **Dynamic Firework Animations:**
    *   Unique particle effects, colors, and behaviors for each firework type.
    *   A special background pulsing effect for `bass_pulse` cues.
    *   Fireworks are rendered on an HTML5 Canvas.
*   **Playback Controls:** Play, pause, stop, and a seekable progress bar.
*   **Real-time Log:** See which cues are being triggered and when.

---

## 🔧 Getting Started / How to Use

### 1. Generate Cues (Python)

*   **Prerequisites:**
    *   Python 3.x
    *   Pip (Python package installer)
    *   `ffmpeg` (often required by Librosa for MP3 and other compressed audio formats – ensure it's installed and in your system's PATH).
*   **Setup:**
    1.  Clone this repository: `git clone https://github.com/your-username/HanabiSync.git` (Replace with your actual repo URL if forked)
    2.  Navigate to the project directory: `cd HanabiSync`
    3.  Install Python dependencies: `pip install librosa numpy`
*   **Run the Analyzer:**
    *   To analyze a specific audio file with a chosen profile (e.g., `noisy`):
        ```bash
        python src/hanabi.py -f path/to/your/audio.mp3 --profile noisy
        ```
    *   Available profiles: `balanced` (default), `noisy`, `quiet`.
    *   If no file is specified with `-f`, the script will attempt to find the first compatible audio file in an `audio/` directory in the project root (create `audio/` if it doesn't exist, and place your audio file there).
        ```bash
        python src/hanabi.py --profile balanced 
        ```
*   **Output:**
    *   A JSON cue file (e.g., `my_song_cues_noisy.json`) will be saved in an `output/` directory in the project root. This file contains the generated cues and basic audio metadata.
    *   The raw, extensive audio feature arrays (like full RMS, spectral centroid frame data) are **not** included in this output JSON to keep file sizes manageable. They are used internally during analysis.
    *   Detailed processing logs will be printed to the console.

### 2. Visualize Your Show (Web)

1.  Open the `hanabiVisualizer/index.html` file in a modern web browser (e.g., Chrome, Firefox, Edge).
2.  Using the file input controls on the page:
    *   Load your original audio file.
    *   Load the `_cues_<profile_name>.json` file generated in the previous step (e.g., `my_song_cues_noisy.json`).
3.  Once both files are loaded, the "Play" button will become active.
4.  Click "Play" and enjoy your music-synchronized firework display!

---

## 🎆 Firework Types & Visuals (Default in Web Visualizer)

The Python script identifies distinct audio events and maps them to firework types. The web visualizer then renders them with default effects:

*   **`mid_burst`**:
    *   **Audio Trigger:** Typically associated with general energy on musical beats with mid-range frequency content.
    *   **Visuals:** Bursts of red/orange particles with a moderate lifespan.
*   **`high_sparkle`**:
    *   **Audio Trigger:** Caused by energetic beats characterized by bright, high-frequency sounds.
    *   **Visuals:** Cascading gold/yellow sparkling particles with a longer duration.
*   **`kick_boom`**:
    *   **Audio Trigger:** Fired by strong, percussive onsets like kick drums or impactful rhythmic hits.
    *   **Visuals:** Intense purple/violet particle explosions, short but impactful.
*   **`bass_pulse`**:
    *   **Audio Trigger:** Triggered by deep bass frequencies detected during onsets.
    *   **Visuals:** A rhythmic pulse of deep color in the background of the visualizer (e.g., deep purples, blues).
*   **`comet_tail`**:
    *   **Audio Trigger:** Generated by sharp, quick accents in the music that show a rapid change in spectral content (high spectral flux).
    *   **Visuals:** Streaking light blue/white particles with visible trails, simulating a comet.

*Note: The visualizer's legend in `index.html` provides a quick reference for these types as well.*

---

## 💡 Ideas & Roadmap

*   [✔️] Web version using JavaScript + WebAudio API for visualization.
*   [✔️] Profile-based parameter tuning (via CLI args for Python script).
*   [ ] Advanced Timeline Editor/GUI: A web-based or desktop interface for manually adjusting, adding, or removing cues generated by the Python script.
*   [ ] Export cue maps in multiple formats (e.g., CSV, FSEQ for xLights/Vixen).
*   [ ] More advanced parameter tuning for Python script (e.g., via a dedicated config file).
*   [ ] More customization options for web visualizer (e.g., user-defined colors, particle effects, background choices).
