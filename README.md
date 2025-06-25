# HanabiSync 🎆🔊

**Turn your music into fireworks.**  
HanabiSync is an open source audio analysis tool that generates synchronized firework cue maps from any song or audio track.

Perfect for backyard firework shows, synced holiday displays, or just nerding out with music-reactive explosions.

---

## 🚀 What It Does

HanabiSync analyzes an audio file using FFT, beat tracking, and amplitude detection to create a timestamped cue map for firework effects.

It outputs a simple JSON file like:
```json
[
  { "timestamp": 2.5, "firework": "burst" },
  { "timestamp": 4.1, "firework": "sparkle" }
]
````

This cue file can be fed into a microcontroller, Raspberry Pi, or custom control system to launch fireworks in sync with music.

---

## 🎧 How It Works

* **FFT & Frequency Bands** – Detect dominant frequencies to classify effects (e.g., bass = boom).
* **RMS & Onset Detection** – Find moments of intensity and musical change.
* **Beat Tracking** – Identify tempo and rhythmic hits.
* **Cue Logic** – Map sound events to firework types based on timing and frequency patterns.

---

## 💡 Ideas & Roadmap

* [ ] Add GUI timeline visualizer
* [ ] Export cue maps in multiple formats (CSV, MIDI?)
* [ ] Web version using JavaScript + WebAudio API
* [ ] Real-time audio reaction mode
* [ ] Integration with DMX or relay hardware

---
