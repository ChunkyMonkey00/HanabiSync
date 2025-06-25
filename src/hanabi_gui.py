# hanabi_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import json
import os
import threading
import time
import math
import random
from datetime import datetime

class FireworkParticle:
    def __init__(self, x, y, firework_type, canvas_width, canvas_height):
        self.x = x
        self.y = y
        self.firework_type = firework_type
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.age = 0
        self.max_age = self.get_max_age()
        self.particles = []
        self.create_particles()
        
    def get_max_age(self):
        if self.firework_type == "sparkle":
            return 2.0  # Sparkles last longer
        elif self.firework_type == "burst":
            return 1.5  # Bursts are medium duration
        else:  # boom
            return 1.0  # Booms are quick
    
    def get_color(self):
        if self.firework_type == "sparkle":
            return ["#FFD700", "#FFFF00", "#FFA500"]  # Gold, Yellow, Orange
        elif self.firework_type == "burst":
            return ["#FF0000", "#FF4500", "#FF6347"]  # Red, OrangeRed, Tomato
        else:  # boom
            return ["#8A2BE2", "#9400D3", "#4B0082"]  # Purple, Violet, Indigo
    
    def create_particles(self):
        colors = self.get_color()
        if self.firework_type == "sparkle":
            # Many small sparkly particles
            num_particles = random.randint(15, 25)
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(20, 60)
                self.particles.append({
                    'x': self.x,
                    'y': self.y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'color': random.choice(colors),
                    'size': random.randint(2, 4)
                })
        elif self.firework_type == "burst":
            # Medium explosion with outward burst
            num_particles = random.randint(20, 35)
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(40, 80)
                self.particles.append({
                    'x': self.x,
                    'y': self.y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'color': random.choice(colors),
                    'size': random.randint(3, 6)
                })
        else:  # boom
            # Large powerful explosion
            num_particles = random.randint(30, 50)
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(60, 120)
                self.particles.append({
                    'x': self.x,
                    'y': self.y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'color': random.choice(colors),
                    'size': random.randint(4, 8)
                })
    
    def update(self, dt):
        self.age += dt
        gravity = 100  # Pixels per second squared
        
        for particle in self.particles:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['vy'] += gravity * dt  # Apply gravity
            
            # Slow down particles over time
            particle['vx'] *= 0.98
            particle['vy'] *= 0.98
    
    def draw(self, canvas):
        if self.age >= self.max_age:
            return False
        
        # Calculate opacity based on age
        opacity_factor = 1.0 - (self.age / self.max_age)
        
        for particle in self.particles:
            # Don't draw particles that are off-screen
            if (particle['x'] < -10 or particle['x'] > self.canvas_width + 10 or
                particle['y'] < -10 or particle['y'] > self.canvas_height + 10):
                continue
                
            # Calculate size based on age (particles shrink over time)
            current_size = max(1, int(particle['size'] * opacity_factor))
            
            # Create circle with opacity effect
            x, y = int(particle['x']), int(particle['y'])
            
            # Draw particle as a circle
            canvas.create_oval(
                x - current_size, y - current_size,
                x + current_size, y + current_size,
                fill=particle['color'],
                outline="",
                tags="firework"
            )
        
        return True  # Still alive

class HanabiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HanabiSync - Firework Cue Visualizer")
        self.root.geometry("1000x700")
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Variables
        self.audio_file = None
        self.cue_file = None
        self.cues = []
        self.current_cue_index = 0
        self.is_playing = False
        self.start_time = None
        self.fireworks = []  # Active firework particles
        
        # Create GUI
        self.create_widgets()
        
        # Start animation loop
        self.animate()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Audio file selection
        ttk.Label(file_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.audio_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.audio_file_var, state="readonly").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_audio_file).grid(row=0, column=2)
        
        # Cue file selection
        ttk.Label(file_frame, text="Cue File:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.cue_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.cue_file_var, state="readonly").grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_cue_file).grid(row=1, column=2, pady=(5, 0))
        
        # Auto-load button
        ttk.Button(file_frame, text="Auto-Load from Output", command=self.auto_load_files).grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=tk.W, pady=(10, 0))
        
        # Playback controls
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_playback)
        self.play_button.grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(control_frame, text="Stop", command=self.stop_playback).grid(row=0, column=1, padx=(0, 5))
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.time_label = ttk.Label(progress_frame, text="00:00 / 00:00")
        self.time_label.grid(row=1, column=0)
        
        # Canvas for fireworks
        canvas_frame = ttk.LabelFrame(main_frame, text="Firework Display", padding="5")
        canvas_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg="black", height=400)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Cue Information", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Add legend
        self.add_legend()
        
    def add_legend(self):
        legend_text = """
Firework Types:
• Sparkle (High frequency): Gold/Yellow particles, longer duration
• Burst (Low-Mid frequency): Red/Orange particles, medium duration  
• Boom (Onsets): Purple/Violet particles, short but intense

Instructions:
1. Use 'Auto-Load from Output' to load the most recent cue file and its audio
2. Or manually browse for audio and cue files
3. Click Play to start the synchronized firework display
        """
        self.info_text.insert(tk.END, legend_text.strip())
        self.info_text.config(state=tk.DISABLED)
    
    def browse_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.audio_file = filename
            self.audio_file_var.set(os.path.basename(filename))
    
    def browse_cue_file(self):
        filename = filedialog.askopenfilename(
            title="Select Cue File",
            filetypes=[
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.load_cue_file(filename)
    
    def auto_load_files(self):
        """Auto-load the most recent cue file from ../output and find its corresponding audio file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "output")
        audio_dir = os.path.join(script_dir, "..", "audio")
        
        if not os.path.exists(output_dir):
            messagebox.showerror("Error", f"Output directory not found: {output_dir}")
            return
        
        # Find the most recent cue file
        cue_files = [f for f in os.listdir(output_dir) if f.endswith('_cues.json')]
        if not cue_files:
            messagebox.showerror("Error", f"No cue files found in {output_dir}")
            return
        
        # Sort by modification time, get the most recent
        cue_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        latest_cue_file = os.path.join(output_dir, cue_files[0])
        
        # Load the cue file
        if not self.load_cue_file(latest_cue_file):
            return
        
        # Find corresponding audio file
        base_name = cue_files[0].replace('_cues.json', '')
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac']
        
        audio_file_found = None
        for ext in audio_extensions:
            potential_audio = os.path.join(audio_dir, base_name + ext)
            if os.path.exists(potential_audio):
                audio_file_found = potential_audio
                break
        
        if audio_file_found:
            self.audio_file = audio_file_found
            self.audio_file_var.set(os.path.basename(audio_file_found))
            self.update_info(f"Auto-loaded: {os.path.basename(latest_cue_file)} with {os.path.basename(audio_file_found)}")
        else:
            self.update_info(f"Loaded cue file: {os.path.basename(latest_cue_file)}, but no matching audio file found in {audio_dir}")
            messagebox.showwarning("Warning", f"Cue file loaded, but no matching audio file found.\nPlease manually select the audio file.")
    
    def load_cue_file(self, filename):
        try:
            with open(filename, 'r') as f:
                self.cues = json.load(f)
            self.cue_file = filename
            self.cue_file_var.set(os.path.basename(filename))
            self.current_cue_index = 0
            self.update_info(f"Loaded {len(self.cues)} cues from {os.path.basename(filename)}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cue file: {str(e)}")
            return False
    
    def update_info(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def toggle_playback(self):
        if not self.audio_file or not self.cues:
            messagebox.showerror("Error", "Please select both audio and cue files first.")
            return
        
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        try:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            self.is_playing = True
            self.start_time = time.time()
            self.current_cue_index = 0
            self.play_button.config(text="Pause")
            self.update_info("Playback started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start playback: {str(e)}")
    
    def pause_playback(self):
        pygame.mixer.music.pause()
        self.is_playing = False
        self.play_button.config(text="Resume")
        self.update_info("Playback paused")
    
    def stop_playback(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.start_time = None
        self.current_cue_index = 0
        self.fireworks.clear()
        self.play_button.config(text="Play")
        self.progress_var.set(0)
        self.time_label.config(text="00:00 / 00:00")
        self.canvas.delete("firework")
        self.update_info("Playback stopped")
    
    def animate(self):
        current_time = time.time()
        dt = 1/60  # 60 FPS
        
        if self.is_playing and self.start_time:
            elapsed_time = current_time - self.start_time
            
            # Check for new cues to trigger
            while (self.current_cue_index < len(self.cues) and 
                   self.cues[self.current_cue_index]["timestamp"] <= elapsed_time):
                cue = self.cues[self.current_cue_index]
                self.trigger_firework(cue["firework"])
                self.update_info(f"Triggered {cue['firework']} at {cue['timestamp']:.2f}s")
                self.current_cue_index += 1
            
            # Update progress
            if self.cues:
                max_time = max(cue["timestamp"] for cue in self.cues) + 5  # Add 5 seconds buffer
                progress = min(100, (elapsed_time / max_time) * 100)
                self.progress_var.set(progress)
                
                # Update time display
                elapsed_str = f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                total_str = f"{int(max_time//60):02d}:{int(max_time%60):02d}"
                self.time_label.config(text=f"{elapsed_str} / {total_str}")
        
        # Update and draw fireworks
        self.canvas.delete("firework")
        self.fireworks = [fw for fw in self.fireworks if fw.update(dt)]
        
        for firework in self.fireworks:
            firework.draw(self.canvas)
        
        # Schedule next frame
        self.root.after(16, self.animate)  # ~60 FPS
    
    def trigger_firework(self, firework_type):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return  # Canvas not ready yet
        
        # Random position for firework
        x = random.randint(50, canvas_width - 50)
        y = random.randint(50, canvas_height - 50)
        
        # Create new firework
        firework = FireworkParticle(x, y, firework_type, canvas_width, canvas_height)
        self.fireworks.append(firework)

def main():
    root = tk.Tk()
    app = HanabiGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()