document.addEventListener('DOMContentLoaded', () => {
    const audioFileInput = document.getElementById('audioFile');
    const cueFileInput = document.getElementById('cueFile');
    const fileStatusDisplay = document.getElementById('file-status');
    const playPauseButton = document.getElementById('playPauseButton');
    const stopButton = document.getElementById('stopButton');
    const progressBar = document.getElementById('progressBar');
    const timeDisplay = document.getElementById('timeDisplay');
    const canvas = document.getElementById('fireworkCanvas');
    const logOutput = document.getElementById('logOutput');
    const ctx = canvas.getContext('2d');

    let audioContext;
    let audioBuffer;
    let audioSourceNode;
    let cues = [];
    let isPlaying = false;
    let startTime = 0;
    let playbackOffset = 0;
    let currentCueIndex = 0;
    let totalAudioDuration = 0;
    let animationFrameId;

    let fireworks = [];
    let backgroundPulse = {
        alpha: 0,
        maxAlpha: 0.5,
        color: 'rgba(75, 0, 130, 1)',
        fadeSpeed: 1.5
    };
    const BASS_PULSE_COLORS = [
        'rgba(75, 0, 130, 1)',
        'rgba(0, 0, 139, 1)',
        'rgba(139, 0, 139, 1)',
        'rgba(40, 40, 100, 1)'
    ];


    function setupCanvas() {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        window.addEventListener('resize', () => {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        });
    }
    setupCanvas();

    function logMessage(message) {
        const timestamp = new Date().toLocaleTimeString();
        logOutput.textContent += `[${timestamp}] ${message}\n`;
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    class FireworkParticleSystem {
    constructor(x, y, fireworkType, canvasWidth, canvasHeight) {
        this.x = x;
        this.y = y;
        this.fireworkType = fireworkType; 
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.age = 0;
        this.maxAge = this.getMaxAge();
        this.particles = [];
        this.createParticles();
    }

    getMaxAge() {
        if (this.fireworkType === "high_sparkle") return 2.0;
        if (this.fireworkType === "mid_burst") return 1.5;
        if (this.fireworkType === "kick_boom") return 1.2;
        if (this.fireworkType === "comet_tail") return 1.8;
        return 1.0; 
    }

    getColors() {
        if (this.fireworkType === "high_sparkle") return ["#FFD700", "#FFFF00", "#FFA500"]; 
        if (this.fireworkType === "mid_burst") return ["#FF0000", "#FF4500", "#FF6347"];   
        if (this.fireworkType === "kick_boom") return ["#8A2BE2", "#9400D3", "#4B0082", "#BA55D3"];
        if (this.fireworkType === "comet_tail") return ["#ADD8E6", "#B0E0E6", "#AFEEEE", "#FFFFFF"];
        return ["#FFFFFF", "#CCCCCC"]; 
    }

    createParticles() {
        const colors = this.getColors();
        let numParticles, speedRange, sizeRange, angleSpread = Math.PI * 2;

        if (this.fireworkType === "high_sparkle") {
            numParticles = Math.floor(Math.random() * 11) + 20; 
            speedRange = [20, 70];
            sizeRange = [1, 3]; 
        } else if (this.fireworkType === "mid_burst") {
            numParticles = Math.floor(Math.random() * 16) + 25;
            speedRange = [40, 90];
            sizeRange = [2, 5];
        } else if (this.fireworkType === "kick_boom") {
            numParticles = Math.floor(Math.random() * 21) + 35; 
            speedRange = [50, 130]; 
            sizeRange = [3, 7];
        } else if (this.fireworkType === "comet_tail") {
            numParticles = Math.floor(Math.random() * 8) + 10;
            speedRange = [100, 180];
            sizeRange = [2, 4];
        } else {
            logMessage(`Warning: Unknown firework type in createParticles: ${this.fireworkType}`);
            numParticles = 15;
            speedRange = [30, 70];
            sizeRange = [2, 4];
        }

        const baseAngle = (this.fireworkType === "comet_tail") ? (Math.random() * Math.PI * 2) : 0;

        for (let i = 0; i < numParticles; i++) {
            const angle = baseAngle + (Math.random() - 0.5) * angleSpread * 2;
            const speed = Math.random() * (speedRange[1] - speedRange[0]) + speedRange[0];
            this.particles.push({
                x: this.x,
                y: this.y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                color: colors[Math.floor(Math.random() * colors.length)],
                size: Math.floor(Math.random() * (sizeRange[1] - sizeRange[0] + 1)) + sizeRange[0],
                trail: [],
                isCometTailCore: (this.fireworkType === "comet_tail")
            });
        }
    }

    update(dt) {
        this.age += dt;
        if (this.age >= this.maxAge) return false;

        const gravity = this.fireworkType === "comet_tail" ? 30 : 98.1;
        const damping = this.fireworkType === "comet_tail" ? 0.99 : 0.98;

        for (const p of this.particles) {
            p.trail.push({ x: p.x, y: p.y, alpha: 1.0 });
            if (p.trail.length > (p.isCometTailCore ? 25 : 8)) {
                p.trail.shift();
            }
            for(let i = 0; i < p.trail.length; i++) {
                p.trail[i].alpha *= (p.isCometTailCore ? 0.92 : 0.85);
            }
            p.trail = p.trail.filter(segment => segment.alpha > 0.05);


            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.vy += gravity * dt;
            
            p.vx *= (1 - (1 - damping) * dt * 60); 
            p.vy *= (1 - (1 - damping) * dt * 60);
        }
        return true;
    }

    draw(ctx) {
        const overallOpacity = Math.max(0, 1.0 - (this.age / this.maxAge));
        
        ctx.save();

        for (const p of this.particles) {
            if (p.x < -10 || p.x > this.canvasWidth + 10 || p.y < -10 || p.y > this.canvasHeight + 10) {
                continue;
            }
            const particleProgress = this.age / this.maxAge;
            let currentSize = Math.max(0.5, p.size * (1 - particleProgress * (p.isCometTailCore ? 0.7 : 0.9)));
            if (currentSize < 0.5) continue;

            if (p.trail.length > 1) {
                ctx.beginPath();
                ctx.moveTo(p.trail[0].x, p.trail[0].y);
                for (let i = 1; i < p.trail.length; i++) {
                    ctx.lineTo(p.trail[i].x, p.trail[i].y);
                }
                const trailBaseAlpha = p.isCometTailCore ? 0.7 : 0.5;
                ctx.strokeStyle = p.color;
                 for (let i = p.trail.length - 1; i > 0; i--) {
                    ctx.beginPath();
                    ctx.moveTo(p.trail[i].x, p.trail[i].y);
                    ctx.lineTo(p.trail[i-1].x, p.trail[i-1].y);
                    ctx.lineWidth = Math.max(0.2, currentSize * (p.isCometTailCore ? 0.7 : 0.4) * (p.trail[i].alpha));
                    ctx.globalAlpha = overallOpacity * p.trail[i].alpha * trailBaseAlpha;
                    ctx.stroke();
                }
            }
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, currentSize, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = overallOpacity * (p.isCometTailCore ? 1.0 : 0.8 + 0.2 * Math.random());
            ctx.fill();
        }
        ctx.restore();
    }
}

    audioFileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        logMessage(`Loading audio file: ${file.name}`);
        try {
            if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await file.reader().readAsArrayBuffer();
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            totalAudioDuration = audioBuffer.duration;
            progressBar.max = totalAudioDuration;
            updateTimeDisplay();
            logMessage(`Audio loaded: ${file.name} (${formatTime(totalAudioDuration)}s)`);
            checkFilesReady();
        } catch (e) {
            logMessage(`Error loading audio: ${e.message}`);
            alert(`Error loading audio: ${e.message}`);
        }
    });
    
    if (File && !File.prototype.reader) {
        File.prototype.reader = function() {
            const file = this;
            return {
                readAsArrayBuffer: () => new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = () => reject(reader.error);
                    reader.readAsArrayBuffer(file);
                }),
                readAsText: () => new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = () => reject(reader.error);
                    reader.readAsText(file);
                })
            };
        };
    }

    cueFileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        logMessage(`Loading cue file: ${file.name}`);
        try {
            const text = await file.reader().readAsText();
            cues = JSON.parse(text);
            cues.sort((a, b) => a.timestamp - b.timestamp);
            logMessage(`Cues loaded: ${cues.length} cues from ${file.name}`);
            checkFilesReady();
        } catch (e) {
            logMessage(`Error loading or parsing cue file: ${e.message}`);
            alert(`Error loading or parsing cue file: ${e.message}`);
        }
    });

    function checkFilesReady() {
        if (audioBuffer && cues.length > 0) {
            playPauseButton.disabled = false;
            stopButton.disabled = false;
            progressBar.disabled = false;
            fileStatusDisplay.textContent = "Audio and cues loaded. Ready to play.";
            logMessage("Ready to play.");
        } else if (audioBuffer) {
            fileStatusDisplay.textContent = "Audio loaded. Waiting for cue file.";
        } else if (cues.length > 0) {
            fileStatusDisplay.textContent = "Cues loaded. Waiting for audio file.";
        }
    }

    playPauseButton.addEventListener('click', () => {
        if (!audioContext) return;
        if (isPlaying) pauseAudio();
        else playAudio();
    });

    stopButton.addEventListener('click', stopAudio);

    function playAudio() {
        if (isPlaying || !audioBuffer) return;
        if (audioContext.state === 'suspended') audioContext.resume();
        
        audioSourceNode = audioContext.createBufferSource();
        audioSourceNode.buffer = audioBuffer;
        audioSourceNode.connect(audioContext.destination);
        audioSourceNode.onended = () => {
            if (isPlaying) {
                isPlaying = false;
                playPauseButton.textContent = 'Play';
                updateProgressBar(totalAudioDuration);
                logMessage("Playback finished.");
            }
        };
        
        audioSourceNode.start(0, playbackOffset);
        startTime = audioContext.currentTime - playbackOffset;
        isPlaying = true;
        playPauseButton.textContent = 'Pause';
        logMessage(`Playback started (offset: ${playbackOffset.toFixed(2)}s).`);
        if (!animationFrameId) {
            lastAnimationTime = performance.now();
            animate();
        }
    }

    function pauseAudio() {
        if (!isPlaying || !audioSourceNode) return;
        audioSourceNode.stop();
        isPlaying = false; 
        playbackOffset = audioContext.currentTime - startTime;
        playPauseButton.textContent = 'Resume';
        logMessage(`Playback paused at ${playbackOffset.toFixed(2)}s.`);
    }

    function stopAudio() {
        if (audioSourceNode) {
            audioSourceNode.onended = null;
            try { audioSourceNode.stop(); } catch(e) { }
            audioSourceNode = null;
        }
        isPlaying = false;
        playbackOffset = 0;
        currentCueIndex = 0;
        startTime = 0;
        fireworks = [];
        backgroundPulse.alpha = 0;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        playPauseButton.textContent = 'Play';
        playPauseButton.disabled = !(audioBuffer && cues.length > 0);
        progressBar.disabled = !(audioBuffer && cues.length > 0);
        
        updateProgressBar(0);
        updateTimeDisplay();
        logMessage("Playback stopped and reset.");
    }

    let lastAnimationTime = 0;
    function animate(currentTime) {
        animationFrameId = requestAnimationFrame(animate);

        const dt = (currentTime - lastAnimationTime) / 1000;
        lastAnimationTime = currentTime;

        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (backgroundPulse.alpha > 0) {
            const colorParts = backgroundPulse.color.match(/\d+/g);
            if (colorParts && colorParts.length >= 3) {
                 ctx.fillStyle = `rgba(${colorParts[0]}, ${colorParts[1]}, ${colorParts[2]}, ${backgroundPulse.alpha})`;
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            backgroundPulse.alpha -= backgroundPulse.fadeSpeed * dt;
            if (backgroundPulse.alpha < 0) backgroundPulse.alpha = 0;
        }

        let currentAudioTime = playbackOffset;
        if (isPlaying) {
            currentAudioTime = audioContext.currentTime - startTime;
            if (currentAudioTime >= totalAudioDuration) {
                currentAudioTime = totalAudioDuration;
            }

            while (currentCueIndex < cues.length && cues[currentCueIndex].timestamp <= currentAudioTime) {
                const cue = cues[currentCueIndex];
                triggerFirework(cue.firework);
                logMessage(`Triggered ${cue.firework} at ${cue.timestamp.toFixed(2)}s (audio: ${currentAudioTime.toFixed(2)}s)`);
                currentCueIndex++;
            }
            updateProgressBar(currentAudioTime);
        }
        updateTimeDisplay(currentAudioTime);

        fireworks = fireworks.filter(fw => fw.update(dt));
        fireworks.forEach(fw => fw.draw(ctx));
    }

    function triggerFirework(fireworkType) {
        if (!canvas.width || !canvas.height) return;

        if (fireworkType === "bass_pulse") {
            backgroundPulse.alpha = backgroundPulse.maxAlpha;
            backgroundPulse.color = BASS_PULSE_COLORS[Math.floor(Math.random() * BASS_PULSE_COLORS.length)];
        } else {
            const x = Math.random() * (canvas.width * 0.8) + (canvas.width * 0.1);
            const y = Math.random() * (canvas.height * 0.6) + (canvas.height * 0.1);
            const fw = new FireworkParticleSystem(x, y, fireworkType, canvas.width, canvas.height);
            fireworks.push(fw);
        }
    }

    progressBar.addEventListener('input', () => {
        const newTime = progressBar.valueAsNumber;
        playbackOffset = newTime;
        currentCueIndex = 0;
        while(currentCueIndex < cues.length && cues[currentCueIndex].timestamp < playbackOffset) {
            currentCueIndex++;
        }
        
        if (isPlaying) {
            if (audioSourceNode) {
                audioSourceNode.onended = null;
                try { audioSourceNode.stop(); } catch(e) {}
            }
            playAudio();
        } else {
            updateTimeDisplay(playbackOffset);
        }
        logMessage(`Seeked to ${playbackOffset.toFixed(2)}s.`);
    });

    function formatTime(seconds) {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    }

    function updateProgressBar(currentTime) {
        if (totalAudioDuration > 0) progressBar.value = currentTime;
        else progressBar.value = 0;
    }

    function updateTimeDisplay(currentTime = playbackOffset) {
        timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(totalAudioDuration)}`;
    }

    logMessage("HanabiSync Web Visualizer initialized. Load files to start.");
    updateTimeDisplay();
    playPauseButton.disabled = true;
    stopButton.disabled = true;
    progressBar.disabled = true;
});