/* ============================================================
   DeepVision — Landing Page Interactions
   No backend logic — pure UI behavior
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
    // ----- Scroll-Reveal Animation -----
    const revealElements = document.querySelectorAll(
        '.card, .models__instruction, .compare-wrap, .quote-section__inner'
    );
    revealElements.forEach(el => el.classList.add('reveal'));
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target); // animate once
                }
            });
        },
        { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    );
    revealElements.forEach(el => observer.observe(el));
    // ----- Scroll Dot Navigation -----
    const dots = document.querySelectorAll('.dot');
    const sections = ['hero', 'models', 'quote'];
    dots.forEach(dot => {
        dot.addEventListener('click', () => {
            const target = document.getElementById(dot.dataset.section);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    // Update active dot on scroll
    const sectionEls = sections.map(id => document.getElementById(id));
    const updateDots = () => {
        const scrollY = window.scrollY + window.innerHeight / 3;
        let activeIndex = 0;
        sectionEls.forEach((sec, i) => {
            if (sec && sec.offsetTop <= scrollY) {
                activeIndex = i;
            }
        });
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === activeIndex);
        });
    };
    window.addEventListener('scroll', updateDots, { passive: true });
    updateDots();
    // ----- Card Hover Parallax (subtle) -----
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -2;
            const rotateY = ((x - centerX) / centerX) * 2;
            card.style.transform = `translateY(-8px) scale(1.05) perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
    // ----- Button Click Ripple Effect -----
    const allButtons = document.querySelectorAll('.card__btn, .compare-btn');
    allButtons.forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            Object.assign(ripple.style, {
                position: 'absolute',
                width: `${size}px`,
                height: `${size}px`,
                left: `${x}px`,
                top: `${y}px`,
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.15)',
                transform: 'scale(0)',
                animation: 'ripple 0.6s ease-out forwards',
                pointerEvents: 'none',
                zIndex: '0',
            });
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            ripple.addEventListener('animationend', () => ripple.remove());
        });
    });
    // Add ripple keyframe dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    // ----- Smooth hero scroll-hint click -----
    const scrollHint = document.querySelector('.hero__scroll-hint');
    if (scrollHint) {
        scrollHint.style.cursor = 'pointer';
        scrollHint.addEventListener('click', () => {
            document.getElementById('models').scrollIntoView({ behavior: 'smooth' });
        });
    }
    // =========================================================
    //  CARD CANVAS ANIMATIONS — always running
    // =========================================================
    const animationFrames = {};
    function setupCanvas(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        function resize() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
        }
        resize();
        window.addEventListener('resize', resize);
        return { canvas, ctx, dpr, getW: () => canvas.width / dpr, getH: () => canvas.height / dpr };
    }
    // ---- CNN: Convolution grid scan ----
    // Animated kernel sliding across a grid, highlighting cells
    function initCNN() {
        const setup = setupCanvas('canvas-cnn');
        if (!setup) return;
        const { ctx, getW, getH } = setup;
        let t = 0;
        function draw() {
            const w = getW(), h = getH();
            ctx.clearRect(0, 0, w, h);
            const cols = 10, rows = 8;
            const cellW = w / cols, cellH = h / rows;
            // Draw grid cells
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x = c * cellW, y = r * cellH;
                    ctx.strokeStyle = 'rgba(99, 102, 241, 0.08)';
                    ctx.lineWidth = 0.5;
                    ctx.strokeRect(x, y, cellW, cellH);
                }
            }
            // Animated 3x3 kernel position
            const kernelSize = 3;
            const totalSteps = (cols - kernelSize + 1) * (rows - kernelSize + 1);
            const speed = 0.04;
            const step = Math.floor(t * speed) % totalSteps;
            const kernelCols = cols - kernelSize + 1;
            const kc = step % kernelCols;
            const kr = Math.floor(step / kernelCols);
            // Highlight kernel cells with glow
            for (let dr = 0; dr < kernelSize; dr++) {
                for (let dc = 0; dc < kernelSize; dc++) {
                    const x = (kc + dc) * cellW;
                    const y = (kr + dr) * cellH;
                    // Glow effect
                    const gradient = ctx.createRadialGradient(
                        x + cellW/2, y + cellH/2, 0,
                        x + cellW/2, y + cellH/2, cellW
                    );
                    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.35)');
                    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
                    ctx.fillStyle = gradient;
                    ctx.fillRect(x - cellW/2, y - cellH/2, cellW * 2, cellH * 2);
                    // Cell fill
                    ctx.fillStyle = 'rgba(99, 102, 241, 0.2)';
                    ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
                    // Cell border
                    ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(x, y, cellW, cellH);
                }
            }
            // Trail: previously scanned cells fade out
            const trailLength = 8;
            for (let i = 1; i <= trailLength; i++) {
                const prevStep = (step - i + totalSteps) % totalSteps;
                const pc = prevStep % kernelCols;
                const pr = Math.floor(prevStep / kernelCols);
                const alpha = 0.08 * (1 - i / trailLength);
                for (let dr = 0; dr < kernelSize; dr++) {
                    for (let dc = 0; dc < kernelSize; dc++) {
                        const x = (pc + dc) * cellW;
                        const y = (pr + dr) * cellH;
                        ctx.fillStyle = `rgba(99, 102, 241, ${alpha})`;
                        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
                    }
                }
            }
            t++;
            animationFrames['cnn'] = requestAnimationFrame(draw);
        }
        draw();
    }
    // ---- ANN: Neural network pulses ----
    // Nodes in layers with animated signal pulses traveling through connections
    function initANN() {
        const setup = setupCanvas('canvas-ann');
        if (!setup) return;
        const { ctx, getW, getH } = setup;
        let t = 0;
        const layers = [3, 5, 4, 2]; // neuron counts per layer
        let pulses = [];
        function spawnPulse() {
            const fromLayer = Math.floor(Math.random() * (layers.length - 1));
            const fromNode = Math.floor(Math.random() * layers[fromLayer]);
            const toNode = Math.floor(Math.random() * layers[fromLayer + 1]);
            pulses.push({
                fromLayer, fromNode, toLayer: fromLayer + 1, toNode,
                progress: 0,
                speed: 0.015 + Math.random() * 0.015,
                color: Math.random() > 0.5 ? '110, 231, 183' : '99, 102, 241'
            });
        }
        function getNodePos(layerIdx, nodeIdx, w, h) {
            const padding = 40;
            const layerX = padding + (layerIdx / (layers.length - 1)) * (w - padding * 2);
            const count = layers[layerIdx];
            const spacing = (h - padding * 2) / (count + 1);
            const nodeY = padding + spacing * (nodeIdx + 1);
            return { x: layerX, y: nodeY };
        }
        function draw() {
            const w = getW(), h = getH();
            ctx.clearRect(0, 0, w, h);
            // Draw connections (faint lines)
            for (let l = 0; l < layers.length - 1; l++) {
                for (let i = 0; i < layers[l]; i++) {
                    for (let j = 0; j < layers[l + 1]; j++) {
                        const from = getNodePos(l, i, w, h);
                        const to = getNodePos(l + 1, j, w, h);
                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }
                }
            }
            // Draw nodes
            for (let l = 0; l < layers.length; l++) {
                for (let i = 0; i < layers[l]; i++) {
                    const pos = getNodePos(l, i, w, h);
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
            // Spawn new pulses randomly
            if (Math.random() < 0.15) spawnPulse();
            // Draw and update pulses
            pulses = pulses.filter(p => p.progress <= 1);
            for (const p of pulses) {
                const from = getNodePos(p.fromLayer, p.fromNode, w, h);
                const to = getNodePos(p.toLayer, p.toNode, w, h);
                const x = from.x + (to.x - from.x) * p.progress;
                const y = from.y + (to.y - from.y) * p.progress;
                // Glow trail
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 16);
                gradient.addColorStop(0, `rgba(${p.color}, 0.6)`);
                gradient.addColorStop(1, `rgba(${p.color}, 0)`);
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, 16, 0, Math.PI * 2);
                ctx.fill();
                // Bright dot
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${p.color}, 0.9)`;
                ctx.fill();
                // Highlight connection line
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(x, y);
                ctx.strokeStyle = `rgba(${p.color}, 0.25)`;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                p.progress += p.speed;
            }
            t++;
            animationFrames['ann'] = requestAnimationFrame(draw);
        }
        draw();
    }
    // ---- GAN: Generator vs Discriminator ----
    // Two opposing wave/particle systems competing — left (Generator, purple) vs right (Discriminator, teal)
    function initGAN() {
        const setup = setupCanvas('canvas-gan');
        if (!setup) return;
        const { ctx, getW, getH } = setup;
        let t = 0;
        const particles = [];
        const PARTICLE_COUNT = 40;
        function initParticles(w, h) {
            particles.length = 0;
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const isGenerator = i < PARTICLE_COUNT / 2;
                particles.push({
                    x: isGenerator ? Math.random() * w * 0.4 : w * 0.6 + Math.random() * w * 0.4,
                    y: Math.random() * h,
                    vx: (isGenerator ? 1 : -1) * (0.3 + Math.random() * 0.7),
                    vy: (Math.random() - 0.5) * 0.8,
                    size: 2 + Math.random() * 2,
                    isGenerator,
                    life: 1,
                });
            }
        }
        function draw() {
            const w = getW(), h = getH();
            ctx.clearRect(0, 0, w, h);
            // Center divider — the battlefront
            const centerX = w / 2;
            const waveAmplitude = 15;
            ctx.beginPath();
            ctx.moveTo(centerX, 0);
            for (let y = 0; y < h; y += 4) {
                const offset = Math.sin(y * 0.03 + t * 0.05) * waveAmplitude;
                ctx.lineTo(centerX + offset, y);
            }
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
            // Labels
            ctx.font = '600 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(168, 85, 247, 0.35)';
            ctx.fillText('Generator', w * 0.22, 20);
            ctx.fillStyle = 'rgba(20, 184, 166, 0.35)';
            ctx.fillText('Discriminator', w * 0.78, 20);
            // Draw and update particles
            for (const p of particles) {
                const color = p.isGenerator ? '168, 85, 247' : '20, 184, 166';
                // Glow
                const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 4);
                gradient.addColorStop(0, `rgba(${color}, ${0.3 * p.life})`);
                gradient.addColorStop(1, `rgba(${color}, 0)`);
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size * 4, 0, Math.PI * 2);
                ctx.fill();
                // Core dot
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${color}, ${0.7 * p.life})`;
                ctx.fill();
                // Move towards center (competing)
                p.x += p.vx;
                p.y += p.vy;
                p.vy += (Math.random() - 0.5) * 0.1;
                // Bounce off edges
                if (p.y < 0 || p.y > h) p.vy *= -1;
                // When particle crosses center, respawn
                if ((p.isGenerator && p.x > centerX + 20) || (!p.isGenerator && p.x < centerX - 20)) {
                    // Flash at collision point
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 255, 255, 0.3)`;
                    ctx.fill();
                    // Respawn
                    p.x = p.isGenerator ? Math.random() * w * 0.3 : w * 0.7 + Math.random() * w * 0.3;
                    p.y = Math.random() * h;
                    p.vy = (Math.random() - 0.5) * 0.8;
                }
            }
            t++;
            animationFrames['gan'] = requestAnimationFrame(draw);
        }
        initParticles(getW(), getH());
        draw();
    }
    // ---- YOLO: Object detection boxes ----
    // Animated bounding boxes appearing, scanning, and locking onto "targets"
    function initYOLO() {
        const setup = setupCanvas('canvas-yolo');
        if (!setup) return;
        const { ctx, getW, getH } = setup;
        let t = 0;
        let boxes = [];
        function spawnBox(w, h) {
            const bw = 30 + Math.random() * 60;
            const bh = 30 + Math.random() * 50;
            boxes.push({
                x: 20 + Math.random() * (w - bw - 40),
                y: 20 + Math.random() * (h - bh - 40),
                w: bw,
                h: bh,
                phase: 0,       // 0: scanning, 1: locking, 2: locked, 3: fade
                timer: 0,
                confidence: (0.7 + Math.random() * 0.29).toFixed(2),
                label: ['person', 'car', 'dog', 'cat', 'bird', 'phone'][Math.floor(Math.random() * 6)],
                color: [
                    '34, 197, 94',   // green
                    '250, 204, 21',  // yellow
                    '59, 130, 246',  // blue
                    '239, 68, 68',   // red
                ][Math.floor(Math.random() * 4)]
            });
        }
        function draw() {
            const w = getW(), h = getH();
            ctx.clearRect(0, 0, w, h);
            // Faint grid (YOLO's detection grid)
            const gridCols = 7, gridRows = 5;
            const cellW = w / gridCols, cellH = h / gridRows;
            for (let r = 0; r <= gridRows; r++) {
                ctx.beginPath();
                ctx.moveTo(0, r * cellH);
                ctx.lineTo(w, r * cellH);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
            for (let c = 0; c <= gridCols; c++) {
                ctx.beginPath();
                ctx.moveTo(c * cellW, 0);
                ctx.lineTo(c * cellW, h);
                ctx.stroke();
            }
            // Scanning line sweeping horizontally
            const scanX = (t * 2) % w;
            ctx.beginPath();
            ctx.moveTo(scanX, 0);
            ctx.lineTo(scanX, h);
            ctx.strokeStyle = 'rgba(34, 197, 94, 0.15)';
            ctx.lineWidth = 2;
            ctx.stroke();
            // Glow on scan line
            const scanGradient = ctx.createLinearGradient(scanX - 30, 0, scanX + 30, 0);
            scanGradient.addColorStop(0, 'rgba(34, 197, 94, 0)');
            scanGradient.addColorStop(0.5, 'rgba(34, 197, 94, 0.06)');
            scanGradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
            ctx.fillStyle = scanGradient;
            ctx.fillRect(scanX - 30, 0, 60, h);
            // Spawn boxes periodically
            if (t % 50 === 0 && boxes.length < 4) {
                spawnBox(w, h);
            }
            // Draw boxes
            boxes = boxes.filter(b => b.phase < 4);
            for (const b of boxes) {
                b.timer++;
                const alpha = b.phase === 3 ? Math.max(0, 1 - b.timer / 30) : 1;
                if (b.phase === 0) {
                    // Scanning: dashed box expanding
                    const scale = Math.min(1, b.timer / 20);
                    const cx = b.x + b.w / 2;
                    const cy = b.y + b.h / 2;
                    const sw = b.w * scale;
                    const sh = b.h * scale;
                    ctx.setLineDash([4, 4]);
                    ctx.strokeStyle = `rgba(${b.color}, 0.5)`;
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(cx - sw/2, cy - sh/2, sw, sh);
                    ctx.setLineDash([]);
                    if (b.timer > 25) { b.phase = 1; b.timer = 0; }
                }
                else if (b.phase === 1) {
                    // Locking: corners drawing in
                    const cornerLen = 8;
                    ctx.strokeStyle = `rgba(${b.color}, ${0.8 * alpha})`;
                    ctx.lineWidth = 2;
                    // Top-left
                    ctx.beginPath();
                    ctx.moveTo(b.x, b.y + cornerLen); ctx.lineTo(b.x, b.y); ctx.lineTo(b.x + cornerLen, b.y);
                    ctx.stroke();
                    // Top-right
                    ctx.beginPath();
                    ctx.moveTo(b.x + b.w - cornerLen, b.y); ctx.lineTo(b.x + b.w, b.y); ctx.lineTo(b.x + b.w, b.y + cornerLen);
                    ctx.stroke();
                    // Bottom-left
                    ctx.beginPath();
                    ctx.moveTo(b.x, b.y + b.h - cornerLen); ctx.lineTo(b.x, b.y + b.h); ctx.lineTo(b.x + cornerLen, b.y + b.h);
                    ctx.stroke();
                    // Bottom-right
                    ctx.beginPath();
                    ctx.moveTo(b.x + b.w - cornerLen, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h); ctx.lineTo(b.x + b.w, b.y + b.h - cornerLen);
                    ctx.stroke();
                    // Fill
                    ctx.fillStyle = `rgba(${b.color}, 0.04)`;
                    ctx.fillRect(b.x, b.y, b.w, b.h);
                    if (b.timer > 15) { b.phase = 2; b.timer = 0; }
                }
                else if (b.phase === 2) {
                    // Locked: solid box with label
                    ctx.strokeStyle = `rgba(${b.color}, ${0.7 * alpha})`;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(b.x, b.y, b.w, b.h);
                    ctx.fillStyle = `rgba(${b.color}, 0.06)`;
                    ctx.fillRect(b.x, b.y, b.w, b.h);
                    // Label background
                    const labelText = `${b.label} ${b.confidence}`;
                    ctx.font = '600 9px Inter, sans-serif';
                    const textWidth = ctx.measureText(labelText).width;
                    ctx.fillStyle = `rgba(${b.color}, 0.8)`;
                    ctx.fillRect(b.x, b.y - 14, textWidth + 8, 14);
                    // Label text
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                    ctx.fillText(labelText, b.x + 4, b.y - 4);
                    if (b.timer > 80) { b.phase = 3; b.timer = 0; }
                }
                else if (b.phase === 3) {
                    // Fading out
                    ctx.strokeStyle = `rgba(${b.color}, ${0.5 * alpha})`;
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(b.x, b.y, b.w, b.h);
                    if (b.timer > 30) b.phase = 4;
                }
            }
            t++;
            animationFrames['yolo'] = requestAnimationFrame(draw);
        }
        draw();
    }
    // Initialize all card animations
    initCNN();
    initANN();
    initGAN();
    initYOLO();
});