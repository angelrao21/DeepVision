/* ============================================================
   DeepVision — GAN Page Animations & Interactions
   ============================================================ */
(function () {
    'use strict';

    /* ---------- Utility ---------- */
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);
    const lerp = (a, b, t) => a + (b - a) * t;
    const rand = (min, max) => Math.random() * (max - min) + min;

    /* ============================================================
       1. HERO PARTICLES — floating neural-net style background
       ============================================================ */
    function initHeroParticles() {
        const canvas = $('#hero-particles-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let W, H, particles = [], mouse = { x: -9999, y: -9999 };

        function resize() {
            W = canvas.width = canvas.offsetWidth * devicePixelRatio;
            H = canvas.height = canvas.offsetHeight * devicePixelRatio;
            ctx.scale(devicePixelRatio, devicePixelRatio);
        }
        resize();
        window.addEventListener('resize', resize);

        // Create particles
        const COUNT = Math.min(120, Math.floor((canvas.offsetWidth * canvas.offsetHeight) / 8000));
        for (let i = 0; i < COUNT; i++) {
            const isGen = i < COUNT / 2;
            particles.push({
                x: rand(0, canvas.offsetWidth),
                y: rand(0, canvas.offsetHeight),
                vx: rand(-0.4, 0.4),
                vy: rand(-0.4, 0.4),
                r: rand(1.5, 3.5),
                color: isGen ? 'rgba(232,67,147,' : 'rgba(0,206,201,',
                baseAlpha: rand(0.3, 0.7),
            });
        }

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
        });
        canvas.addEventListener('mouseleave', () => { mouse.x = -9999; mouse.y = -9999; });

        function drawParticles() {
            const w = canvas.offsetWidth, h = canvas.offsetHeight;
            ctx.clearRect(0, 0, w, h);

            for (const p of particles) {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;

                // Mouse repulsion
                const dx = p.x - mouse.x, dy = p.y - mouse.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    const force = (120 - dist) / 120 * 0.8;
                    p.x += (dx / dist) * force;
                    p.y += (dy / dist) * force;
                }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = p.color + p.baseAlpha + ')';
                ctx.fill();
            }

            // Draw connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 100) {
                        const alpha = (1 - dist / 100) * 0.15;
                        // Gradient line between gen and disc colors
                        const isGenI = particles[i].color.includes('232');
                        const isGenJ = particles[j].color.includes('232');
                        if (isGenI !== isGenJ) {
                            ctx.strokeStyle = `rgba(180,120,180,${alpha})`;
                        } else if (isGenI) {
                            ctx.strokeStyle = `rgba(232,67,147,${alpha})`;
                        } else {
                            ctx.strokeStyle = `rgba(0,206,201,${alpha})`;
                        }
                        ctx.lineWidth = 0.8;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            requestAnimationFrame(drawParticles);
        }
        drawParticles();
    }

    /* ============================================================
       2. PIPELINE FLOW ANIMATION — glowing data packets
       ============================================================ */
    function initPipelineAnimation() {
        const pipeline = $('#gan-pipeline');
        if (!pipeline) return;

        const nodes = pipeline.querySelectorAll('.pipe-node__box');
        let currentNode = 0;
        let interval;

        function pulseNode(idx) {
            nodes.forEach((n, i) => {
                n.classList.remove('active');
                n.style.boxShadow = '';
            });
            if (idx < nodes.length) {
                const node = nodes[idx];
                node.classList.add('active');
                const colors = [
                    '0 0 25px rgba(255,255,255,0.15)',  // noise
                    '0 0 30px rgba(232,67,147,0.4)',     // generator
                    '0 0 25px rgba(255,107,107,0.35)',   // fake
                    '0 0 30px rgba(0,206,201,0.4)',      // discriminator
                    '0 0 25px rgba(0,184,148,0.35)',     // decision
                ];
                node.style.boxShadow = colors[idx] || colors[0];
            }
        }

        function startPulse() {
            interval = setInterval(() => {
                pulseNode(currentNode);
                currentNode = (currentNode + 1) % nodes.length;
            }, 800);
        }

        // Start when visible
        const obs = new IntersectionObserver((entries) => {
            entries.forEach(e => {
                if (e.isIntersecting) { startPulse(); obs.unobserve(e.target); }
            });
        }, { threshold: 0.3 });
        obs.observe(pipeline);
    }

    /* ============================================================
       3. TRAINING LOOP — Animated Canvas Visualization
       ============================================================ */
    function initTrainingLoop() {
        const canvas = $('#training-loop-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let W, H, animId, running = false, epoch = 0;
        const maxEpochs = 200;

        // Generator and Discriminator state
        let genSkill = 0.1;     // 0..1
        let discSkill = 0.5;    // 0..1
        let genLoss = 1.0;
        let discLoss = 0.5;
        const history = { genLoss: [], discLoss: [], genSkill: [], discSkill: [] };

        // "Brain" nodes for each network
        const genNodes = [], discNodes = [];
        const GEN_LAYERS = [4, 6, 8, 6, 4];
        const DISC_LAYERS = [4, 6, 8, 6, 4];

        function buildNodes(layers, centerX, startY, endY, arr) {
            arr.length = 0;
            const layerSpacing = 70;
            const totalWidth = (layers.length - 1) * layerSpacing;
            const startX = centerX - totalWidth / 2;
            layers.forEach((count, li) => {
                const lx = startX + li * layerSpacing;
                const nodeSpacing = (endY - startY) / (count + 1);
                for (let ni = 0; ni < count; ni++) {
                    arr.push({
                        x: lx, y: startY + nodeSpacing * (ni + 1),
                        layer: li, activation: rand(0.2, 0.8),
                        targetAct: rand(0.2, 0.8),
                        r: 5
                    });
                }
            });
        }

        function resize() {
            const rect = canvas.parentElement.getBoundingClientRect();
            W = canvas.width = rect.width * devicePixelRatio;
            H = canvas.height = rect.height * devicePixelRatio;
            ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
            const w = rect.width, h = rect.height;
            buildNodes(GEN_LAYERS, w * 0.22, 40, h - 80, genNodes);
            buildNodes(DISC_LAYERS, w * 0.78, 40, h - 80, discNodes);
        }
        resize();
        window.addEventListener('resize', resize);

        // Fake image grid (generated by G)
        const fakePixels = [];
        for (let i = 0; i < 64; i++) fakePixels.push(rand(0, 1));

        // Data flow particles
        let particles = [];

        function spawnParticle(sx, sy, ex, ey, color) {
            particles.push({ sx, sy, ex, ey, t: 0, speed: rand(0.008, 0.02), color, r: rand(2, 4) });
        }

        function updateParticles() {
            particles = particles.filter(p => p.t < 1);
            for (const p of particles) {
                p.t += p.speed;
            }
        }

        function drawParticles(w) {
            for (const p of particles) {
                const x = lerp(p.sx, p.ex, p.t);
                const y = lerp(p.sy, p.ey, p.t);
                const alpha = Math.sin(p.t * Math.PI);
                ctx.beginPath();
                ctx.arc(x, y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = p.color.replace('1)', alpha + ')');
                ctx.fill();
                // Trail
                ctx.beginPath();
                ctx.arc(x - (p.ex - p.sx) * 0.02, y - (p.ey - p.sy) * 0.02, p.r * 0.6, 0, Math.PI * 2);
                ctx.fillStyle = p.color.replace('1)', alpha * 0.4 + ')');
                ctx.fill();
            }
        }

        function drawNetwork(nodes, baseColor, glowColor, skill, label, centerX) {
            const w = canvas.parentElement.getBoundingClientRect().width;
            const h = canvas.parentElement.getBoundingClientRect().height;

            // Draw connections between adjacent layers
            const maxLayer = Math.max(...nodes.map(n => n.layer));
            for (let li = 0; li < maxLayer; li++) {
                const fromNodes = nodes.filter(n => n.layer === li);
                const toNodes = nodes.filter(n => n.layer === li + 1);
                for (const fn of fromNodes) {
                    for (const tn of toNodes) {
                        const strength = (fn.activation + tn.activation) / 2 * skill;
                        ctx.beginPath();
                        ctx.moveTo(fn.x, fn.y);
                        ctx.lineTo(tn.x, tn.y);
                        ctx.strokeStyle = baseColor.replace('1)', (strength * 0.4) + ')');
                        ctx.lineWidth = strength * 2;
                        ctx.stroke();
                    }
                }
            }

            // Draw nodes
            for (const n of nodes) {
                n.activation = lerp(n.activation, n.targetAct, 0.03);
                const glow = n.activation * skill;

                // Glow
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r + 8 * glow, 0, Math.PI * 2);
                ctx.fillStyle = glowColor.replace('1)', glow * 0.3 + ')');
                ctx.fill();

                // Core
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.fillStyle = baseColor.replace('1)', 0.3 + glow * 0.7 + ')');
                ctx.fill();
                ctx.strokeStyle = baseColor.replace('1)', 0.5 + glow * 0.5 + ')');
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Label
            ctx.font = '700 14px Inter, sans-serif';
            ctx.fillStyle = baseColor.replace('1)', '0.9)');
            ctx.textAlign = 'center';
            ctx.fillText(label, centerX, 28);

            // Skill bar
            const barW = 100, barH = 6, barX = centerX - barW / 2, barY = h - 50;
            ctx.fillStyle = 'rgba(255,255,255,0.06)';
            ctx.fillRect(barX, barY, barW, barH);
            ctx.fillStyle = baseColor.replace('1)', '0.7)');
            ctx.fillRect(barX, barY, barW * skill, barH);
            ctx.font = '500 11px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.fillText(`Skill: ${(skill * 100).toFixed(0)}%`, centerX, barY + 20);
        }

        function drawFakeImage(w, h) {
            const cx = w / 2, cy = h * 0.35;
            const size = 48;
            const cellSize = size / 8;
            const startX = cx - size / 2, startY = cy - size / 2;

            // Border glow
            ctx.shadowBlur = 15;
            ctx.shadowColor = genSkill > 0.5 ? 'rgba(0,184,148,0.5)' : 'rgba(255,107,107,0.5)';
            ctx.strokeStyle = genSkill > 0.5 ? 'rgba(0,184,148,0.6)' : 'rgba(255,107,107,0.6)';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX - 2, startY - 2, size + 4, size + 4);
            ctx.shadowBlur = 0;

            for (let i = 0; i < 64; i++) {
                const row = Math.floor(i / 8), col = i % 8;
                const noise = fakePixels[i] * (1 - genSkill) + genSkill * (Math.sin(i * 0.5 + epoch * 0.05) * 0.5 + 0.5);
                const gray = Math.floor(noise * 255);
                ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
                ctx.fillRect(startX + col * cellSize, startY + row * cellSize, cellSize, cellSize);
            }

            // Label
            ctx.font = '600 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillStyle = genSkill > 0.5 ? 'rgba(0,184,148,0.8)' : 'rgba(255,107,107,0.8)';
            ctx.fillText(genSkill > 0.5 ? '↑ Improving' : 'Noisy Output', cx, cy + size / 2 + 18);
        }

        function drawArrows(w, h) {
            const cy = h * 0.35;
            // Gen → Center
            drawGlowArrow(w * 0.38, cy, w * 0.45, cy, 'rgba(232,67,147,1)');
            // Center → Disc
            drawGlowArrow(w * 0.55, cy, w * 0.62, cy, 'rgba(0,206,201,1)');
            // Feedback loop (bottom arc)
            ctx.beginPath();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(255,255,255,0.12)';
            ctx.lineWidth = 1.5;
            ctx.moveTo(w * 0.72, h * 0.55);
            ctx.quadraticCurveTo(w * 0.5, h * 0.82, w * 0.28, h * 0.55);
            ctx.stroke();
            ctx.setLineDash([]);
            // Feedback label
            ctx.font = '500 10px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.textAlign = 'center';
            ctx.fillText('Feedback Loop', w * 0.5, h * 0.78);
        }

        function drawGlowArrow(x1, y1, x2, y2, color) {
            const t = (Date.now() % 2000) / 2000;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = color.replace('1)', '0.4)');
            ctx.lineWidth = 2;
            ctx.stroke();
            // Arrowhead
            const angle = Math.atan2(y2 - y1, x2 - x1);
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - 8 * Math.cos(angle - 0.4), y2 - 8 * Math.sin(angle - 0.4));
            ctx.lineTo(x2 - 8 * Math.cos(angle + 0.4), y2 - 8 * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fillStyle = color.replace('1)', '0.7)');
            ctx.fill();
            // Moving dot
            const dx = x2 - x1, dy = y2 - y1;
            const px = x1 + dx * t, py = y1 + dy * t;
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.shadowBlur = 10;
            ctx.shadowColor = color;
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        function drawLossChart(w, h) {
            if (history.genLoss.length < 2) return;
            const chartX = w * 0.3, chartY = h * 0.6;
            const chartW = w * 0.4, chartH = h * 0.25;

            // Background
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(chartX, chartY, chartW, chartH);
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.strokeRect(chartX, chartY, chartW, chartH);

            // Title
            ctx.font = '600 10px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.textAlign = 'center';
            ctx.fillText(`Loss Over Training (Epoch ${epoch})`, chartX + chartW / 2, chartY - 6);

            // Draw lines
            const maxPts = Math.min(history.genLoss.length, 100);
            const startIdx = Math.max(0, history.genLoss.length - maxPts);
            const step = chartW / maxPts;

            // Gen loss line
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(232,67,147,0.8)';
            ctx.lineWidth = 1.5;
            for (let i = 0; i < maxPts; i++) {
                const x = chartX + i * step;
                const y = chartY + chartH - history.genLoss[startIdx + i] * chartH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Disc loss line
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0,206,201,0.8)';
            ctx.lineWidth = 1.5;
            for (let i = 0; i < maxPts; i++) {
                const x = chartX + i * step;
                const y = chartY + chartH - history.discLoss[startIdx + i] * chartH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Legend
            ctx.font = '500 9px Inter, sans-serif';
            ctx.fillStyle = 'rgba(232,67,147,0.9)';
            ctx.textAlign = 'left';
            ctx.fillText('● G Loss', chartX + 8, chartY + 14);
            ctx.fillStyle = 'rgba(0,206,201,0.9)';
            ctx.fillText('● D Loss', chartX + 70, chartY + 14);
        }

        function step() {
            if (epoch >= maxEpochs) {
                running = false;
                const btn = $('#loop-play-btn');
                if (btn) btn.textContent = '✓ Complete';
                return;
            }
            epoch++;

            // Simulate adversarial training dynamics
            const noise = rand(-0.03, 0.03);
            // Generator slowly improves, disc tries to keep up
            genSkill = Math.min(0.95, genSkill + rand(0.002, 0.006));
            discSkill = Math.min(0.95, discSkill + rand(0.001, 0.004));
            // As gen improves, disc loss increases temporarily
            genLoss = Math.max(0.1, 1 - genSkill + noise);
            discLoss = Math.max(0.15, 0.5 - (discSkill - genSkill) * 0.5 + noise);

            history.genLoss.push(Math.min(1, Math.max(0, genLoss)));
            history.discLoss.push(Math.min(1, Math.max(0, discLoss)));
            history.genSkill.push(genSkill);
            history.discSkill.push(discSkill);

            // Randomize node targets periodically
            if (epoch % 5 === 0) {
                for (const n of genNodes) n.targetAct = rand(0.2 + genSkill * 0.3, 0.5 + genSkill * 0.5);
                for (const n of discNodes) n.targetAct = rand(0.2 + discSkill * 0.3, 0.5 + discSkill * 0.5);
            }

            // Spawn flow particles
            const w = canvas.parentElement.getBoundingClientRect().width;
            const h = canvas.parentElement.getBoundingClientRect().height;
            if (epoch % 3 === 0) {
                spawnParticle(w * 0.35, h * 0.35, w * 0.5, h * 0.35, 'rgba(232,67,147,1)');
                spawnParticle(w * 0.5, h * 0.35, w * 0.65, h * 0.35, 'rgba(0,206,201,1)');
            }
        }

        function draw() {
            const w = canvas.parentElement.getBoundingClientRect().width;
            const h = canvas.parentElement.getBoundingClientRect().height;
            ctx.clearRect(0, 0, w, h);

            drawNetwork(genNodes, 'rgba(232,67,147,1)', 'rgba(232,67,147,1)', genSkill, 'GENERATOR', w * 0.22);
            drawNetwork(discNodes, 'rgba(0,206,201,1)', 'rgba(0,206,201,1)', discSkill, 'DISCRIMINATOR', w * 0.78);
            drawFakeImage(w, h);
            drawArrows(w, h);
            updateParticles();
            drawParticles(w);
            drawLossChart(w, h);

            animId = requestAnimationFrame(draw);
        }

        // Controls
        let stepInterval;
        const playBtn = $('#loop-play-btn');
        const resetBtn = $('#loop-reset-btn');

        if (playBtn) {
            playBtn.addEventListener('click', () => {
                if (running) {
                    running = false;
                    clearInterval(stepInterval);
                    playBtn.textContent = '▶ Resume';
                } else {
                    running = true;
                    playBtn.textContent = '⏸ Pause';
                    stepInterval = setInterval(step, 80);
                }
            });
        }
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                running = false;
                clearInterval(stepInterval);
                epoch = 0;
                genSkill = 0.1;
                discSkill = 0.5;
                genLoss = 1.0;
                discLoss = 0.5;
                history.genLoss.length = 0;
                history.discLoss.length = 0;
                history.genSkill.length = 0;
                history.discSkill.length = 0;
                for (const n of genNodes) n.targetAct = rand(0.2, 0.5);
                for (const n of discNodes) n.targetAct = rand(0.2, 0.5);
                particles = [];
                if (playBtn) playBtn.textContent = '▶ Play Training';
            });
        }

        // Start rendering
        resize();
        draw();
    }

    /* ============================================================
       4. GENERATOR VISUALIZATION — Noise → Image morphing
       ============================================================ */
    function initGeneratorVis() {
        const noiseCanvas  = $('#genvis-noise-canvas');
        const netCanvas    = $('#genvis-network-canvas');
        const outCanvas    = $('#genvis-output-canvas');
        if (!noiseCanvas || !netCanvas || !outCanvas) return;

        const nCtx = noiseCanvas.getContext('2d');
        const gCtx = netCanvas.getContext('2d');
        const oCtx = outCanvas.getContext('2d');

        const noiseSlider    = $('#noise-slider');
        const noiseSliderVal = $('#noise-slider-val');
        const learnSlider    = $('#learning-slider');
        const learnSliderVal = $('#learning-slider-val');
        const generateBtn    = $('#genvis-generate-btn');
        const animateBtn     = $('#genvis-animate-btn');
        const qualityLabel   = $('#genvis-quality-label');
        const outputWrap     = $('#genvis-output-wrap');

        let noiseSeed = 50;
        let learnProgress = 0;    // 0..1
        let autoLearnRunning = false;
        let autoLearnInterval = null;

        // ---- Digit templates (10x10 bitmaps for digits 0-9) ----
        const DIGITS = [
            // 0
            [0,1,1,1,1,1,1,1,1,0,
             1,1,0,0,0,0,0,0,1,1,
             1,0,0,0,0,0,0,0,0,1,
             1,0,0,0,0,0,0,0,0,1,
             1,0,0,0,0,0,0,0,0,1,
             1,0,0,0,0,0,0,0,0,1,
             1,0,0,0,0,0,0,0,0,1,
             1,0,0,0,0,0,0,0,0,1,
             1,1,0,0,0,0,0,0,1,1,
             0,1,1,1,1,1,1,1,1,0],
            // 1
            [0,0,0,0,1,1,0,0,0,0,
             0,0,0,1,1,1,0,0,0,0,
             0,0,1,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,1,1,1,1,1,1,0,0],
            // 2
            [0,1,1,1,1,1,1,1,0,0,
             1,1,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,1,1,0,0,0,0,0,
             0,0,1,0,0,0,0,0,0,0,
             0,1,0,0,0,0,0,0,0,0,
             1,1,1,1,1,1,1,1,1,0],
            // 3
            [0,1,1,1,1,1,1,1,0,0,
             1,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,1,1,1,1,0,0,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,1,0,
             1,0,0,0,0,0,0,1,1,0,
             0,1,1,1,1,1,1,1,0,0],
            // 4
            [0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,0,1,1,1,0,0,
             0,0,0,0,1,0,1,1,0,0,
             0,0,0,1,0,0,1,1,0,0,
             0,0,1,0,0,0,1,1,0,0,
             0,1,0,0,0,0,1,1,0,0,
             1,1,1,1,1,1,1,1,1,1,
             0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,0,0,1,1,0,0],
            // 5
            [1,1,1,1,1,1,1,1,1,0,
             1,0,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,
             1,1,1,1,1,1,1,0,0,0,
             0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,1,0,
             1,0,0,0,0,0,0,1,1,0,
             0,1,1,1,1,1,1,1,0,0],
            // 6
            [0,0,1,1,1,1,1,1,0,0,
             0,1,1,0,0,0,0,0,0,0,
             1,1,0,0,0,0,0,0,0,0,
             1,0,0,0,0,0,0,0,0,0,
             1,1,1,1,1,1,1,0,0,0,
             1,1,0,0,0,0,1,1,0,0,
             1,0,0,0,0,0,0,1,0,0,
             1,0,0,0,0,0,0,1,0,0,
             0,1,0,0,0,0,1,1,0,0,
             0,0,1,1,1,1,1,0,0,0],
            // 7
            [1,1,1,1,1,1,1,1,1,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,1,1,0,0,
             0,0,0,0,0,1,1,0,0,0,
             0,0,0,0,1,1,0,0,0,0,
             0,0,0,0,1,0,0,0,0,0,
             0,0,0,1,1,0,0,0,0,0,
             0,0,0,1,0,0,0,0,0,0,
             0,0,0,1,0,0,0,0,0,0,
             0,0,0,1,0,0,0,0,0,0],
            // 8
            [0,1,1,1,1,1,1,1,0,0,
             1,1,0,0,0,0,0,1,1,0,
             1,0,0,0,0,0,0,0,1,0,
             1,1,0,0,0,0,0,1,1,0,
             0,1,1,1,1,1,1,1,0,0,
             1,1,0,0,0,0,0,1,1,0,
             1,0,0,0,0,0,0,0,1,0,
             1,0,0,0,0,0,0,0,1,0,
             1,1,0,0,0,0,0,1,1,0,
             0,1,1,1,1,1,1,1,0,0],
            // 9
            [0,1,1,1,1,1,1,0,0,0,
             1,1,0,0,0,0,1,1,0,0,
             1,0,0,0,0,0,0,1,0,0,
             1,0,0,0,0,0,0,1,1,0,
             0,1,1,1,1,1,1,1,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,0,1,0,
             0,0,0,0,0,0,0,1,1,0,
             0,0,0,0,0,0,1,1,0,0,
             0,0,1,1,1,1,1,0,0,0],
        ];

        // ---- Seeded random for deterministic noise ----
        function seededRand(seed) {
            let s = seed;
            return function() {
                s = (s * 16807 + 0) % 2147483647;
                return (s - 1) / 2147483646;
            };
        }

        // ---- Noise particles ----
        const NOISE_COUNT = 100;
        let noiseParticles = [];

        function generateNoiseParticles() {
            const rng = seededRand(Math.floor(noiseSeed * 137) + 1);
            noiseParticles = [];
            const cw = noiseCanvas.width, ch = noiseCanvas.height;
            for (let i = 0; i < NOISE_COUNT; i++) {
                noiseParticles.push({
                    x: 20 + rng() * (cw - 40),
                    y: 20 + rng() * (ch - 40),
                    val: rng(),
                    r: 3 + rng() * 3,
                    phase: rng() * Math.PI * 2,
                });
            }
        }

        // ---- Output particles (morph from noise to digit) ----
        let outputParticles = [];

        function getTargetDigit() {
            return Math.floor((noiseSeed / 100) * 10) % 10;
        }

        function generateOutputParticles() {
            const rng = seededRand(Math.floor(noiseSeed * 97) + 7);
            outputParticles = [];
            const cw = outCanvas.width, ch = outCanvas.height;
            const digit = DIGITS[getTargetDigit()];
            const gridSize = 10;
            const cellW = (cw - 40) / gridSize;
            const cellH = (ch - 40) / gridSize;

            for (let row = 0; row < gridSize; row++) {
                for (let col = 0; col < gridSize; col++) {
                    const idx = row * gridSize + col;
                    const isOn = digit[idx] === 1;
                    // Target position (grid)
                    const tx = 20 + col * cellW + cellW / 2;
                    const ty = 20 + row * cellH + cellH / 2;
                    // Noise position (random scatter)
                    const nx = 20 + rng() * (cw - 40);
                    const ny = 20 + rng() * (ch - 40);
                    outputParticles.push({
                        nx, ny,  // noise pos
                        tx, ty,  // target pos
                        isOn,
                        r: Math.min(cellW, cellH) * 0.38,
                        val: rng(),
                        phase: rng() * Math.PI * 2,
                    });
                }
            }
        }

        // ---- Draw Noise Panel ----
        function drawNoise(t) {
            const cw = noiseCanvas.width, ch = noiseCanvas.height;
            nCtx.clearRect(0, 0, cw, ch);

            for (const p of noiseParticles) {
                const wobbleX = Math.sin(t * 0.002 + p.phase) * 3;
                const wobbleY = Math.cos(t * 0.0015 + p.phase * 1.3) * 3;
                const px = p.x + wobbleX;
                const py = p.y + wobbleY;
                const brightness = 0.3 + p.val * 0.7;

                // Glow
                nCtx.beginPath();
                nCtx.arc(px, py, p.r + 4, 0, Math.PI * 2);
                nCtx.fillStyle = `rgba(232,67,147,${brightness * 0.15})`;
                nCtx.fill();

                // Core
                nCtx.beginPath();
                nCtx.arc(px, py, p.r, 0, Math.PI * 2);
                const hue = 340 + p.val * 30;
                nCtx.fillStyle = `hsla(${hue},80%,${50 + brightness * 30}%,${brightness})`;
                nCtx.fill();
            }

            // Label
            nCtx.font = '500 11px JetBrains Mono, monospace';
            nCtx.fillStyle = 'rgba(255,255,255,0.3)';
            nCtx.textAlign = 'center';
            nCtx.fillText(`seed: ${Math.floor(noiseSeed)}`, cw / 2, ch - 8);
        }

        // ---- Draw Network Panel ----
        const NET_LAYERS = [8, 12, 16, 12, 8];
        let netNodes = [];

        function buildNetNodes() {
            netNodes = [];
            const cw = netCanvas.width, ch = netCanvas.height;
            const layerSpacing = (cw - 60) / (NET_LAYERS.length - 1);
            NET_LAYERS.forEach((count, li) => {
                const lx = 30 + li * layerSpacing;
                const nodeSpacing = (ch - 60) / (count + 1);
                for (let ni = 0; ni < count; ni++) {
                    netNodes.push({
                        x: lx, y: 30 + nodeSpacing * (ni + 1),
                        layer: li,
                        activation: 0,
                    });
                }
            });
        }
        buildNetNodes();

        // Flow particles in network
        let netFlowParticles = [];

        function spawnNetFlow() {
            const cw = netCanvas.width, ch = netCanvas.height;
            for (let i = 0; i < 5; i++) {
                netFlowParticles.push({
                    x: 10 + rand(0, 20),
                    y: rand(40, ch - 40),
                    vx: rand(1.5, 3.5),
                    vy: rand(-0.5, 0.5),
                    life: 1,
                    r: rand(2, 4),
                });
            }
        }

        function drawNetwork(t) {
            const cw = netCanvas.width, ch = netCanvas.height;
            gCtx.clearRect(0, 0, cw, ch);

            // Draw connections
            const maxLayer = NET_LAYERS.length - 1;
            for (let li = 0; li < maxLayer; li++) {
                const fromNodes = netNodes.filter(n => n.layer === li);
                const toNodes = netNodes.filter(n => n.layer === li + 1);
                for (let fi = 0; fi < fromNodes.length; fi += 2) {
                    for (let ti = 0; ti < toNodes.length; ti += 2) {
                        const fn = fromNodes[fi], tn = toNodes[ti];
                        const strength = 0.03 + learnProgress * 0.08;
                        gCtx.beginPath();
                        gCtx.moveTo(fn.x, fn.y);
                        gCtx.lineTo(tn.x, tn.y);
                        gCtx.strokeStyle = `rgba(232,67,147,${strength})`;
                        gCtx.lineWidth = 0.5;
                        gCtx.stroke();
                    }
                }
            }

            // Draw nodes
            for (const n of netNodes) {
                const wave = Math.sin(t * 0.003 + n.x * 0.02 + n.y * 0.01) * 0.5 + 0.5;
                const act = 0.15 + wave * learnProgress * 0.85;
                n.activation = act;

                // Glow
                gCtx.beginPath();
                gCtx.arc(n.x, n.y, 4 + act * 5, 0, Math.PI * 2);
                gCtx.fillStyle = `rgba(232,67,147,${act * 0.2})`;
                gCtx.fill();

                // Core
                gCtx.beginPath();
                gCtx.arc(n.x, n.y, 3, 0, Math.PI * 2);
                gCtx.fillStyle = `rgba(232,67,147,${0.25 + act * 0.75})`;
                gCtx.fill();
            }

            // Flow particles
            netFlowParticles = netFlowParticles.filter(p => p.life > 0 && p.x < cw);
            for (const p of netFlowParticles) {
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 0.008;
                const progress = p.x / cw;
                const alpha = p.life * (1 - Math.abs(progress - 0.5) * 1.2);
                gCtx.beginPath();
                gCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                gCtx.fillStyle = `rgba(232,67,147,${Math.max(0, alpha)})`;
                gCtx.fill();
                // Trail glow
                gCtx.beginPath();
                gCtx.arc(p.x, p.y, p.r + 3, 0, Math.PI * 2);
                gCtx.fillStyle = `rgba(232,67,147,${Math.max(0, alpha * 0.3)})`;
                gCtx.fill();
            }

            // Layer labels
            gCtx.font = '600 9px Inter, sans-serif';
            gCtx.fillStyle = 'rgba(255,255,255,0.3)';
            gCtx.textAlign = 'center';
            const labels = ['Input', 'Dense 1', 'Dense 2', 'Dense 3', 'Output'];
            const layerSpacing = (cw - 60) / (NET_LAYERS.length - 1);
            labels.forEach((lbl, i) => {
                gCtx.fillText(lbl, 30 + i * layerSpacing, ch - 8);
            });
        }

        // ---- Draw Output Panel ----
        function drawOutput(t) {
            const cw = outCanvas.width, ch = outCanvas.height;
            oCtx.clearRect(0, 0, cw, ch);

            const lp = learnProgress;  // 0..1

            for (const p of outputParticles) {
                // Interpolate position: noise → target
                const x = lerp(p.nx, p.tx, lp);
                const y = lerp(p.ny, p.ty, lp);

                // Wobble (decreases as learning increases)
                const wobble = (1 - lp) * 6;
                const wx = Math.sin(t * 0.003 + p.phase) * wobble;
                const wy = Math.cos(t * 0.0025 + p.phase * 1.7) * wobble;

                const px = x + wx;
                const py = y + wy;

                // Size: starts small random, becomes grid-perfect
                const baseR = lerp(rand(2, 5), p.r, lp * lp);

                if (p.isOn) {
                    // Foreground pixel — white/bright
                    const brightness = lerp(0.3 + p.val * 0.4, 0.85 + p.val * 0.15, lp);

                    // Glow (learning signal)
                    if (lp > 0.1) {
                        oCtx.beginPath();
                        oCtx.arc(px, py, baseR + 3 * lp, 0, Math.PI * 2);
                        oCtx.fillStyle = `rgba(232,67,147,${lp * 0.15})`;
                        oCtx.fill();
                    }

                    oCtx.beginPath();
                    if (lp > 0.7) {
                        // Square pixels at high learning
                        const s = baseR * 1.8;
                        oCtx.rect(px - s/2, py - s/2, s, s);
                    } else {
                        oCtx.arc(px, py, baseR, 0, Math.PI * 2);
                    }
                    const gray = Math.floor(brightness * 255);
                    oCtx.fillStyle = `rgb(${gray},${gray},${gray})`;
                    oCtx.fill();
                } else {
                    // Background pixel — dark
                    const alpha = lerp(0.4, 0.08, lp);
                    oCtx.beginPath();
                    if (lp > 0.7) {
                        const s = baseR * 1.8;
                        oCtx.rect(px - s/2, py - s/2, s, s);
                    } else {
                        oCtx.arc(px, py, baseR, 0, Math.PI * 2);
                    }
                    oCtx.fillStyle = `rgba(150,100,160,${alpha})`;
                    oCtx.fill();
                }
            }

            // Digit label at high learning
            if (lp > 0.6) {
                const alpha = (lp - 0.6) / 0.4;
                oCtx.font = '700 16px Inter, sans-serif';
                oCtx.fillStyle = `rgba(0,184,148,${alpha * 0.8})`;
                oCtx.textAlign = 'center';
                oCtx.fillText(`Digit: ${getTargetDigit()}`, cw / 2, ch - 10);
            }
        }

        // ---- Quality label & glow ----
        function updateQualityUI() {
            const lp = learnProgress;
            let label, cls;
            if (lp < 0.15) { label = 'Quality: Pure noise'; cls = ''; }
            else if (lp < 0.35) { label = 'Quality: Noisy blobs'; cls = ''; }
            else if (lp < 0.55) { label = 'Quality: Emerging structure'; cls = 'glow-improving'; }
            else if (lp < 0.75) { label = 'Quality: Recognizable shapes'; cls = 'glow-improving'; }
            else if (lp < 0.9) { label = 'Quality: Near-realistic'; cls = 'glow-trained'; }
            else { label = 'Quality: High fidelity ✓'; cls = 'glow-trained'; }

            if (qualityLabel) qualityLabel.textContent = label;
            if (outputWrap) {
                outputWrap.classList.remove('glow-improving', 'glow-trained');
                if (cls) outputWrap.classList.add(cls);
            }
        }

        // ---- Main render loop ----
        let animRunning = true;
        let lastFlowSpawn = 0;

        function render(t) {
            if (!animRunning) return;

            // Spawn network flow particles periodically
            if (t - lastFlowSpawn > 300) {
                spawnNetFlow();
                lastFlowSpawn = t;
            }

            drawNoise(t);
            drawNetwork(t);
            drawOutput(t);

            requestAnimationFrame(render);
        }

        // ---- Events ----
        if (noiseSlider) {
            noiseSlider.addEventListener('input', () => {
                noiseSeed = parseFloat(noiseSlider.value);
                if (noiseSliderVal) noiseSliderVal.textContent = Math.floor(noiseSeed);
                generateNoiseParticles();
                generateOutputParticles();
            });
        }

        if (learnSlider) {
            learnSlider.addEventListener('input', () => {
                learnProgress = parseFloat(learnSlider.value) / 100;
                if (learnSliderVal) learnSliderVal.textContent = Math.floor(learnProgress * 100) + '%';
                updateQualityUI();
            });
        }

        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                // Randomize noise and regenerate
                noiseSeed = Math.floor(rand(0, 100));
                if (noiseSlider) noiseSlider.value = noiseSeed;
                if (noiseSliderVal) noiseSliderVal.textContent = Math.floor(noiseSeed);
                generateNoiseParticles();
                generateOutputParticles();
            });
        }

        if (animateBtn) {
            animateBtn.addEventListener('click', () => {
                if (autoLearnRunning) {
                    autoLearnRunning = false;
                    clearInterval(autoLearnInterval);
                    animateBtn.textContent = '▶ Auto-Learn';
                } else {
                    autoLearnRunning = true;
                    animateBtn.textContent = '⏸ Pause';
                    // Reset to 0 if at max
                    if (learnProgress >= 0.98) {
                        learnProgress = 0;
                        if (learnSlider) learnSlider.value = 0;
                    }
                    autoLearnInterval = setInterval(() => {
                        learnProgress = Math.min(1, learnProgress + 0.005);
                        if (learnSlider) learnSlider.value = Math.floor(learnProgress * 100);
                        if (learnSliderVal) learnSliderVal.textContent = Math.floor(learnProgress * 100) + '%';
                        updateQualityUI();
                        if (learnProgress >= 1) {
                            clearInterval(autoLearnInterval);
                            autoLearnRunning = false;
                            animateBtn.textContent = '▶ Auto-Learn';
                        }
                    }, 40);
                }
            });
        }

        // Pause when out of view
        const visObs = new IntersectionObserver((entries) => {
            entries.forEach(e => {
                animRunning = e.isIntersecting;
                if (animRunning) requestAnimationFrame(render);
            });
        }, { threshold: 0.1 });
        const section = $('#generator-vis');
        if (section) visObs.observe(section);

        // ---- Init ----
        generateNoiseParticles();
        generateOutputParticles();
        updateQualityUI();
        requestAnimationFrame(render);
    }

    /* ============================================================
       5. DISCRIMINATOR VISUALIZATION — Real vs Fake classification
       ============================================================ */
    function initDiscriminatorVis() {
        const classifyBtn = $('#disc-classify-btn');
        const newBtn      = $('#disc-new-btn');
        const meterReal   = $('#disc-meter-real');
        const meterFake   = $('#disc-meter-fake');
        const confReal    = $('#disc-conf-real');
        const confFake    = $('#disc-conf-fake');
        const decReal     = $('#disc-dec-real');
        const decFake     = $('#disc-dec-fake');
        const brain       = $('#disc-brain');

        if (!classifyBtn) return;

        // Simple digit patterns (8x8) for "real" images
        const REAL_PATTERNS = [
            [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0], // 0
            [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0], // 1
            [0,1,1,1,1,1,0,0, 1,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,1,0,0,0,0,0,0, 1,1,1,1,1,1,1,0], // 2
            [0,1,1,1,1,1,0,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,1,1,1,0,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0, 0,1,1,1,1,1,0,0], // 3
            [0,0,0,0,1,1,0,0, 0,0,0,1,0,1,0,0, 0,0,1,0,0,1,0,0, 0,1,0,0,0,1,0,0, 1,1,1,1,1,1,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0], // 4
            [1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,0, 1,1,1,1,1,0,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 1,1,1,1,1,0,0,0], // 5
            [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 1,1,1,1,1,0,0,0, 1,0,0,0,0,1,0,0, 1,0,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0], // 6
            [1,1,1,1,1,1,1,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0], // 7
        ];

        function drawDigit(canvasId, pattern, isReal) {
            const c = document.getElementById(canvasId);
            if (!c) return;
            const ctx = c.getContext('2d');
            ctx.clearRect(0, 0, 64, 64);
            const cell = 8;
            for (let i = 0; i < 64; i++) {
                const row = Math.floor(i / 8), col = i % 8;
                const on = pattern[i] === 1;
                if (isReal) {
                    // Clean, crisp real digit
                    const v = on ? 200 + Math.floor(Math.random() * 55) : 5 + Math.floor(Math.random() * 20);
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                } else {
                    // Noisy fake
                    const noise = Math.random();
                    const base = on ? 80 + noise * 100 : noise * 120;
                    const r = Math.floor(base * (0.8 + Math.random() * 0.4));
                    const g = Math.floor(base * (0.6 + Math.random() * 0.3));
                    const b = Math.floor(base * (0.7 + Math.random() * 0.5));
                    ctx.fillStyle = `rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,b)})`;
                }
                ctx.fillRect(col * cell, row * cell, cell, cell);
            }
            // Green/red border indicator
            ctx.strokeStyle = isReal ? 'rgba(0,184,148,0.4)' : 'rgba(255,107,107,0.4)';
            ctx.lineWidth = 2;
            ctx.strokeRect(1, 1, 62, 62);
        }

        function generateSamples() {
            // Reset meters
            if (meterReal) meterReal.style.width = '0%';
            if (meterFake) meterFake.style.width = '0%';
            if (confReal) confReal.textContent = '0%';
            if (confFake) confFake.textContent = '0%';
            if (decReal) decReal.classList.remove('lit');
            if (decFake) decFake.classList.remove('lit');
            if (brain) brain.classList.remove('active');

            // Pick 4 random real digits
            for (let i = 0; i < 4; i++) {
                const idx = Math.floor(Math.random() * REAL_PATTERNS.length);
                drawDigit('disc-real-' + i, REAL_PATTERNS[idx], true);
            }
            // Pick 4 random fake (noisy versions)
            for (let i = 0; i < 4; i++) {
                const idx = Math.floor(Math.random() * REAL_PATTERNS.length);
                drawDigit('disc-fake-' + i, REAL_PATTERNS[idx], false);
            }
        }

        function classify() {
            if (brain) brain.classList.add('active');

            // Animate confidence meters with slight delay
            const realConf = 85 + Math.floor(Math.random() * 14); // 85-98%
            const fakeConf = 72 + Math.floor(Math.random() * 22); // 72-93%

            setTimeout(() => {
                if (meterReal) meterReal.style.width = realConf + '%';
                if (confReal) confReal.textContent = realConf + '%';
                if (decReal) decReal.classList.add('lit');
            }, 400);

            setTimeout(() => {
                if (meterFake) meterFake.style.width = fakeConf + '%';
                if (confFake) confFake.textContent = fakeConf + '%';
                if (decFake) decFake.classList.add('lit');
            }, 800);
        }

        classifyBtn.addEventListener('click', classify);
        if (newBtn) newBtn.addEventListener('click', generateSamples);

        // Initial samples
        generateSamples();
    }

    /* ============================================================
       6. DISCRIMINATOR INTERNALS — Feature extraction visualization
       ============================================================ */
    function initDiscInternals() {
        const toggleReal = $('#di-toggle-real');
        const toggleFake = $('#di-toggle-fake');
        const runBtn     = $('#di-run-btn');
        if (!runBtn) return;

        let inputType = 'real'; // 'real' | 'fake'
        let isRunning = false;

        // Digit patterns (8x8)
        const DIGIT_PATTERNS = [
            [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0],
            [0,1,1,1,1,1,0,0, 1,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,1,0,0,0,0,0,0, 1,1,1,1,1,1,1,0],
            [0,0,0,0,1,1,0,0, 0,0,0,1,0,1,0,0, 0,0,1,0,0,1,0,0, 0,1,0,0,0,1,0,0, 1,1,1,1,1,1,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0],
            [1,1,1,1,1,1,1,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0],
        ];

        // --- Draw input image ---
        function drawInput() {
            const c = document.getElementById('di-input-canvas');
            if (!c) return;
            const ctx = c.getContext('2d');
            ctx.clearRect(0, 0, 80, 80);
            const pattern = DIGIT_PATTERNS[Math.floor(Math.random() * DIGIT_PATTERNS.length)];
            const cell = 10;
            const isReal = inputType === 'real';

            for (let i = 0; i < 64; i++) {
                const row = Math.floor(i / 8), col = i % 8;
                const on = pattern[i] === 1;
                if (isReal) {
                    const v = on ? 190 + Math.floor(Math.random() * 65) : 8 + Math.floor(Math.random() * 18);
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                } else {
                    const noise = Math.random();
                    const base = on ? 60 + noise * 110 : noise * 100;
                    const r = Math.floor(base * (0.7 + Math.random() * 0.5));
                    const g = Math.floor(base * (0.5 + Math.random() * 0.4));
                    const b = Math.floor(base * (0.6 + Math.random() * 0.6));
                    ctx.fillStyle = `rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,b)})`;
                }
                ctx.fillRect(col * cell, row * cell, cell, cell);
            }
            // Border
            ctx.strokeStyle = isReal ? 'rgba(0,184,148,0.5)' : 'rgba(255,107,107,0.5)';
            ctx.lineWidth = 2;
            ctx.strokeRect(1, 1, 78, 78);

            // Update label
            const label = document.getElementById('di-input-label');
            if (label) label.textContent = isReal ? '28×28 — Real data' : '28×28 — Generator output';

            return pattern;
        }

        // --- Convolution feature maps ---
        function drawFeatureMaps(pattern) {
            const isReal = inputType === 'real';

            // Kernels
            const kernels = [
                // Horizontal edges
                [[-1,-1,-1],[0,0,0],[1,1,1]],
                // Vertical edges
                [[-1,0,1],[-1,0,1],[-1,0,1]],
                // Diagonal
                [[1,0,-1],[0,1,0],[-1,0,1]],
                // Corners
                [[0,1,0],[1,-4,1],[0,1,0]],
            ];
            const colors = [
                [0, 206, 201],  // teal
                [232, 67, 147], // pink
                [0, 184, 148],  // green
                [253, 121, 168],// salmon
            ];

            // Build 8x8 float image
            const img = [];
            for (let i = 0; i < 64; i++) img.push(pattern[i]);

            kernels.forEach((kernel, ki) => {
                const canvas = document.getElementById('di-fmap-' + ki);
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, 48, 48);
                const outSize = 6; // 8-3+1 = 6
                const cell = 8;

                for (let r = 0; r < outSize; r++) {
                    for (let c = 0; c < outSize; c++) {
                        let sum = 0;
                        for (let kr = 0; kr < 3; kr++) {
                            for (let kc = 0; kc < 3; kc++) {
                                sum += img[(r + kr) * 8 + (c + kc)] * kernel[kr][kc];
                            }
                        }
                        // ReLU
                        let val = Math.max(0, sum);
                        // Add noise for fake images
                        if (!isReal) val += (Math.random() - 0.5) * 0.8;
                        val = Math.min(1, Math.max(0, val / 3));

                        const [cr, cg, cb] = colors[ki];
                        ctx.fillStyle = `rgba(${cr},${cg},${cb},${0.15 + val * 0.85})`;
                        ctx.fillRect(c * cell, r * cell, cell, cell);
                    }
                }
                // Subtle border
                ctx.strokeStyle = `rgba(${colors[ki][0]},${colors[ki][1]},${colors[ki][2]},0.3)`;
                ctx.lineWidth = 1;
                ctx.strokeRect(0, 0, 48, 48);
            });
        }

        // --- Dense layer network visualization ---
        let denseAnimFrame = null;
        function drawDenseNetwork(progress) {
            const canvas = document.getElementById('di-dense-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const W = 120, H = 140;
            ctx.clearRect(0, 0, W, H);

            const layers = [8, 12, 6, 2];
            const layerX = [15, 45, 75, 105];
            const isReal = inputType === 'real';
            const baseColor = isReal ? [0, 184, 148] : [255, 107, 107];

            // Draw connections
            for (let li = 0; li < layers.length - 1; li++) {
                const fromCount = layers[li];
                const toCount = layers[li + 1];
                const fromSpacing = (H - 20) / (fromCount + 1);
                const toSpacing = (H - 20) / (toCount + 1);

                for (let fi = 0; fi < fromCount; fi++) {
                    for (let ti = 0; ti < toCount; ti++) {
                        const fx = layerX[li], fy = 10 + fromSpacing * (fi + 1);
                        const tx = layerX[li + 1], ty = 10 + toSpacing * (ti + 1);
                        const strength = progress * (0.05 + Math.random() * 0.1);
                        ctx.beginPath();
                        ctx.moveTo(fx, fy);
                        ctx.lineTo(tx, ty);
                        ctx.strokeStyle = `rgba(${baseColor[0]},${baseColor[1]},${baseColor[2]},${strength})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }

            // Draw nodes
            for (let li = 0; li < layers.length; li++) {
                const count = layers[li];
                const spacing = (H - 20) / (count + 1);
                for (let ni = 0; ni < count; ni++) {
                    const x = layerX[li];
                    const y = 10 + spacing * (ni + 1);
                    const act = progress * (0.3 + Math.random() * 0.7);

                    // Glow
                    ctx.beginPath();
                    ctx.arc(x, y, 3 + act * 4, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${baseColor[0]},${baseColor[1]},${baseColor[2]},${act * 0.2})`;
                    ctx.fill();

                    // Core
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${baseColor[0]},${baseColor[1]},${baseColor[2]},${0.3 + act * 0.7})`;
                    ctx.fill();
                }
            }

            // Layer labels
            ctx.font = '500 7px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.25)';
            ctx.textAlign = 'center';
            ['FC₁', 'FC₂', 'FC₃', 'Out'].forEach((lbl, i) => {
                ctx.fillText(lbl, layerX[i], H - 2);
            });
        }

        // Animate dense network
        function animateDense(duration, callback) {
            const startTime = performance.now();
            function frame(t) {
                const elapsed = t - startTime;
                const progress = Math.min(1, elapsed / duration);
                drawDenseNetwork(progress);
                if (progress < 1) {
                    denseAnimFrame = requestAnimationFrame(frame);
                } else if (callback) {
                    callback();
                }
            }
            denseAnimFrame = requestAnimationFrame(frame);
        }

        // --- Set probability output ---
        function setOutput(pReal) {
            const pFake = 1 - pReal;
            const probRealFill = document.getElementById('di-prob-real');
            const probFakeFill = document.getElementById('di-prob-fake');
            const probRealVal  = document.getElementById('di-prob-real-val');
            const probFakeVal  = document.getElementById('di-prob-fake-val');
            const verdict      = document.getElementById('di-verdict');

            if (probRealFill) probRealFill.style.width = (pReal * 100) + '%';
            if (probFakeFill) probFakeFill.style.width = (pFake * 100) + '%';
            if (probRealVal) probRealVal.textContent = (pReal * 100).toFixed(1) + '%';
            if (probFakeVal) probFakeVal.textContent = (pFake * 100).toFixed(1) + '%';

            if (verdict) {
                if (pReal > 0.5) {
                    verdict.textContent = '✓ CLASSIFIED: REAL';
                    verdict.className = 'di-verdict real';
                } else {
                    verdict.textContent = '✗ CLASSIFIED: FAKE';
                    verdict.className = 'di-verdict fake';
                }
            }
        }

        // --- Reset all stages ---
        function resetStages() {
            const litClass = inputType === 'real' ? 'lit-real' : 'lit-fake';
            ['di-stage-input', 'di-stage-conv', 'di-stage-dense', 'di-stage-output'].forEach(id => {
                const el = document.getElementById(id);
                if (el) { el.classList.remove('active', 'lit-real', 'lit-fake', 'lit-disc'); }
            });
            ['di-arrow-1', 'di-arrow-2', 'di-arrow-3'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });
            // Reset probs
            const probRealFill = document.getElementById('di-prob-real');
            const probFakeFill = document.getElementById('di-prob-fake');
            const probRealVal  = document.getElementById('di-prob-real-val');
            const probFakeVal  = document.getElementById('di-prob-fake-val');
            const verdict      = document.getElementById('di-verdict');
            if (probRealFill) probRealFill.style.width = '0%';
            if (probFakeFill) probFakeFill.style.width = '0%';
            if (probRealVal) probRealVal.textContent = '—';
            if (probFakeVal) probFakeVal.textContent = '—';
            if (verdict) { verdict.textContent = 'Awaiting input…'; verdict.className = 'di-verdict'; }

            // Clear dense canvas
            const denseC = document.getElementById('di-dense-canvas');
            if (denseC) denseC.getContext('2d').clearRect(0, 0, 120, 140);
            // Clear feature maps
            for (let i = 0; i < 4; i++) {
                const fm = document.getElementById('di-fmap-' + i);
                if (fm) fm.getContext('2d').clearRect(0, 0, 48, 48);
            }
        }

        // --- Run evaluation pipeline ---
        function runEvaluation() {
            if (isRunning) return;
            isRunning = true;
            if (denseAnimFrame) cancelAnimationFrame(denseAnimFrame);
            resetStages();

            const litClass = inputType === 'real' ? 'lit-real' : 'lit-fake';

            // Step 1: Show input (immediate)
            const pattern = drawInput();
            const stageInput = document.getElementById('di-stage-input');
            if (stageInput) { stageInput.classList.add('active', litClass); }

            // Step 2: Arrow 1 lights up (400ms)
            setTimeout(() => {
                const a1 = document.getElementById('di-arrow-1');
                if (a1) a1.classList.add('active');
            }, 400);

            // Step 3: Feature maps (700ms)
            setTimeout(() => {
                drawFeatureMaps(pattern);
                const stageConv = document.getElementById('di-stage-conv');
                if (stageConv) stageConv.classList.add('active', 'lit-disc');
            }, 700);

            // Step 4: Arrow 2 (1100ms)
            setTimeout(() => {
                const a2 = document.getElementById('di-arrow-2');
                if (a2) a2.classList.add('active');
            }, 1100);

            // Step 5: Dense layers animate (1400ms, 1s animation)
            setTimeout(() => {
                const stageDense = document.getElementById('di-stage-dense');
                if (stageDense) stageDense.classList.add('active', 'lit-disc');
                animateDense(1000, () => {
                    // Step 6: Arrow 3
                    const a3 = document.getElementById('di-arrow-3');
                    if (a3) a3.classList.add('active');

                    // Step 7: Output (after small delay)
                    setTimeout(() => {
                        const stageOutput = document.getElementById('di-stage-output');
                        if (stageOutput) stageOutput.classList.add('active', litClass);

                        // Set probability
                        if (inputType === 'real') {
                            setOutput(0.85 + Math.random() * 0.13); // 85-98% real
                        } else {
                            setOutput(0.05 + Math.random() * 0.2);  // 5-25% real → fake
                        }
                        isRunning = false;
                    }, 300);
                });
            }, 1400);
        }

        // --- Toggle handlers ---
        function setToggle(type) {
            inputType = type;
            if (toggleReal) toggleReal.classList.toggle('ditoggle-btn--active', type === 'real');
            if (toggleFake) toggleFake.classList.toggle('ditoggle-btn--active', type === 'fake');
        }

        if (toggleReal) toggleReal.addEventListener('click', () => setToggle('real'));
        if (toggleFake) toggleFake.addEventListener('click', () => setToggle('fake'));
        runBtn.addEventListener('click', runEvaluation);

        // Draw initial input
        drawInput();
    }

    /* ============================================================
       7. LOSS GAME — Adversarial tug-of-war + dual loss chart
       ============================================================ */
    function initLossGame() {
        const canvas    = $('#lossgame-chart');
        const playBtn   = $('#lg-play-btn');
        const resetBtn  = $('#lg-reset-btn');
        const speedEl   = $('#lg-speed');
        const epochEl   = $('#lossgame-epoch');
        const towKnot   = $('#tow-knot');
        const towGenLoss  = $('#tow-gen-loss');
        const towDiscLoss = $('#tow-disc-loss');
        const towGenAvatar  = $('#tow-gen-avatar');
        const towDiscAvatar = $('#tow-disc-avatar');

        if (!canvas || !playBtn) return;
        const ctx = canvas.getContext('2d');

        const MAX_EPOCHS = 300;
        let epoch = 0;
        let running = false;
        let interval = null;

        // Loss state
        let gLoss = 2.5;   // Generator starts high
        let dLoss = 0.7;   // Discriminator starts low
        const history = { g: [], d: [] };

        // Adversarial dynamics simulation
        function step() {
            if (epoch >= MAX_EPOCHS) {
                running = false;
                clearInterval(interval);
                if (playBtn) playBtn.textContent = '✓ Converged';
                return;
            }
            epoch++;

            // Phase-based training dynamics for realistic curves
            const t = epoch / MAX_EPOCHS;
            const phase = Math.sin(t * Math.PI * 6) * 0.12;         // Oscillation
            const drift = Math.sin(t * Math.PI * 2.5) * 0.08;
            const noise = (Math.random() - 0.5) * 0.06;

            // Generator: starts high, drops with oscillation, converges ~0.7
            const gTarget = 2.5 * (1 - t * 0.7) * Math.exp(-t * 1.2) + 0.65;
            gLoss = gLoss + (gTarget - gLoss) * 0.08 + phase + noise;
            gLoss = Math.max(0.3, Math.min(3, gLoss));

            // Discriminator: starts low, rises as G improves, then settles ~0.7
            const dTarget = 0.7 + 0.4 * Math.sin(t * Math.PI * 1.5) * (1 - t * 0.5) + 0.1 * t;
            dLoss = dLoss + (dTarget - dLoss) * 0.08 - phase * 0.7 + drift + noise;
            dLoss = Math.max(0.2, Math.min(2.5, dLoss));

            // At convergence, both approach ~0.69 (log(2))
            if (t > 0.8) {
                const conv = (t - 0.8) / 0.2;
                gLoss = lerp(gLoss, 0.693 + noise * 0.5, conv * 0.15);
                dLoss = lerp(dLoss, 0.693 + noise * 0.5, conv * 0.15);
            }

            history.g.push(gLoss);
            history.d.push(dLoss);

            updateUI();
        }

        // --- Draw chart ---
        function drawChart() {
            const W = canvas.width, H = canvas.height;
            const dpr = window.devicePixelRatio || 1;
            // Set hi-dpi
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = 300 * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            const w = rect.width, h = 300;

            ctx.clearRect(0, 0, w, h);

            const pad = { top: 20, right: 20, bottom: 36, left: 50 };
            const chartW = w - pad.left - pad.right;
            const chartH = h - pad.top - pad.bottom;
            const maxLoss = 3;

            // Background grid
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = pad.top + (chartH / 5) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(pad.left + chartW, y);
                ctx.stroke();
            }

            // Equilibrium line at log(2) ≈ 0.693
            const eqY = pad.top + chartH - (0.693 / maxLoss) * chartH;
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(pad.left, eqY);
            ctx.lineTo(pad.left + chartW, eqY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Equilibrium label
            ctx.font = '500 9px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.textAlign = 'left';
            ctx.fillText('log(2) ≈ 0.693', pad.left + chartW - 80, eqY - 5);

            // Axis labels
            ctx.font = '600 10px Inter, sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.textAlign = 'center';
            ctx.fillText('Epoch', pad.left + chartW / 2, h - 4);

            ctx.save();
            ctx.translate(12, pad.top + chartH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Loss', 0, 0);
            ctx.restore();

            // Y-axis tick labels
            ctx.font = '500 9px JetBrains Mono, monospace';
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {
                const val = (maxLoss / 5 * (5 - i)).toFixed(1);
                const y = pad.top + (chartH / 5) * i;
                ctx.fillText(val, pad.left - 6, y + 3);
            }

            // X-axis tick labels
            ctx.textAlign = 'center';
            for (let i = 0; i <= 4; i++) {
                const ep = Math.floor((MAX_EPOCHS / 4) * i);
                const x = pad.left + (chartW / 4) * i;
                ctx.fillText(ep, x, h - 18);
            }

            if (history.g.length < 2) return;

            const pts = history.g.length;
            const xStep = chartW / MAX_EPOCHS;

            // Draw Generator loss
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(232,67,147,0.9)';
            ctx.lineWidth = 2;
            ctx.shadowBlur = 6;
            ctx.shadowColor = 'rgba(232,67,147,0.4)';
            for (let i = 0; i < pts; i++) {
                const x = pad.left + i * xStep;
                const y = pad.top + chartH - (history.g[i] / maxLoss) * chartH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Draw Discriminator loss
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0,206,201,0.9)';
            ctx.lineWidth = 2;
            ctx.shadowBlur = 6;
            ctx.shadowColor = 'rgba(0,206,201,0.4)';
            for (let i = 0; i < pts; i++) {
                const x = pad.left + i * xStep;
                const y = pad.top + chartH - (history.d[i] / maxLoss) * chartH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Current value dots
            if (pts > 0) {
                const lastX = pad.left + (pts - 1) * xStep;
                // G dot
                const gY = pad.top + chartH - (history.g[pts-1] / maxLoss) * chartH;
                ctx.beginPath();
                ctx.arc(lastX, gY, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#e84393';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(lastX, gY, 8, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(232,67,147,0.2)';
                ctx.fill();

                // D dot
                const dY = pad.top + chartH - (history.d[pts-1] / maxLoss) * chartH;
                ctx.beginPath();
                ctx.arc(lastX, dY, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#00cec9';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(lastX, dY, 8, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0,206,201,0.2)';
                ctx.fill();
            }
        }

        // --- Update tug-of-war + UI ---
        function updateUI() {
            // Epoch
            if (epochEl) epochEl.textContent = `Epoch: ${epoch} / ${MAX_EPOCHS}`;

            // Loss values
            if (towGenLoss) towGenLoss.textContent = `Loss: ${gLoss.toFixed(3)}`;
            if (towDiscLoss) towDiscLoss.textContent = `Loss: ${dLoss.toFixed(3)}`;

            // Tug-of-war knot position: compare relative losses
            // When G loss > D loss → D is winning → knot moves right
            // When D loss > G loss → G is winning → knot moves left
            const diff = gLoss - dLoss;       // positive = D winning
            const maxDiff = 2;
            const ratio = Math.max(-1, Math.min(1, diff / maxDiff));
            const knotPos = 50 + ratio * 40;   // 10% – 90%
            if (towKnot) towKnot.style.left = knotPos + '%';

            // Avatar glow
            if (towGenAvatar && towDiscAvatar) {
                towGenAvatar.classList.toggle('winning', ratio < -0.1);
                towDiscAvatar.classList.toggle('winning', ratio > 0.1);
            }

            drawChart();
        }

        // --- Controls ---
        function start() {
            if (running) {
                running = false;
                clearInterval(interval);
                playBtn.textContent = '▶ Resume';
                return;
            }
            running = true;
            playBtn.textContent = '⏸ Pause';
            const speed = speedEl ? parseInt(speedEl.value) : 3;
            const delays = [120, 80, 50, 30, 15];
            interval = setInterval(step, delays[speed - 1] || 50);
        }

        function reset() {
            running = false;
            clearInterval(interval);
            epoch = 0;
            gLoss = 2.5;
            dLoss = 0.7;
            history.g.length = 0;
            history.d.length = 0;
            playBtn.textContent = '▶ Start Training';
            updateUI();
        }

        playBtn.addEventListener('click', start);
        if (resetBtn) resetBtn.addEventListener('click', reset);

        // Speed change mid-run
        if (speedEl) {
            speedEl.addEventListener('input', () => {
                if (running) {
                    clearInterval(interval);
                    const speed = parseInt(speedEl.value);
                    const delays = [120, 80, 50, 30, 15];
                    interval = setInterval(step, delays[speed - 1] || 50);
                }
            });
        }

        // Initial draw
        updateUI();
    }

    /* ============================================================
       8. TRAINING CYCLE — Step-by-step circular flow
       ============================================================ */
    function initTrainCycle() {
        const nextBtn  = $('#tc-next-btn');
        const autoBtn  = $('#tc-auto-btn');
        const resetBtn = $('#tc-reset-btn');
        if (!nextBtn) return;

        const STEPS = [
            {
                title: 'Step 1: Generate Fake Image',
                desc: 'The Generator takes a <strong>random noise vector z</strong> and passes it through its neural network to produce a fake image. At early stages, this looks like pure noise.',
                panelClass: 'step-gen',
                arcColor: 'rgba(232,67,147,0.7)',
                centerLabel: 'Generating…'
            },
            {
                title: 'Step 2: Discriminator Evaluates',
                desc: 'The Discriminator receives <strong>both real images from the training set and fake images from the Generator</strong>. It outputs a probability P(real) for each sample.',
                panelClass: 'step-disc',
                arcColor: 'rgba(0,206,201,0.7)',
                centerLabel: 'Evaluating…'
            },
            {
                title: 'Step 3: Compute Loss',
                desc: 'Both networks compute their <strong>loss using binary cross-entropy</strong>. D wants to maximize log(D(x)) + log(1-D(G(z))), while G wants to minimize log(1-D(G(z))).',
                panelClass: 'step-loss',
                arcColor: 'rgba(255,107,107,0.7)',
                centerLabel: 'Computing…'
            },
            {
                title: 'Step 4: Update Discriminator',
                desc: 'Backpropagate through D and <strong>update its weights via gradient descent</strong>. This makes D better at telling real from fake — but only D\'s weights are updated here.',
                panelClass: 'step-disc',
                arcColor: 'rgba(0,206,201,0.7)',
                centerLabel: 'Updating D…'
            },
            {
                title: 'Step 5: Update Generator',
                desc: 'Backpropagate through G (while <strong>freezing D\'s weights</strong>) and update G\'s weights. This makes G better at fooling D. The cycle then repeats from Step 1.',
                panelClass: 'step-gen',
                arcColor: 'rgba(232,67,147,0.7)',
                centerLabel: 'Updating G…'
            },
        ];

        let currentStep = -1;  // -1 = idle
        let iteration = 0;
        let autoPlaying = false;
        let autoInterval = null;

        const nodes = [];
        for (let i = 0; i < 5; i++) {
            nodes.push(document.getElementById('tc-node-' + i));
        }
        const arcs = [];
        for (let i = 1; i <= 5; i++) {
            arcs.push(document.getElementById('tc-arc-' + i));
        }

        const iterNum     = $('#tc-iter-num');
        const stepTitle   = $('#tc-step-title');
        const descPanel   = $('#tc-desc-panel');
        const descTitleEl = $('#tc-desc-title');
        const descBodyEl  = $('#tc-desc-body');
        const descIconEl  = $('#tc-desc-icon');

        function updateDisplay() {
            // Clear all nodes
            nodes.forEach(n => {
                if (n) n.classList.remove('active', 'completed');
            });
            arcs.forEach(a => {
                if (a) {
                    a.classList.remove('active');
                    a.style.stroke = '';
                }
            });

            if (currentStep < 0) {
                // Idle state
                if (stepTitle) stepTitle.textContent = 'Ready to begin';
                if (iterNum) iterNum.textContent = iteration;
                if (descPanel) descPanel.className = 'tc-desc-panel';
                if (descTitleEl) descTitleEl.textContent = 'GAN Training Cycle';
                if (descBodyEl) descBodyEl.innerHTML = 'Click <strong>Next Step</strong> or <strong>Auto-Play</strong> to walk through each stage of a single GAN training iteration. The cycle repeats until both networks converge.';
                return;
            }

            const step = STEPS[currentStep];

            // Mark completed nodes
            for (let i = 0; i < currentStep; i++) {
                if (nodes[i]) nodes[i].classList.add('completed');
            }

            // Mark active node
            if (nodes[currentStep]) nodes[currentStep].classList.add('active');

            // Highlight arc leading to current step
            if (currentStep > 0 && arcs[currentStep - 1]) {
                arcs[currentStep - 1].classList.add('active');
                arcs[currentStep - 1].style.stroke = step.arcColor;
            }
            // Also highlight arc from previous step to show flow
            if (currentStep === 0 && iteration > 0 && arcs[4]) {
                arcs[4].classList.add('active');
                arcs[4].style.stroke = STEPS[4].arcColor;
            }

            // Center
            if (stepTitle) stepTitle.textContent = step.centerLabel;
            if (iterNum) iterNum.textContent = iteration;

            // Description panel
            if (descPanel) descPanel.className = 'tc-desc-panel ' + step.panelClass;
            if (descTitleEl) descTitleEl.textContent = step.title;
            if (descBodyEl) descBodyEl.innerHTML = step.desc;
        }

        function advanceStep() {
            currentStep++;
            if (currentStep >= STEPS.length) {
                // Complete one iteration → loop
                currentStep = 0;
                iteration++;
            }
            updateDisplay();
        }

        function reset() {
            stopAuto();
            currentStep = -1;
            iteration = 0;
            updateDisplay();
            if (nextBtn) nextBtn.textContent = '→ Next Step';
        }

        function startAuto() {
            autoPlaying = true;
            autoBtn.textContent = '⏸ Pause';
            autoInterval = setInterval(() => {
                advanceStep();
            }, 1800);
        }

        function stopAuto() {
            autoPlaying = false;
            clearInterval(autoInterval);
            if (autoBtn) autoBtn.textContent = '▶ Auto-Play';
        }

        // Events
        nextBtn.addEventListener('click', advanceStep);

        if (autoBtn) {
            autoBtn.addEventListener('click', () => {
                if (autoPlaying) {
                    stopAuto();
                } else {
                    startAuto();
                }
            });
        }

        if (resetBtn) resetBtn.addEventListener('click', reset);

        // Click on nodes to jump
        nodes.forEach((node, i) => {
            if (node) {
                node.addEventListener('click', () => {
                    currentStep = i;
                    if (currentStep === 0 && iteration === 0) iteration = 1;
                    updateDisplay();
                });
            }
        });

        // Init
        updateDisplay();
    }

    /* ============================================================
       9. EPOCH PROGRESSION
       ============================================================ */
    function initEpochProg() {
        const slider = $('#ep-slider'), sliderVal = $('#ep-slider-val'), badge = $('#ep-badge');
        const qualFill = $('#ep-quality-fill'), qualVal = $('#ep-quality-val');
        const foolFill = $('#ep-fool-fill'), foolVal = $('#ep-fool-val');
        const descEl = $('#ep-stage-desc'), animBtn = $('#ep-animate-btn');
        if (!slider) return;

        const DIGIT = [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0];
        const STAGES = [
            {e:1,  q:2,  f:3,  d:'Pure random noise — the Generator has not learned any structure yet.'},
            {e:10, q:18, f:12, d:'Blurry shapes emerging — the Generator begins to learn basic spatial patterns.'},
            {e:50, q:62, f:48, d:'Clearer digit forms — recognizable structure with some artifacts remaining.'},
            {e:100,q:95, f:88, d:'Near-realistic output — the Generator can fool the Discriminator most of the time.'}
        ];

        function drawAtEpoch(canvasId, epoch, w, h) {
            const c = document.getElementById(canvasId); if (!c) return;
            const ctx = c.getContext('2d'); ctx.clearRect(0, 0, w, h);
            const t = Math.min(1, epoch / 100), cell = w / 8;
            for (let i = 0; i < 64; i++) {
                const row = Math.floor(i/8), col = i%8, on = DIGIT[i]===1;
                const clarity = t * t; // quadratic improvement
                const noise = (1 - clarity) * Math.random();
                if (on) {
                    const v = Math.floor(40 + clarity * 215 + noise * 40);
                    const tint = (1-clarity)*80; // color noise at early epochs
                    ctx.fillStyle = `rgb(${v-tint*0.3},${v-tint*0.1},${v+tint*0.2})`;
                } else {
                    const v = Math.floor(noise * (120 - clarity*100) + clarity*5);
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                }
                ctx.fillRect(col*cell, row*cell, cell, cell);
            }
        }

        function getStageDesc(epoch) {
            if (epoch <= 1) return STAGES[0]; if (epoch <= 10) return STAGES[1];
            if (epoch <= 50) return STAGES[2]; return STAGES[3];
        }

        function update(epoch) {
            drawAtEpoch('ep-main-canvas', epoch, 160, 160);
            if (badge) badge.textContent = 'Epoch ' + epoch;
            if (sliderVal) sliderVal.textContent = epoch;
            const t = Math.min(1, epoch/100), q = Math.floor(2+93*t*t), f = Math.floor(3+85*t*t*0.9);
            if (qualFill) qualFill.style.width = q+'%'; if (qualVal) qualVal.textContent = q+'%';
            if (foolFill) foolFill.style.width = f+'%'; if (foolVal) foolVal.textContent = f+'%';
            if (descEl) descEl.textContent = getStageDesc(epoch).d;
            // Milestone highlights
            const thresholds = [1,10,50,100];
            for (let i = 0; i < 4; i++) {
                const m = document.getElementById('ep-mile-'+i);
                if (m) m.classList.toggle('ep-mile--active', epoch >= thresholds[i]);
            }
        }

        // Draw milestones
        [1,10,50,100].forEach((e,i) => drawAtEpoch('ep-mile-c'+i, e, 80, 80));

        slider.addEventListener('input', () => update(parseInt(slider.value)));

        let animId = null, animRunning = false;
        if (animBtn) animBtn.addEventListener('click', () => {
            if (animRunning) { animRunning = false; clearInterval(animId); animBtn.textContent = '▶ Animate'; return; }
            animRunning = true; animBtn.textContent = '⏸ Stop';
            slider.value = 1;
            animId = setInterval(() => {
                let v = parseInt(slider.value) + 1;
                if (v > 100) { animRunning = false; clearInterval(animId); animBtn.textContent = '▶ Animate'; return; }
                slider.value = v; update(v);
            }, 60);
        });
        update(1);
    }

    /* ============================================================
       10. LATENT SPACE EXPLORER
       ============================================================ */
    function initLatentSpace() {
        const grid = document.getElementById('latent-grid');
        const output = document.getElementById('latent-output');
        const coordsEl = $('#latent-coords'), interpBtn = $('#latent-interp-btn');
        if (!grid || !output) return;
        const gCtx = grid.getContext('2d'), oCtx = output.getContext('2d');

        const DIGITS = [
            [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0],
            [0,1,1,1,1,1,0,0, 1,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,1,0,0,0,0,0,0, 1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,0, 0,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0],
        ];

        // Generate image from z coords
        function genFromZ(z1, z2, ctx, w, h) {
            ctx.clearRect(0, 0, w, h);
            const idx1 = Math.abs(Math.floor(z1*2)) % DIGITS.length;
            const idx2 = Math.abs(Math.floor(z2*2)) % DIGITS.length;
            const blend = (Math.abs(z1*2) % 1); const cell = w/8;
            const p1 = DIGITS[idx1], p2 = DIGITS[idx2];
            for (let i = 0; i < 64; i++) {
                const r = Math.floor(i/8), c = i%8;
                const v1 = p1[i], v2 = p2[i], val = v1*(1-blend)+v2*blend;
                const bright = Math.floor(val * 220 + (1-val)*10 + (Math.random()-0.5)*15);
                const hue = (z1+1)*60; const sat = 5+val*10;
                ctx.fillStyle = `hsl(${hue},${sat}%,${Math.max(2,Math.min(95,bright/2.55))}%)`;
                ctx.fillRect(c*cell, r*cell, cell, cell);
            }
        }

        // Draw grid
        function drawGrid() {
            const s = 300, cells = 10, cellSize = s/cells;
            gCtx.clearRect(0, 0, s, s);
            for (let r = 0; r < cells; r++) for (let c = 0; c < cells; c++) {
                const z1 = (c/cells)*4-2, z2 = (r/cells)*4-2;
                const idx = Math.abs(Math.floor(z1*2+z2)) % DIGITS.length;
                const bright = 15 + Math.abs(Math.sin(z1*2+z2*3))*40;
                const hue = ((z1+2)*45) % 360;
                gCtx.fillStyle = `hsl(${hue},20%,${bright}%)`;
                gCtx.fillRect(c*cellSize, r*cellSize, cellSize-1, cellSize-1);
            }
            // Grid lines
            gCtx.strokeStyle = 'rgba(255,255,255,0.04)'; gCtx.lineWidth = 1;
            for (let i = 0; i <= cells; i++) {
                gCtx.beginPath(); gCtx.moveTo(i*cellSize,0); gCtx.lineTo(i*cellSize,s); gCtx.stroke();
                gCtx.beginPath(); gCtx.moveTo(0,i*cellSize); gCtx.lineTo(s,i*cellSize); gCtx.stroke();
            }
            // Axis labels
            gCtx.font = '500 9px Inter'; gCtx.fillStyle = 'rgba(255,255,255,0.3)'; gCtx.textAlign = 'center';
            gCtx.fillText('z₁', s/2, s-3); gCtx.save(); gCtx.translate(8, s/2); gCtx.rotate(-Math.PI/2);
            gCtx.fillText('z₂', 0, 0); gCtx.restore();
        }

        grid.addEventListener('click', e => {
            const rect = grid.getBoundingClientRect();
            const z1 = ((e.clientX-rect.left)/rect.width)*4-2;
            const z2 = ((e.clientY-rect.top)/rect.height)*4-2;
            genFromZ(z1, z2, oCtx, 120, 120);
            if (coordsEl) coordsEl.textContent = `z = [${z1.toFixed(2)}, ${z2.toFixed(2)}]`;
        });

        function drawInterp() {
            const z1a = Math.random()*4-2, z2a = Math.random()*4-2;
            const z1b = Math.random()*4-2, z2b = Math.random()*4-2;
            for (let i = 0; i < 5; i++) {
                const t = i/4;
                const z1 = z1a*(1-t)+z1b*t, z2 = z2a*(1-t)+z2b*t;
                const c = document.getElementById('interp-'+i);
                if (c) genFromZ(z1, z2, c.getContext('2d'), 48, 48);
            }
        }

        if (interpBtn) interpBtn.addEventListener('click', drawInterp);
        drawGrid(); genFromZ(0, 0, oCtx, 120, 120); drawInterp();
    }

    /* ============================================================
       11. BATTLE MODE
       ============================================================ */
    function initBattleMode() {
        const startBtn = $('#bm-start-btn'), resetBtn = $('#bm-reset-btn');
        const genPct = $('#bm-gen-pct'), discPct = $('#bm-disc-pct');
        const genBar = $('#bm-gen-bar'), discBar = $('#bm-disc-bar');
        const genSide = $('#bm-gen-side'), discSide = $('#bm-disc-side');
        const msgEl = $('#bm-msg'), subEl = $('#bm-sub');
        if (!startBtn) return;

        let running = false, interval = null, round = 0;
        let gRate = 15, dRate = 85; // starting rates

        const DIGIT = [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0];

        function drawSample(id, quality, isReal) {
            const c = document.getElementById(id); if (!c) return;
            const ctx = c.getContext('2d'); ctx.clearRect(0, 0, 48, 48);
            const cell = 6;
            for (let i = 0; i < 64; i++) {
                const r = Math.floor(i/8), co = i%8, on = DIGIT[i]===1;
                if (isReal) {
                    const v = on ? 200+Math.random()*55 : 5+Math.random()*15;
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                } else {
                    const clarity = quality/100, noise = (1-clarity)*Math.random();
                    const v = on ? 40+clarity*200+noise*30 : noise*(100-clarity*80);
                    const tint = (1-clarity)*50;
                    ctx.fillStyle = `rgb(${Math.min(255,v+tint*0.3)},${Math.min(255,v-tint*0.2)},${Math.min(255,v+tint*0.1)})`;
                }
                ctx.fillRect(co*cell, r*cell, cell, cell);
            }
        }

        function updateUI() {
            if (genPct) genPct.textContent = Math.round(gRate)+'%';
            if (discPct) discPct.textContent = Math.round(dRate)+'%';
            if (genBar) genBar.style.width = gRate+'%';
            if (discBar) discBar.style.width = dRate+'%';
            if (subEl) subEl.textContent = 'Round '+round;
            if (genSide) genSide.classList.toggle('winning', gRate > dRate);
            if (discSide) discSide.classList.toggle('winning', dRate > gRate);

            // Feedback
            if (msgEl) {
                if (gRate > dRate + 10) { msgEl.textContent = '🔥 Generator is dominating!'; msgEl.style.color = '#e84393'; }
                else if (dRate > gRate + 10) { msgEl.textContent = '🛡 Discriminator is dominating!'; msgEl.style.color = '#00cec9'; }
                else if (Math.abs(gRate-dRate) < 5) { msgEl.textContent = '⚖ Nash Equilibrium reached!'; msgEl.style.color = '#00b894'; }
                else if (gRate > 50) { msgEl.textContent = '📈 Generator is improving!'; msgEl.style.color = '#e84393'; }
                else { msgEl.textContent = '🔍 Discriminator is catching fakes'; msgEl.style.color = '#00cec9'; }
            }

            // Draw samples
            for (let i = 0; i < 3; i++) { drawSample('bm-gen-s'+i, gRate, false); drawSample('bm-disc-s'+i, 95, true); }
        }

        function step() {
            round++;
            const t = Math.min(1, round/80);
            const osc = Math.sin(t*Math.PI*5)*8;
            const noise = (Math.random()-0.5)*6;
            // G improves over time, D fights back
            gRate = Math.max(5, Math.min(95, 15 + t*60 + osc + noise));
            dRate = Math.max(5, Math.min(95, 85 - t*35 - osc*0.7 + noise));
            // Converge toward 50/50
            if (t > 0.7) { const c = (t-0.7)/0.3; gRate = lerp(gRate, 52+noise, c*0.2); dRate = lerp(dRate, 48+noise, c*0.2); }
            updateUI();
        }

        startBtn.addEventListener('click', () => {
            if (running) { running = false; clearInterval(interval); startBtn.textContent = '⚔ Resume'; return; }
            running = true; startBtn.textContent = '⏸ Pause';
            interval = setInterval(step, 300);
        });
        if (resetBtn) resetBtn.addEventListener('click', () => {
            running = false; clearInterval(interval); round = 0; gRate = 15; dRate = 85;
            startBtn.textContent = '⚔ Start Battle';
            if (msgEl) { msgEl.textContent = 'Press Start to begin the battle'; msgEl.style.color = ''; }
            updateUI();
        });
        updateUI();
    }

    /* ============================================================
       12. FINAL OUTPUT GALLERY
       ============================================================ */
    function initFinalOutput() {
        const grid = $('#fo-grid'), newBtn = $('#fo-new-btn');
        const btnGen = $('#fo-btn-gen'), btnCompare = $('#fo-btn-compare');
        if (!grid) return;

        const DIGITS = [
            [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],
            [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0],
            [0,1,1,1,1,1,0,0, 1,0,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,1,0,0,0,0,0,0, 1,1,1,1,1,1,1,0],
            [0,0,0,0,1,1,0,0, 0,0,0,1,0,1,0,0, 0,0,1,0,0,1,0,0, 0,1,0,0,0,1,0,0, 1,1,1,1,1,1,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0],
        ];
        let mode = 'gen'; // 'gen' | 'compare'

        function drawCard(canvas, pattern, isReal) {
            const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h); const cell = w/8;
            for (let i = 0; i < 64; i++) {
                const r = Math.floor(i/8), c = i%8, on = pattern[i]===1;
                if (isReal) {
                    const v = on ? 200+Math.random()*55 : 5+Math.random()*15;
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                } else {
                    const v = on ? 180+Math.random()*70 : 8+Math.random()*20;
                    const tint = Math.random()*12;
                    ctx.fillStyle = `rgb(${Math.min(255,v+tint)},${Math.min(255,v-tint*0.5)},${Math.min(255,v+tint*0.3)})`;
                }
                ctx.fillRect(c*cell, r*cell, cell, cell);
            }
        }

        function generate() {
            grid.innerHTML = '';
            const count = 8;
            for (let i = 0; i < count; i++) {
                const isReal = mode === 'compare' && i % 2 === 0;
                const card = document.createElement('div');
                card.className = 'fo-card ' + (isReal ? 'fo-card--real' : 'fo-card--fake');
                const canvas = document.createElement('canvas');
                canvas.width = 80; canvas.height = 80;
                const tag = document.createElement('span');
                tag.className = 'fo-card__tag ' + (isReal ? 'fo-card__tag--real' : 'fo-card__tag--fake');
                tag.textContent = isReal ? 'REAL' : 'FAKE';
                card.appendChild(canvas); card.appendChild(tag);
                grid.appendChild(card);
                const pat = DIGITS[Math.floor(Math.random()*DIGITS.length)];
                drawCard(canvas, pat, isReal);
            }
        }

        if (btnGen) btnGen.addEventListener('click', () => { mode = 'gen'; btnGen.classList.add('fo-toggle-btn--active'); btnCompare.classList.remove('fo-toggle-btn--active'); generate(); });
        if (btnCompare) btnCompare.addEventListener('click', () => { mode = 'compare'; btnCompare.classList.add('fo-toggle-btn--active'); btnGen.classList.remove('fo-toggle-btn--active'); generate(); });
        if (newBtn) newBtn.addEventListener('click', generate);
        generate();
    }

    /* ============================================================
       13. INTERACTIVE GAN PIPELINE
       ============================================================ */
    function initGanPipeline() {
        const drawCanvas = document.getElementById('gp-draw-canvas');
        const processBtn = $('#gp-process-btn'), clearBtn = $('#gp-clear-btn');
        const statusEl = $('#gp-model-status');
        const restartBtn = $('#gp-restart-btn'), randomBtn = $('#gp-random-btn');
        const epochSlider = $('#gp-epoch-slider'), epochVal = $('#gp-epoch-val');
        if (!drawCanvas || !processBtn) return;
        const drawCtx = drawCanvas.getContext('2d');

        let currentStep = 0, genModel = null, latentVector = null, detectedDigit = 0, classifierModel = null;
        const LATENT_DIM = 100;

        function getCenteredMnistCanvas(sourceCanvas) {
            const w = sourceCanvas.width, h = sourceCanvas.height;
            const ctx = sourceCanvas.getContext('2d');
            const imgData = ctx.getImageData(0, 0, w, h);
            const data = imgData.data;
            let minX = w, minY = h, maxX = -1, maxY = -1, empty = true;
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    if (data[(y * w + x) * 4] > 20) {
                        if (x < minX) minX = x; if (x > maxX) maxX = x;
                        if (y < minY) minY = y; if (y > maxY) maxY = y;
                        empty = false;
                    }
                }
            }
            const outCanvas = document.createElement('canvas');
            outCanvas.width = 28; outCanvas.height = 28;
            const outCtx = outCanvas.getContext('2d');
            outCtx.fillStyle = '#000'; outCtx.fillRect(0, 0, 28, 28);
            if (empty) return outCanvas;
            const boxW = maxX - minX + 1, boxH = maxY - minY + 1;
            const scale = 20.0 / Math.max(boxW, boxH);
            const scaledW = boxW * scale, scaledH = boxH * scale;
            const dx = (28 - scaledW) / 2, dy = (28 - scaledH) / 2;
            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = boxW; cropCanvas.height = boxH;
            cropCanvas.getContext('2d').putImageData(ctx.getImageData(minX, minY, boxW, boxH), 0, 0);
            outCtx.imageSmoothingEnabled = true;
            outCtx.drawImage(cropCanvas, dx, dy, scaledW, scaledH);
            return outCanvas;
        }

        // Real CNN-based digit detection
        async function detectDigit() {
            if (!classifierModel) return Math.floor(Math.random() * 10);
            
            try {
                const centeredCanvas = getCenteredMnistCanvas(drawCanvas);
                const tensor = tf.tidy(() => {
                    let t = tf.browser.fromPixels(centeredCanvas, 1);
                    t = t.toFloat().div(255);
                    const maxVal = t.max();
                    return t.div(maxVal.add(1e-5)).expandDims(0);
                });
                
                const output = classifierModel.predict(tensor);
                const data = await output.data();
                tf.dispose([tensor, output]);
                
                let maxProb = -1;
                let maxIdx = 0;
                for (let i = 0; i < 10; i++) {
                    if (data[i] > maxProb) { maxProb = data[i]; maxIdx = i; }
                }
                return maxIdx;
            } catch (e) {
                console.warn('CNN detection failed:', e);
                return Math.floor(Math.random() * 10);
            }
        }

        // --- Drawing ---
        let drawing = false;
        drawCtx.fillStyle = '#0a0a0a';
        drawCtx.fillRect(0, 0, 280, 280);
        drawCtx.lineWidth = 14;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.strokeStyle = '#fff';

        function getPos(e) {
            const r = drawCanvas.getBoundingClientRect();
            const t = e.touches ? e.touches[0] : e;
            return { x: (t.clientX - r.left) * (280/r.width), y: (t.clientY - r.top) * (280/r.height) };
        }
        drawCanvas.addEventListener('mousedown', e => { drawing = true; const p = getPos(e); drawCtx.beginPath(); drawCtx.moveTo(p.x, p.y); });
        drawCanvas.addEventListener('mousemove', e => { if (!drawing) return; const p = getPos(e); drawCtx.lineTo(p.x, p.y); drawCtx.stroke(); });
        drawCanvas.addEventListener('mouseup', () => drawing = false);
        drawCanvas.addEventListener('mouseleave', () => drawing = false);
        drawCanvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; const p = getPos(e); drawCtx.beginPath(); drawCtx.moveTo(p.x, p.y); }, {passive:false});
        drawCanvas.addEventListener('touchmove', e => { e.preventDefault(); if (!drawing) return; const p = getPos(e); drawCtx.lineTo(p.x, p.y); drawCtx.stroke(); }, {passive:false});
        drawCanvas.addEventListener('touchend', () => drawing = false);

        if (clearBtn) clearBtn.addEventListener('click', () => { drawCtx.fillStyle = '#0a0a0a'; drawCtx.fillRect(0, 0, 280, 280); });

        // --- Upload zone: click-to-browse + drag-and-drop ---
        const uploadZone = document.getElementById('upload-zone');
        const uploadInput = document.getElementById('gp-upload-input');

        function handleUploadedFile(file) {
            if (!file || !file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                const img = new Image();
                img.onload = () => {
                    drawCtx.fillStyle = '#0a0a0a';
                    drawCtx.fillRect(0, 0, 280, 280);
                    drawCtx.drawImage(img, 0, 0, 280, 280);
                    // Show preview text on upload zone
                    if (uploadZone) {
                        uploadZone.innerHTML = '<div style="text-align:center;"><span style="color:#00b894; font-size:14px; font-weight:600;">✓ Image Loaded</span><br><span style="color:rgba(255,255,255,0.5); font-size:12px; margin-top:4px; display:inline-block;">Click "Process with GAN" to continue</span></div>';
                    }
                };
                img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
        }

        if (uploadZone && uploadInput) {
            // Click to browse
            uploadZone.addEventListener('click', () => uploadInput.click());

            // File input change
            uploadInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) handleUploadedFile(e.target.files[0]);
            });

            // Drag-and-drop
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.style.borderColor = 'rgba(0,184,148,0.6)';
                uploadZone.style.background = 'rgba(0,184,148,0.05)';
            });
            uploadZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadZone.style.borderColor = 'rgba(255,255,255,0.2)';
                uploadZone.style.background = 'rgba(0,0,0,0.2)';
            });
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.style.borderColor = 'rgba(255,255,255,0.2)';
                uploadZone.style.background = 'rgba(0,0,0,0.2)';
                if (e.dataTransfer.files && e.dataTransfer.files[0]) handleUploadedFile(e.dataTransfer.files[0]);
            });
        }
        // --- Step management ---
        function setStep(n) {
            currentStep = n;
            for (let i = 0; i < 5; i++) {
                const s = document.getElementById('gp-s'+i), p = document.getElementById('gp-panel-'+i);
                if (s) { s.classList.remove('gp-step--active','gp-step--done'); if (i < n) s.classList.add('gp-step--done'); if (i === n) s.classList.add('gp-step--active'); }
                if (p) p.classList.toggle('gp-panel--active', i === n);
            }
            $$('.gp-step-line').forEach((l, i) => l.classList.toggle('done', i < n));
        }

        // --- Model loading (local files for speed) ---
        async function loadModel() {
            // Load ACGAN Generator from local models/ directory
            if (!genModel) {
                if (statusEl) statusEl.textContent = 'Model: loading…';
                try {
                    genModel = await tf.loadLayersModel('models/acgan_model.json');
                    if (statusEl) statusEl.textContent = 'Model: ✓ ACGAN loaded (local)';
                } catch(e) {
                    console.warn('Local GAN model failed, trying remote…', e);
                    try {
                        genModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-examples/mnist-acgan/dist/generator/model.json');
                        if (statusEl) statusEl.textContent = 'Model: ✓ ACGAN loaded (remote)';
                    } catch(e2) {
                        console.warn('GAN model failed to load:', e2);
                        if (statusEl) statusEl.textContent = 'Model: load failed — check connection';
                    }
                }
            }
            // Classifier not available — use target digit selector or random fallback
            // (the old GitHub URL 404s, so we rely on the dropdown selector)
        }

        // --- Preprocessing (Step 2) ---
        async function runPreprocess() {
            setStep(1);
            // Original thumbnail
            const orig = document.getElementById('gp-orig');
            if (orig) { const oc = orig.getContext('2d'); oc.drawImage(drawCanvas, 0, 0, 80, 80); }
            // Processed 28x28
            const proc = document.getElementById('gp-processed');
            if (proc) {
                const pc = proc.getContext('2d');
                const centeredCanvas = getCenteredMnistCanvas(drawCanvas);
                pc.imageSmoothingEnabled = false;
                pc.drawImage(centeredCanvas, 0, 0, 80, 80);
            }
            
            const targetSelect = document.getElementById('gp-target-digit');
            if (targetSelect && targetSelect.value !== 'auto') {
                detectedDigit = parseInt(targetSelect.value);
            } else {
                detectedDigit = await detectDigit();
            }
            // Cascade to final step
            setTimeout(runEncode, 1000);
        }

        // --- Latent encoding (Step 3) ---
        function runEncode() {
            setStep(2);
            const centeredCanvas = getCenteredMnistCanvas(drawCanvas);
            const src = document.getElementById('gp-latent-src');
            if (src) { const sc = src.getContext('2d'); sc.imageSmoothingEnabled = false; sc.drawImage(centeredCanvas, 0, 0, 80, 80); }
            // Generate latent vector from pixel data
            const pixels = centeredCanvas.getContext('2d').getImageData(0, 0, 28, 28).data;
            latentVector = new Float32Array(LATENT_DIM);
            for (let i = 0; i < LATENT_DIM; i++) {
                let sum = 0;
                for (let j = i*8; j < Math.min((i+1)*8, pixels.length); j += 4) sum += pixels[j];
                latentVector[i] = (sum / (8*255) - 0.5) * 2 + (Math.random()-0.5)*0.3;
            }
            // Animate vector bars
            drawLatentBars('gp-latent-vec', latentVector, 300, 60);
            
            // Show detected/target digit label
            const latentLabel = document.getElementById('gp-panel-2');
            if (latentLabel) {
                const hint = latentLabel.querySelector('.gp-panel__hint');
                if (hint) hint.innerHTML = `Your drawing was encoded into 100 values.<br><strong style="color:#fdcb6e;">Class Label passed to Generator: ${detectedDigit}</strong>`;
            }

            setTimeout(runGenerate, 3000);
        }

        function drawLatentBars(canvasId, vec, w, h) {
            const c = document.getElementById(canvasId); if (!c) return;
            const ctx = c.getContext('2d'); ctx.clearRect(0, 0, w, h);
            const barW = w / vec.length;
            for (let i = 0; i < vec.length; i++) {
                const v = vec[i], mid = h/2;
                const barH = Math.abs(v) * mid * 0.9;
                const hue = v > 0 ? 180 : 330;
                ctx.fillStyle = `hsla(${hue},70%,55%,0.7)`;
                if (v > 0) ctx.fillRect(i*barW, mid-barH, barW-0.5, barH);
                else ctx.fillRect(i*barW, mid, barW-0.5, barH);
            }
            // Center line
            ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
        }

        // --- Generator (Step 4) ---
        async function runGenerate() {
            setStep(3);
            drawLatentBars('gp-gen-input', latentVector, 200, 40);
            // Draw layer heatmap
            const layerCanvas = document.getElementById('gp-layer-vis');
            if (layerCanvas) {
                const lc = layerCanvas.getContext('2d'); lc.clearRect(0, 0, 200, 100);
                const layers = [
                    {name:'Dense', w:8, h:8}, {name:'Reshape', w:7, h:7},
                    {name:'Conv1', w:14, h:14}, {name:'Conv2', w:28, h:28}
                ];
                let xOff = 5;
                layers.forEach((l, li) => {
                    const cellW = 44, cellH = 90 / l.h;
                    const cW = Math.min(cellW/l.w, 5);
                    for (let r = 0; r < Math.min(l.h, 14); r++) for (let c = 0; c < Math.min(l.w, 8); c++) {
                        const v = Math.abs(Math.sin(r*3+c*7+li*11+latentVector[li*10%100]*5));
                        const hue = 320 - li*30;
                        lc.fillStyle = `hsla(${hue},60%,${20+v*50}%,0.8)`;
                        lc.fillRect(xOff + c*cW, 5 + r*(90/Math.min(l.h,14)), cW-0.5, 90/Math.min(l.h,14)-0.5);
                    }
                    xOff += 50;
                });
            }

            // Generate output using ACGAN (two inputs: latent z + class label)
            let outputPixels;
            if (genModel) {
                try {
                    // Detect which digit was drawn via simple pixel analysis
                    const classLabel = tf.tensor2d([[detectedDigit]]);
                    const zInput = tf.tensor2d([Array.from(latentVector)]);
                    const output = genModel.predict([zInput, classLabel]);
                    const squeezed = output.squeeze();
                    const data = await squeezed.data();
                    outputPixels = data;
                    tf.dispose([zInput, classLabel, output, squeezed]);
                } catch(e) { console.warn('Inference error:', e); outputPixels = null; }
            }

            lastOutputPixels = outputPixels;
            // Draw raw output
            const genOut = document.getElementById('gp-gen-out');
            if (genOut) drawOutput(genOut, outputPixels, 80, 1.0);

            setTimeout(() => runOutput(outputPixels), 3000);
        }

        // --- Output (Step 5) ---
        function runOutput(outputPixels) {
            setStep(4);
            const inputC = document.getElementById('gp-final-input');
            if (inputC) { inputC.getContext('2d').drawImage(drawCanvas, 0, 0, 120, 120); }
            const outC = document.getElementById('gp-final-output');
            if (outC) drawOutput(outC, outputPixels, 120, 1.0);
        }

        function drawOutput(canvas, pixels, size, epochFactor) {
            const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, size, size);
            const cell = size / 28;
            if (pixels && pixels.length >= 784) {
                for (let i = 0; i < 784; i++) {
                    const r = Math.floor(i/28), c = i%28;
                    let v = (pixels[i]+1)/2 * 255 * epochFactor;
                    const noise = (1-epochFactor) * Math.random() * 200;
                    v = Math.max(0, Math.min(255, v + noise));
                    ctx.fillStyle = `rgb(${v},${v},${v})`;
                    ctx.fillRect(c*cell, r*cell, cell+0.5, cell+0.5);
                }
            } else {
                // Simulated output from latent vector
                const DIGIT = [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0];
                const bigCell = size/8;
                for (let i = 0; i < 64; i++) {
                    const row = Math.floor(i/8), col = i%8, on = DIGIT[i]===1;
                    const clarity = epochFactor * epochFactor;
                    const n = (1-clarity) * Math.random();
                    let v = on ? 40+clarity*215+n*40 : n*(120-clarity*100);
                    ctx.fillStyle = `rgb(${Math.min(255,v)},${Math.min(255,v)},${Math.min(255,v)})`;
                    ctx.fillRect(col*bigCell, row*bigCell, bigCell, bigCell);
                }
            }
        }

        // --- Process button ---
        processBtn.addEventListener('click', async () => {
            await loadModel();
            runPreprocess();
        });

        // --- Epoch slider ---
        let lastOutputPixels = null;
        if (epochSlider) epochSlider.addEventListener('input', () => {
            const ep = parseInt(epochSlider.value);
            if (epochVal) epochVal.textContent = ep;
            const outC = document.getElementById('gp-final-output');
            if (outC) drawOutput(outC, lastOutputPixels, 120, ep/100);
        });

        // --- Random noise output ---
        if (randomBtn) randomBtn.addEventListener('click', async () => {
            await loadModel();
            latentVector = new Float32Array(LATENT_DIM);
            for (let i = 0; i < LATENT_DIM; i++) latentVector[i] = (Math.random()-0.5)*2;
            let pixels = null;
            if (genModel) {
                try {
                    const digit = Math.floor(Math.random()*10);
                    const zInput = tf.tensor2d([Array.from(latentVector)]);
                    const classLabel = tf.tensor2d([[digit]]);
                    const output = genModel.predict([zInput, classLabel]);
                    const squeezed = output.squeeze();
                    pixels = await squeezed.data();
                    tf.dispose([zInput, classLabel, output, squeezed]);
                } catch(e) { console.warn('Random gen error:', e); }
            }
            lastOutputPixels = pixels;
            const outC = document.getElementById('gp-final-output');
            if (outC) drawOutput(outC, pixels, 120, (epochSlider ? parseInt(epochSlider.value) : 100)/100);
        });

        // --- Restart ---
        if (restartBtn) restartBtn.addEventListener('click', () => {
            setStep(0);
            drawCtx.fillStyle = '#0a0a0a';
            drawCtx.fillRect(0, 0, 280, 280);
            // Reset upload zone
            if (uploadZone) {
                uploadZone.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" stroke-width="2" style="margin-bottom:16px;">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <span style="color:rgba(255,255,255,0.7); font-size:14px;">Drag & Drop an MNIST image here</span>
                    <span style="color:rgba(255,255,255,0.4); font-size:12px; margin-top:8px;">or click to browse</span>
                `;
            }
        });
    }

    /* ============================================================
       14. SCROLL REVEAL
       ============================================================ */
    function initReveal() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(e => {
                if (e.isIntersecting) {
                    e.target.classList.add('visible');
                }
            });
        }, { threshold: 0.12 });

        $$('.reveal').forEach(el => observer.observe(el));
    }

    /* ============================================================
       BOOT
       ============================================================ */
    document.addEventListener('DOMContentLoaded', () => {
        initHeroParticles();
        initPipelineAnimation();
        initTrainingLoop();
        initGeneratorVis();
        initDiscriminatorVis();
        initDiscInternals();
        initLossGame();
        initTrainCycle();
        initEpochProg();
        initLatentSpace();
        initBattleMode();
        initFinalOutput();
        initGanPipeline();
        initReveal();
    });
})();
