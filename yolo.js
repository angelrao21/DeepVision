/**
 * YOLO Interactive Explainer Logic
 * Handles image upload, canvas drawing, and step-by-step simulations.
 */

document.addEventListener('DOMContentLoaded', () => {
    
    // --- State ---
    const state = {
        image: null,
        imgWidth: 400,
        imgHeight: 400,
        gridSize: 7, // Default S = 7
        boxes: [], // Raw predictions
        nmsBoxes: [], // Filtered predictions
        classes: [], // Populated dynamically
        colors: ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'],
        realObjects: [], // Populated by TFJS
        confThreshold: 0.25,
        iouThreshold: 0.45,
        modelLoaded: false,
        model: null
    };

    // Load COCO-SSD Model
    const loadModel = async () => {
        try {
            state.model = await cocoSsd.load();
            state.modelLoaded = true;
            console.log('Model loaded');
        } catch (e) {
            console.error('Failed to load model', e);
        }
    };
    loadModel();

    // --- Elements ---
    const uploadInput = document.getElementById('image-upload-input');
    const uploadZone = document.getElementById('upload-zone');
    const previewZone = document.getElementById('preview-zone');
    const btnSendModel = document.getElementById('btn-send-model');
    const previewCanvas = document.getElementById('input-preview-canvas');
    const previewCtx = previewCanvas.getContext('2d');

    // --- Canvases ---
    const canvases = {
        hero: document.getElementById('hero-yolo-canvas'),
        grid: document.getElementById('grid-canvas'),
        bbox: document.getElementById('bbox-canvas'),
        classProb: document.getElementById('class-canvas'),
        nms: document.getElementById('nms-canvas'),
        final: document.getElementById('final-canvas')
    };

    // Initialize all canvas contexts
    const ctxs = {};
    for (let key in canvases) {
        if (canvases[key]) {
            ctxs[key] = canvases[key].getContext('2d');
        }
    }

    // --- Helper Functions ---
    
    // Draw image onto a canvas covering it
    function drawImageCover(ctx, img) {
        ctx.clearRect(0, 0, 400, 400);
        if (!img) return;
        const scale = Math.max(400 / img.width, 400 / img.height);
        const x = (400 / 2) - (img.width / 2) * scale;
        const y = (400 / 2) - (img.height / 2) * scale;
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
    }

    function generateMockPredictions() {
        state.boxes = [];
        const S = state.gridSize;
        const cellSize = 400 / S;
        
        // Use real objects detected by the model
        const objects = state.realObjects;

        for (let row = 0; row < S; row++) {
            for (let col = 0; col < S; col++) {
                const cellX = col * cellSize;
                const cellY = row * cellSize;
                
                // Each cell predicts B=2 boxes
                for (let b = 0; b < 2; b++) {
                    // Base random box
                    let box = {
                        cx: cellX + Math.random() * cellSize,
                        cy: cellY + Math.random() * cellSize,
                        w: cellSize + Math.random() * 100,
                        h: cellSize + Math.random() * 100,
                        conf: Math.random() * 0.3, // Mostly low confidence
                        classIdx: Math.floor(Math.random() * state.classes.length)
                    };

                    // Check if cell is near a "real" object
                    objects.forEach(obj => {
                        const dist = Math.hypot(box.cx - obj.cx, box.cy - obj.cy);
                        if (dist < cellSize * 1.5) {
                            box.cx = obj.cx + (Math.random() - 0.5) * 30;
                            box.cy = obj.cy + (Math.random() - 0.5) * 30;
                            box.w = obj.w + (Math.random() - 0.5) * 40;
                            box.h = obj.h + (Math.random() - 0.5) * 40;
                            box.conf = 0.5 + Math.random() * 0.45; // High confidence
                            box.classIdx = obj.classIdx;
                        }
                    });

                    state.boxes.push(box);
                }
            }
        }
    }

    function calculateIOU(b1, b2) {
        const x1 = Math.max(b1.cx - b1.w/2, b2.cx - b2.w/2);
        const y1 = Math.max(b1.cy - b1.h/2, b2.cy - b2.h/2);
        const x2 = Math.min(b1.cx + b1.w/2, b2.cx + b2.w/2);
        const y2 = Math.min(b1.cy + b1.h/2, b2.cy + b2.h/2);
        
        if (x2 < x1 || y2 < y1) return 0.0;
        
        const intersection = (x2 - x1) * (y2 - y1);
        const area1 = b1.w * b1.h;
        const area2 = b2.w * b2.h;
        return intersection / (area1 + area2 - intersection);
    }

    function runNMS() {
        // Filter by confidence
        let candidates = state.boxes.filter(b => b.conf >= state.confThreshold);
        // Sort by confidence descending
        candidates.sort((a, b) => b.conf - a.conf);
        
        let finalBoxes = [];
        while (candidates.length > 0) {
            const best = candidates.shift();
            finalBoxes.push(best);
            // Remove boxes with high IOU with the best box
            candidates = candidates.filter(b => calculateIOU(best, b) < state.iouThreshold);
        }
        state.nmsBoxes = finalBoxes;
    }

    // --- Hero Animation ---
    function initHeroAnimation() {
        const ctx = ctxs.hero;
        if (!ctx) return;
        let t = 0;
        function drawHero() {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, 400, 400);
            
            // Draw fake grid
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 1;
            for(let i=0; i<=400; i+=40) {
                ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 400); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(400, i); ctx.stroke();
            }

            // Pulsing scanner line
            const scanY = (t * 2) % 400;
            ctx.fillStyle = 'rgba(34, 197, 94, 0.2)';
            ctx.fillRect(0, scanY, 400, 40);
            ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
            ctx.fillRect(0, scanY + 40, 400, 2);

            // Draw a few fake boxes
            const pulse = Math.sin(t * 0.05) * 0.5 + 0.5;
            ctx.strokeStyle = `rgba(59, 130, 246, ${0.5 + pulse*0.5})`;
            ctx.lineWidth = 2;
            ctx.strokeRect(100, 100, 150, 200);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = "14px Inter";
            ctx.fillText("Object 85%", 100, 95);

            t++;
            requestAnimationFrame(drawHero);
        }
        drawHero();
    }
    initHeroAnimation();

    // --- Step 1: Upload ---
    uploadZone.addEventListener('click', () => uploadInput.click());
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.style.borderColor = '#3b82f6'; });
    uploadZone.addEventListener('dragleave', e => { e.preventDefault(); uploadZone.style.borderColor = 'rgba(255,255,255,0.1)'; });
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(255,255,255,0.1)';
        if (e.dataTransfer.files.length) handleImage(e.dataTransfer.files[0]);
    });
    uploadInput.addEventListener('change', e => {
        if (e.target.files.length) handleImage(e.target.files[0]);
    });

    async function runInference(img) {
        if (!state.modelLoaded) await loadModel();
        
        const wrap = document.querySelector('.yolo-canvas-wrapper');
        const scan = document.createElement('div');
        scan.className = 'scan-overlay';
        if (wrap) wrap.appendChild(scan);

        const predictions = await state.model.detect(img);
        
        if (wrap && scan.parentNode === wrap) wrap.removeChild(scan);
        
        state.classes = [];
        state.realObjects = [];
        
        predictions.forEach(p => {
            let classIdx = state.classes.indexOf(p.class);
            if (classIdx === -1) {
                state.classes.push(p.class);
                classIdx = state.classes.length - 1;
            }
            
            const scaleX = 400 / img.width;
            const scaleY = 400 / img.height;
            
            state.realObjects.push({
                cx: (p.bbox[0] + p.bbox[2]/2) * scaleX,
                cy: (p.bbox[1] + p.bbox[3]/2) * scaleY,
                w: p.bbox[2] * scaleX,
                h: p.bbox[3] * scaleY,
                classIdx: classIdx,
                conf: p.score,
                real: true
            });
        });

        if (state.realObjects.length === 0) {
            // Fallback to CADI-AI dataset defaults for the sample image (1.txt)
            state.classes = ['Abiotic', 'Insect', 'Disease'];
            state.realObjects.push({ cx: 82, cy: 284, w: 28, h: 38, classIdx: 0, conf: 0.88, real: true }); // Abiotic
        }

        generateMockPredictions();
    }

    function handleImage(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = async () => {
                state.image = img;
                uploadZone.style.display = 'none';
                previewZone.style.display = 'block';
                drawImageCover(previewCtx, img);
                
                btnSendModel.classList.add('loading');
                btnSendModel.textContent = '';
                btnSendModel.disabled = true;
                
                await runInference(img);
                
                btnSendModel.classList.remove('loading');
                btnSendModel.textContent = 'Process Image 👉';
                btnSendModel.disabled = false;
                btnSendModel.classList.add('animate-pulse');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    const useSample = document.getElementById('use-sample');
    if (useSample) {
        useSample.addEventListener('click', (e) => {
            e.preventDefault();
            const img = new Image();
            img.onload = async () => {
                state.image = img;
                uploadZone.style.display = 'none';
                previewZone.style.display = 'block';
                drawImageCover(previewCtx, img);
                
                btnSendModel.classList.add('loading');
                btnSendModel.textContent = '';
                btnSendModel.disabled = true;
                
                await runInference(img);
                
                btnSendModel.classList.remove('loading');
                btnSendModel.textContent = 'Process Image 👉';
                btnSendModel.disabled = false;
                btnSendModel.classList.add('animate-pulse');
            };
            img.src = '../Data/train/train/images/1.jpg';
        });
    }

    // Next Step Handlers
    btnSendModel.addEventListener('click', () => {
        btnSendModel.classList.remove('animate-pulse');
        document.getElementById('step-connector-1-2').style.display = 'flex';
        document.getElementById('step-2').style.display = 'block';
        document.getElementById('step-2').scrollIntoView({ behavior: 'smooth' });
        drawGridStep();
    });

    // --- Step 2: Grid ---
    const gridSlider = document.getElementById('grid-slider');
    const gridVal = document.getElementById('grid-val');
    
    function drawGridStep() {
        const ctx = ctxs.grid;
        drawImageCover(ctx, state.image);
        
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        const S = state.gridSize;
        const step = 400 / S;
        
        for (let i = 1; i < S; i++) {
            ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, 400); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i * step); ctx.lineTo(400, i * step); ctx.stroke();
        }
    }

    gridSlider.addEventListener('input', e => {
        state.gridSize = parseInt(e.target.value);
        gridVal.textContent = `${state.gridSize}×${state.gridSize}`;
        generateMockPredictions(); // regenerate for new grid
        drawGridStep();
    });

    document.getElementById('btn-step-2-next').addEventListener('click', () => {
        document.getElementById('step-connector-2-3').style.display = 'flex';
        document.getElementById('step-3').style.display = 'block';
        document.getElementById('step-3').scrollIntoView({ behavior: 'smooth' });
        drawBBoxStep();
    });

    // --- Step 3: BBox ---
    function drawBBoxStep() {
        const ctx = ctxs.bbox;
        drawImageCover(ctx, state.image);
        
        // Draw all boxes, low opacity
        state.boxes.forEach(b => {
            ctx.strokeStyle = `rgba(236, 72, 153, ${Math.max(0.1, b.conf)})`;
            ctx.lineWidth = 1;
            ctx.strokeRect(b.cx - b.w/2, b.cy - b.h/2, b.w, b.h);
            
            // Draw center point
            ctx.fillStyle = `rgba(236, 72, 153, ${b.conf})`;
            ctx.beginPath();
            ctx.arc(b.cx, b.cy, 2, 0, Math.PI*2);
            ctx.fill();
        });
        
        document.getElementById('bbox-stats').innerHTML = `Generated <strong>${state.boxes.length}</strong> raw bounding boxes across ${state.gridSize*state.gridSize} cells.`;
    }

    document.getElementById('btn-step-3-next').addEventListener('click', () => {
        document.getElementById('step-connector-3-4').style.display = 'flex';
        document.getElementById('step-4').style.display = 'block';
        document.getElementById('step-4').scrollIntoView({ behavior: 'smooth' });
        drawClassStep();
    });

    // --- Step 4: Class Probs ---
    function drawClassStep() {
        const ctx = ctxs.classProb;
        drawImageCover(ctx, state.image);
        
        const S = state.gridSize;
        const step = 400 / S;
        
        // Aggregate dominant class per cell
        for (let row = 0; row < S; row++) {
            for (let col = 0; col < S; col++) {
                // Find highest conf box in this cell
                const cellBoxes = state.boxes.filter(b => 
                    b.cx >= col*step && b.cx < (col+1)*step &&
                    b.cy >= row*step && b.cy < (row+1)*step
                );
                
                if (cellBoxes.length > 0) {
                    const bestBox = cellBoxes.reduce((max, box) => max.conf > box.conf ? max : box);
                    if (bestBox.conf > 0.1) {
                        ctx.fillStyle = state.colors[bestBox.classIdx] + '80'; // 50% opacity
                        ctx.fillRect(col*step, row*step, step, step);
                    }
                }
            }
        }
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        for (let i = 1; i < S; i++) {
            ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, 400); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i * step); ctx.lineTo(400, i * step); ctx.stroke();
        }

        // Setup Legend
        const legend = document.getElementById('class-legend');
        legend.innerHTML = '';
        state.classes.forEach((cls, i) => {
            legend.innerHTML += `
                <div class="yolo-legend-item">
                    <div class="yolo-legend-color" style="background: ${state.colors[i]}"></div>
                    ${cls}
                </div>
            `;
        });
    }

    document.getElementById('btn-step-4-next').addEventListener('click', () => {
        document.getElementById('step-connector-4-5').style.display = 'flex';
        document.getElementById('step-5').style.display = 'block';
        document.getElementById('step-5').scrollIntoView({ behavior: 'smooth' });
        drawNMSStep();
    });

    // --- Step 5: NMS ---
    const confSlider = document.getElementById('conf-slider');
    const iouSlider = document.getElementById('iou-slider');
    const confVal = document.getElementById('conf-val');
    const iouVal = document.getElementById('iou-val');

    function drawNMSStep() {
        runNMS();
        const ctx = ctxs.nms;
        drawImageCover(ctx, state.image);
        
        // Draw remaining boxes
        state.nmsBoxes.forEach(b => {
            const color = state.colors[b.classIdx];
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(b.cx - b.w/2, b.cy - b.h/2, b.w, b.h);
            
            // Label background
            ctx.fillStyle = color;
            ctx.fillRect(b.cx - b.w/2, b.cy - b.h/2 - 20, 40, 20);
            
            // Label text
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.fillText(`${(b.conf*100).toFixed(0)}%`, b.cx - b.w/2 + 5, b.cy - b.h/2 - 5);
        });

        document.getElementById('nms-count').textContent = state.nmsBoxes.length;
    }

    confSlider.addEventListener('input', e => {
        state.confThreshold = e.target.value / 100;
        confVal.textContent = state.confThreshold.toFixed(2);
        drawNMSStep();
    });

    iouSlider.addEventListener('input', e => {
        state.iouThreshold = e.target.value / 100;
        iouVal.textContent = state.iouThreshold.toFixed(2);
        drawNMSStep();
    });

    document.getElementById('btn-step-5-next').addEventListener('click', () => {
        document.getElementById('step-connector-5-6').style.display = 'flex';
        document.getElementById('step-6').style.display = 'block';
        document.getElementById('step-6').scrollIntoView({ behavior: 'smooth' });
        drawFinalStep();
    });

    // --- Step 6: Final Output ---
    function drawFinalStep() {
        const ctx = ctxs.final;
        drawImageCover(ctx, state.image);
        
        const list = document.getElementById('final-list');
        list.innerHTML = '';

        if (state.nmsBoxes.length === 0) {
            list.innerHTML = '<div style="color:#9ca3af; text-align:center;">No objects detected. Try lowering the confidence threshold.</div>';
        }

        state.nmsBoxes.forEach(b => {
            const className = state.classes[b.classIdx];
            const color = state.colors[b.classIdx];
            
            // Draw Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            const x = Math.max(0, b.cx - b.w/2);
            const y = Math.max(0, b.cy - b.h/2);
            ctx.strokeRect(x, y, b.w, b.h);
            
            // Draw Label
            const text = `${className} ${(b.conf*100).toFixed(1)}%`;
            ctx.font = 'bold 14px Inter';
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 24, textWidth + 10, 24);
            ctx.fillStyle = '#fff';
            ctx.fillText(text, x + 5, y - 7);

            // Add to list
            list.innerHTML += `
                <div class="detection-item" style="border-left-color: ${color}">
                    <span style="font-weight: bold; color: #fff;">${className}</span>
                    <span style="color: #9ca3af;">${(b.conf*100).toFixed(1)}% Conf</span>
                </div>
            `;
        });
    }

    // --- Quiz Logic ---
    const quizQuestions = document.querySelectorAll('.quiz-question');
    const quizBtns = document.querySelectorAll('.quiz-btn');
    const quizResult = document.getElementById('quiz-result');

    quizBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const isCorrect = e.target.dataset.correct === 'true';
            const parent = e.target.closest('.quiz-question');
            const btns = parent.querySelectorAll('.quiz-btn');
            const feedback = parent.querySelector('.quiz-feedback');

            // Disable all buttons in this question
            btns.forEach(b => b.disabled = true);

            if (isCorrect) {
                e.target.classList.add('selected-correct');
                feedback.textContent = 'Correct! ✅';
                feedback.style.color = '#22c55e';
            } else {
                e.target.classList.add('selected-wrong');
                feedback.textContent = 'Incorrect ❌';
                feedback.style.color = '#ef4444';
                // Show the correct one
                btns.forEach(b => {
                    if(b.dataset.correct === 'true') {
                        b.style.borderColor = '#22c55e';
                        b.style.color = '#22c55e';
                    }
                });
            }
            feedback.style.display = 'block';

            // Show next question
            const nextId = parseInt(parent.id.split('-')[2]) + 1;
            const nextQ = document.getElementById(`quiz-question-${nextId}`);
            if (nextQ) {
                setTimeout(() => {
                    nextQ.style.display = 'block';
                    nextQ.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 800);
            } else {
                // End of quiz
                setTimeout(() => {
                    quizResult.style.display = 'block';
                    quizResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 800);
            }
        });
    });

});
