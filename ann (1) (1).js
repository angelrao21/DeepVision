/**
 * ANN Explainer Wizard Module
 */

const MNIST_IMAGES_SPRITE_PATH = 'models/mnist_images.png';
const MNIST_LABELS_PATH = 'models/mnist_labels_uint8';
const IMAGE_SIZE = 784;

const ARCH = {
    l0: { total: 784, visual: 16, label: 'Input' },
    l1: { total: 128, visual: 12, label: 'Hidden 1' },
    l2: { total: 64,  visual: 8,  label: 'Hidden 2' },
    l3: { total: 10,  visual: 10, label: 'Output' }
};

let datasetImages = null;
let datasetLabels = null; // Uint8Array of one-hot encoded labels
let sampleElements = [];
let selectedSampleIndex = -1;
let currentTrueLabel = -1;
let currentPredOutput = null;

let annModel = null;
let annMultiOutputModel = null;
let svgNodes = [[], [], [], []];
let svgEdges = [[], [], []];

let currentStep = -1; // -1=Intro, 0=Input, 1=Forward, 2=Loss, 3=Backprop, 4=AdamW
let isAnimating = false;

document.addEventListener('DOMContentLoaded', async () => {
    await loadMnistDataset();
    populateGrid();
    setupEventListeners();
    await buildModel();
    renderSVGArchitecture();
    initPredictionBars();
    // Enable button for intro step immediately
    document.getElementById('btn-next-step').disabled = false;
});

// ==========================================
// DATA LOADING
// ==========================================
async function loadMnistDataset() {
    const statusEl = document.getElementById('loading-status');
    try {
        const img = new Image(); img.crossOrigin = '';
        const imgReq = new Promise(r => { img.onload = () => r(img); img.src = MNIST_IMAGES_SPRITE_PATH; });
        
        const labelsRes = await fetch(MNIST_LABELS_PATH);
        const labelsBuffer = await labelsRes.arrayBuffer();
        datasetLabels = new Uint8Array(labelsBuffer); // 65000 * 10 one-hot, no header 
        
        const loadedImg = await imgReq;
        const canvas = document.createElement('canvas');
        canvas.width = loadedImg.width; canvas.height = loadedImg.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(loadedImg, 0, 0);
        
        datasetImages = new Float32Array(65000 * IMAGE_SIZE);
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < 65000 * IMAGE_SIZE; i++) {
            datasetImages[i] = imgData.data[i * 4] / 255.0;
        }
        statusEl.textContent = 'Loaded ✓';
    } catch (e) { console.error(e); statusEl.textContent = 'Failed'; }
}

function getLabelFromOneHot(idx) {
    const offset = idx * 10;
    for (let i = 0; i < 10; i++) {
        if (datasetLabels[offset + i] === 1) return i;
    }
    return 0;
}

function populateGrid() {
    if (!datasetImages) return;
    const grid = document.getElementById('mnist-grid');
    grid.innerHTML = '';
    sampleElements = [];
    for (let i = 0; i < 40; i++) {
        const randIdx = Math.floor(Math.random() * 10000);
        const canvas = document.createElement('canvas');
        canvas.width = 28; canvas.height = 28;
        drawDigit(canvas, randIdx);
        canvas.addEventListener('click', () => selectDigit(canvas, randIdx));
        grid.appendChild(canvas);
        sampleElements.push({ canvas, idx: randIdx });
    }
}

function drawDigit(canvas, idx) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(28, 28);
    const offset = idx * IMAGE_SIZE;
    for (let i = 0; i < IMAGE_SIZE; i++) {
        const val = Math.round(datasetImages[offset + i] * 255);
        imgData.data[i*4] = val; imgData.data[i*4+1] = val; imgData.data[i*4+2] = val; imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
}

function selectDigit(selectedCanvas, idx) {
    sampleElements.forEach(item => item.canvas.classList.remove('selected'));
    if (selectedCanvas) selectedCanvas.classList.add('selected');
    selectedSampleIndex = idx;
    currentTrueLabel = getLabelFromOneHot(idx);
    
    drawDigit(document.getElementById('selected-input-canvas'), idx);
    document.getElementById('selected-label-value').textContent = currentTrueLabel;
    
    // Light up SVG input layer based on image
    const offset = idx * IMAGE_SIZE;
    const inputArr = new Float32Array(IMAGE_SIZE);
    for(let i=0; i<IMAGE_SIZE; i++) inputArr[i] = datasetImages[offset + i];
    applyGlowToNodes(0, inputArr, ARCH.l0.visual);

    document.getElementById('btn-next-step').disabled = false;
    document.getElementById('btn-next-text').textContent = 'Step 2: Forward Propagation';
}

function setupEventListeners() {
    document.getElementById('btn-random-sample').addEventListener('click', () => {
        if (sampleElements.length === 0) return;
        const randomItem = sampleElements[Math.floor(Math.random() * sampleElements.length)];
        selectDigit(randomItem.canvas, randomItem.idx);
    });

    document.getElementById('btn-next-step').addEventListener('click', handleNextStep);
}

// ==========================================
// TF.JS MODEL & SVG ARCHITECTURE
// ==========================================
async function buildModel() {
    annModel = tf.sequential();
    annModel.add(tf.layers.dense({ units: ARCH.l1.total, activation: 'relu', inputShape: [ARCH.l0.total], kernelInitializer: 'glorotNormal' }));
    annModel.add(tf.layers.dense({ units: ARCH.l2.total, activation: 'relu', kernelInitializer: 'glorotNormal' }));
    annModel.add(tf.layers.dense({ units: ARCH.l3.total, activation: 'softmax', kernelInitializer: 'glorotNormal' }));
    annModel.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const layerOutputs = annModel.layers.map(l => l.output);
    annMultiOutputModel = tf.model({ inputs: annModel.inputs, outputs: layerOutputs });
}

function renderSVGArchitecture() {
    const container = document.getElementById('ann-svg-container');
    const w = container.clientWidth || 800; const h = container.clientHeight || 400;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%");
    
    const columns = [
        { nodes: ARCH.l0.visual, x: w * 0.1, label: 'Input' },
        { nodes: ARCH.l1.visual, x: w * 0.35, label: 'Hidden 1' },
        { nodes: ARCH.l2.visual, x: w * 0.6, label: 'Hidden 2' },
        { nodes: ARCH.l3.visual, x: w * 0.85, label: 'Output' }
    ];

    const layerCoords = [];
    columns.forEach(col => {
        const coords = [];
        const spacing = (h - 80) / Math.max(1, col.nodes - 1);
        const startY = 40 + (h - 80 - (spacing * (col.nodes - 1))) / 2;
        for (let i = 0; i < col.nodes; i++) coords.push({ x: col.x, y: startY + i * spacing });
        layerCoords.push(coords);
    });

    svgEdges = [[], [], []];
    for (let l = 0; l < layerCoords.length - 1; l++) {
        for (let n1 = 0; n1 < layerCoords[l].length; n1++) {
            for (let n2 = 0; n2 < layerCoords[l+1].length; n2++) {
                const path = document.createElementNS(ns, "line");
                path.setAttribute("x1", layerCoords[l][n1].x); path.setAttribute("y1", layerCoords[l][n1].y);
                path.setAttribute("x2", layerCoords[l+1][n2].x); path.setAttribute("y2", layerCoords[l+1][n2].y);
                path.setAttribute("class", "ann-edge");
                svg.appendChild(path); svgEdges[l].push(path);
            }
        }
    }

    svgNodes = [[], [], [], []];
    layerCoords.forEach((layer, lIdx) => {
        layer.forEach((coord, nIdx) => {
            const circle = document.createElementNS(ns, "circle");
            circle.setAttribute("cx", coord.x); circle.setAttribute("cy", coord.y);
            circle.setAttribute("r", lIdx === 0 || lIdx === 3 ? 6 : 8);
            circle.setAttribute("class", "ann-node");
            svg.appendChild(circle); svgNodes[lIdx].push(circle);

            if (lIdx === 3) {
                const lText = document.createElementNS(ns, "text");
                lText.setAttribute("x", coord.x + 15); lText.setAttribute("y", coord.y + 4);
                lText.setAttribute("class", "ann-label"); lText.textContent = nIdx;
                svg.appendChild(lText);
            }
        });
    });
    container.appendChild(svg);
}

function initPredictionBars() {
    const container = document.getElementById('prob-bars');
    for (let i = 0; i < 10; i++) {
        container.innerHTML += `<div class="prob-col" id="prob-col-${i}"><div class="prob-val" id="prob-val-${i}">0%</div><div class="prob-bar-fill" id="prob-bar-${i}"></div><div class="prob-label">${i}</div></div>`;
    }
}

// ==========================================
// STATE MACHINE WIZARD
// ==========================================
async function handleNextStep() {
    if (isAnimating) return;
    isAnimating = true;
    const btn = document.getElementById('btn-next-step');
    btn.disabled = true;

    // Hide all step content
    document.querySelectorAll('.step-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.progress-dot').forEach(el => el.classList.remove('active'));
    
    currentStep++;
    if (currentStep > 4) currentStep = 1; // Loop back to forward prop
    
    // Map step to content id
    const stepIds = ['step-intro-content', 'step-0-content', 'step-1-content', 'step-2-content', 'step-3-content', 'step-4-content'];
    const dotIds = ['dot-intro', 'dot-0', 'dot-1', 'dot-2', 'dot-3', 'dot-4'];
    const stepIdx = currentStep + 1; // -1 -> 0, 0 -> 1, etc.
    
    document.getElementById(stepIds[stepIdx]).classList.add('active');
    document.getElementById(dotIds[stepIdx]).classList.add('active');
    for(let i = 0; i < stepIdx; i++) {
        document.getElementById(dotIds[i]).classList.add('completed');
    }
    // Mark progress lines
    const lines = document.getElementsByClassName('progress-line');
    for(let i = 0; i < Math.min(stepIdx, lines.length); i++) {
        lines[i].classList.add('completed');
    }

    // Execute Logic
    if (currentStep === 0) {
        // Transitioning from intro to data selection - no async work needed
    } else if (currentStep === 1) await runForwardPropagation();
    else if (currentStep === 2) await runLossCalculation();
    else if (currentStep === 3) await runBackpropagation();
    else if (currentStep === 4) await runWeightUpdate();

    // Prepare for next
    isAnimating = false;
    
    if (currentStep === 0) {
        // Need to select an image before proceeding
        btn.disabled = true;
        document.getElementById('btn-next-text').textContent = 'Select an Image to Continue';
    } else {
        btn.disabled = false;
        const nextLabels = {
            1: 'Step 3: Calculate Loss',
            2: 'Step 4: Backpropagate Error',
            3: 'Step 5: Update Weights (AdamW)',
            4: 'Next Epoch: Forward Prop'
        };
        document.getElementById('btn-next-text').textContent = nextLabels[currentStep];
    }
}

// ==========================================
// WIZARD STEPS ANIMATION LOGIC
// ==========================================

async function runForwardPropagation() {
    log('Flatted input applied to 784 input nodes.');
    const offset = selectedSampleIndex * IMAGE_SIZE;
    const inputArr = new Float32Array(IMAGE_SIZE);
    for(let i=0; i<IMAGE_SIZE; i++) inputArr[i] = datasetImages[offset + i];

    const inputTensor = tf.tensor2d([inputArr], [1, 784]);
    const [act1, act2, act3] = annMultiOutputModel.predict(inputTensor);
    const h1Arr = await act1.data(), h2Arr = await act2.data();
    currentPredOutput = await act3.data();
    inputTensor.dispose(); act1.dispose(); act2.dispose(); act3.dispose();

    // Reset styles
    applyGlowToNodes(1, new Float32Array(ARCH.l1.visual), ARCH.l1.visual);
    applyGlowToNodes(2, new Float32Array(ARCH.l2.visual), ARCH.l2.visual);
    applyGlowToNodes(3, new Float32Array(ARCH.l3.visual), ARCH.l3.visual);

    activateEdges(0); await sleep(400); applyGlowToNodes(1, h1Arr, ARCH.l1.visual, true);
    deactivateEdges(0); activateEdges(1); await sleep(400); applyGlowToNodes(2, h2Arr, ARCH.l2.visual, true);
    deactivateEdges(1); activateEdges(2); await sleep(400); applyGlowToNodes(3, currentPredOutput, ARCH.l3.visual, false);
    deactivateEdges(2);
    
    updatePredictionBars(currentPredOutput);
    log('Forward propagation complete.');
}

async function runLossCalculation() {
    // Categorical Crossentropy: -sum(true * log(pred))
    // Since true is one-hot, it's just -log(pred[true_label])
    let loss = -Math.log(currentPredOutput[currentTrueLabel] + 1e-7);
    
    document.getElementById('loss-value-display').textContent = loss.toFixed(4);
    const fillPct = Math.min(100, (loss / 5) * 100);
    document.getElementById('loss-bar-fill').style.width = `${fillPct}%`;
}

async function runBackpropagation() {
    // Flash red/blue backwards
    svgEdges[2].forEach(e => e.classList.add(Math.random()>0.5?'backprop-pos':'backprop-neg'));
    await sleep(400);
    svgEdges[2].forEach(e => e.className.baseVal = 'ann-edge');
    svgEdges[1].forEach(e => e.classList.add(Math.random()>0.5?'backprop-pos':'backprop-neg'));
    await sleep(400);
    svgEdges[1].forEach(e => e.className.baseVal = 'ann-edge');
    svgEdges[0].forEach(e => e.classList.add(Math.random()>0.5?'backprop-pos':'backprop-neg'));
    await sleep(400);
    svgEdges[0].forEach(e => e.className.baseVal = 'ann-edge');
}

let simM = 0, simV = 0;
async function runWeightUpdate() {
    // 1. Actually train the model for 1 batch to update weights
    const offset = selectedSampleIndex * IMAGE_SIZE;
    const inputArr = new Float32Array(IMAGE_SIZE);
    for(let i=0; i<IMAGE_SIZE; i++) inputArr[i] = datasetImages[offset + i];
    const xTensor = tf.tensor2d([inputArr], [1, 784]);
    const yArr = new Array(10).fill(0); yArr[currentTrueLabel] = 1;
    const yTensor = tf.tensor2d([yArr], [1, 10]);
    
    await annModel.fit(xTensor, yTensor, { batchSize: 1, epochs: 1 });
    xTensor.dispose(); yTensor.dispose();

    // 2. Simulate AdamW Maths visually
    let simG = (Math.random() - 0.5) * 0.8; 
    simM = 0.9 * simM + 0.1 * simG;
    simV = 0.999 * simV + 0.001 * (simG * simG);
    document.getElementById('a-bar-g').style.width = `${Math.abs(simG)*100}%`;
    document.getElementById('a-bar-m').style.width = `${Math.abs(simM)*200}%`;
    document.getElementById('a-bar-v').style.width = `${simV*5000}%`;

    // 3. Render real heatmap
    const weightsTensor = annModel.layers[0].getWeights()[0];
    const wData = await weightsTensor.data();
    const canvas = document.getElementById('weight-heatmap');
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
        const val = wData[i * 128]; // Sample first hidden neuron connections
        const color = val > 0 ? [59, 130, 246] : [239, 68, 68];
        const intensity = Math.min(255, Math.abs(val) * 1000);
        imgData.data[i*4] = color[0]; imgData.data[i*4+1] = color[1]; imgData.data[i*4+2] = color[2]; imgData.data[i*4+3] = intensity;
    }
    ctx.putImageData(imgData, 0, 0);
}

// --- Utils ---
function applyGlowToNodes(layerIdx, dataArr, visualCount, norm=false) {
    const stride = Math.floor(dataArr.length / visualCount);
    let max = 1; if(norm) { for(let i=0;i<dataArr.length;i++) if(dataArr[i]>max) max=dataArr[i]; }
    for (let i = 0; i < visualCount; i++) {
        const val = dataArr[i * stride] / max;
        const node = svgNodes[layerIdx][i];
        if (val > 0.01) {
            const v = Math.min(1, Math.max(0.1, val));
            node.style.fill = `rgba(255,255,255,${v})`; node.style.stroke = `rgba(16,185,129,${v})`;
            node.style.filter = `drop-shadow(0 0 ${v*10}px rgba(16,185,129,${v}))`;
        } else {
            node.style.fill = '#111'; node.style.stroke = 'rgba(255,255,255,0.2)'; node.style.filter = 'none';
        }
    }
}
function activateEdges(l) { svgEdges[l].forEach(e => e.classList.add('active')); }
function deactivateEdges(l) { svgEdges[l].forEach(e => e.classList.remove('active')); }
function updatePredictionBars(probArray) {
    let maxIdx = 0, maxProb = 0;
    for(let i=0;i<10;i++) { if(probArray[i]>maxProb) { maxProb=probArray[i]; maxIdx=i; } }
    for(let i=0;i<10;i++) {
        const p = probArray[i] * 100;
        document.getElementById(`prob-bar-${i}`).style.height = `${p}%`;
        document.getElementById(`prob-val-${i}`).textContent = `${p.toFixed(1)}%`;
        const col = document.getElementById(`prob-col-${i}`);
        i === maxIdx ? col.classList.add('predicted') : col.classList.remove('predicted');
    }
}
function log(msg) { const l=document.getElementById('log-forward'); if(l) l.innerHTML = `> ${msg}`; }
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
