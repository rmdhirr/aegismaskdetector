const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');
const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', { willReadFrequently: true });
const statusPanel = document.getElementById('status-panel');

let session;
const TARGET_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.4;

/**
 * Initialize the ONNX Runtime WASM Session
 */
async function loadModel() {
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('./best.onnx', { executionProviders: ['wasm'] });
        statusPanel.innerText = "STANDBY: AWAITING CAMERA INITIALIZATION";
        startCamera();
    } catch (e) {
        console.error("Model Mount Failure:", e);
        statusPanel.innerText = "SYSTEM FAILURE: UNABLE TO MOUNT MODEL";
        statusPanel.style.borderColor = "#ff0000";
        statusPanel.style.color = "#ff0000";
    }
}

/**
 * Bind the hardware stream to the video element (Optimized for both Laptops and Mobile)
 */
async function startCamera() {
    try {
        // Relaxed constraints to ensure compatibility with standard laptop webcams
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 } 
            },
            audio: false
        });
        
        video.srcObject = stream;
        
        // Explicitly command the video to play to circumvent browser autoplay blocking
        await video.play();

        video.onloadedmetadata = () => {
            statusPanel.innerText = "AWAITING SUBJECT...";
            requestAnimationFrame(processFrame);
        };
    } catch (e) {
        console.error("Camera Access Failure:", e);
        statusPanel.innerText = "SYSTEM FAILURE: CAMERA ACCESS DENIED";
        statusPanel.style.borderColor = "#ff0000";
        statusPanel.style.color = "#ff0000";
    }
}

/**
 * Intersection over Union (IoU) Calculation
 */
function calculateIoU(box1, box2) {
    const xA = Math.max(box1.x, box2.x);
    const yA = Math.max(box1.y, box2.y);
    const xB = Math.min(box1.x + box1.w, box2.x + box2.w);
    const yB = Math.min(box1.y + box1.h, box2.y + box2.h);

    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = box1.w * box1.h;
    const box2Area = box2.w * box2.h;

    return intersectionArea / (box1Area + box2Area - intersectionArea);
}

/**
 * Non-Maximum Suppression (NMS)
 */
function nonMaxSuppression(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const current = boxes.shift();
        result.push(current);
        boxes = boxes.filter(box => calculateIoU(current, box) < iouThreshold);
    }
    return result;
}

/**
 * Main Inference and Rendering Loop
 */
async function processFrame() {
    if (!session) return;

    // 1. Frame Extraction & Scaling
    ctxProcessor.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
    const imageData = ctxProcessor.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;

    // 2. Tensor Memory Allocation [1, 3, 640, 640] Float32
    const float32Data = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        float32Data[i]                                   = imageData[i * 4] / 255.0;     // R
        float32Data[i + TARGET_SIZE * TARGET_SIZE]       = imageData[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE]   = imageData[i * 4 + 2] / 255.0; // B
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_SIZE, TARGET_SIZE]);
    
    // 3. Execution
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = results[session.outputNames[0]].data; 
    
    let rawBoxes = [];
    const elements = 8400;

    // 4. Output Parsing (1, 6, 8400 shape)
    for (let i = 0; i < elements; i++) {
        const x = output[i];
        const y = output[i + elements];
        const w = output[i + 2 * elements];
        const h = output[i + 3 * elements];
        
        const scoreMask = output[i + 4 * elements]; 
        const scoreNoMask = output[i + 5 * elements];

        const maxScore = Math.max(scoreMask, scoreNoMask);

        if (maxScore > CONFIDENCE_THRESHOLD) {
            const classId = scoreNoMask > scoreMask ? 1 : 0;
            rawBoxes.push({
                x: x - w / 2,
                y: y - h / 2,
                w: w,
                h: h,
                score: maxScore,
                classId: classId
            });
        }
    }

    // 5. Post-Processing
    const finalBoxes = nonMaxSuppression(rawBoxes, IOU_THRESHOLD);

    // 6. UI Rendering Phase
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    let isViolating = false;

    if (finalBoxes.length > 0) {
        finalBoxes.forEach(box => {
            // Un-squish coordinates from 640x640 model input back to 640x480 UI rendering resolution
            const scaleX = overlay.width / TARGET_SIZE;
            const scaleY = overlay.height / TARGET_SIZE;
            
            const scaledX = box.x * scaleX;
            const scaledY = box.y * scaleY;
            const scaledW = box.w * scaleX;
            const scaledH = box.h * scaleY;

            // Define UI Parameters
            const isNoMask = box.classId === 1;
            if (isNoMask) isViolating = true;
            
            const color = isNoMask ? '#FF3B30' : '#34C759'; 
            const labelText = isNoMask ? `NO MASK ${(box.score * 100).toFixed(1)}%` : `MASK ${(box.score * 100).toFixed(1)}%`;

            // Draw Checkpoint Bounding Box
            ctxOverlay.strokeStyle = color;
            ctxOverlay.lineWidth = 4;
            ctxOverlay.strokeRect(scaledX, scaledY, scaledW, scaledH);
            
            // Render Dynamic Background Block for Text Readability
            ctxOverlay.font = 'bold 18px monospace';
            const textWidth = ctxOverlay.measureText(labelText).width;
            ctxOverlay.fillStyle = color;
            ctxOverlay.fillRect(scaledX - 2, scaledY - 28, textWidth + 12, 28);
            
            // Render Text Overlay
            ctxOverlay.fillStyle = '#FFFFFF';
            ctxOverlay.fillText(labelText, scaledX + 4, scaledY - 8);
        });

        // Evaluate Global Sterile Access Protocol
        if (isViolating) {
            statusPanel.innerText = "🚨 ACCESS DENIED: PROTOCOL VIOLATED";
            statusPanel.style.backgroundColor = "#4a0000";
            statusPanel.style.borderColor = "#FF3B30";
            statusPanel.style.color = "#ffcccc";
        } else {
            statusPanel.innerText = "✅ ACCESS GRANTED: PROCEED TO AIRLOCK";
            statusPanel.style.backgroundColor = "#003300";
            statusPanel.style.borderColor = "#34C759";
            statusPanel.style.color = "#ccffcc";
        }
    } else {
        statusPanel.innerText = "AWAITING SUBJECT...";
        statusPanel.style.backgroundColor = "transparent";
        statusPanel.style.borderColor = "#555";
        statusPanel.style.color = "#ffffff";
    }

    // Await next browser repaint
    requestAnimationFrame(processFrame);
}

// Bootstrap
window.onload = loadModel;
