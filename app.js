const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');
const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', { willReadFrequently: true });
const statusPanel = document.getElementById('status-panel');

let session;
const TARGET_SIZE = 640;

// Set overlay canvas dimensions to match the CSS dimensions
overlay.width = 640;
overlay.height = 480;

/**
 * Initialize the ONNX Runtime Session
 */
async function loadModel() {
    try {
        // Specify WASM execution provider for browser environments
        session = await ort.InferenceSession.create('./best.onnx', { executionProviders: ['wasm'] });
        statusPanel.innerText = "STANDBY: AWAITING CAMERA INITIALIZATION";
        startCamera();
    } catch (e) {
        console.error(e);
        statusPanel.innerText = "SYSTEM FAILURE: UNABLE TO MOUNT MODEL";
        statusPanel.style.borderColor = "#ff0000";
        statusPanel.style.color = "#ff0000";
    }
}

/**
 * Request and bind the webcam stream
 */
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            statusPanel.innerText = "AWAITING SUBJECT...";
            requestAnimationFrame(processFrame);
        };
    } catch (e) {
        console.error(e);
        statusPanel.innerText = "SYSTEM FAILURE: CAMERA ACCESS DENIED";
    }
}

/**
 * Standard Intersection over Union (IoU) calculation for NMS
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
 * Non-Maximum Suppression to filter overlapping predictions
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
 * Main execution loop per frame
 */
async function processFrame() {
    if (!session) return;

    // Draw the current video frame onto the 640x640 hidden processor canvas
    ctxProcessor.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
    const imageData = ctxProcessor.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;

    // Allocate memory for the Float32 tensor
    const float32Data = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);

    // Convert RGBA interleaved array to planar NCHW format [1, 3, 640, 640] and normalize to [0, 1]
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        float32Data[i]                                   = imageData[i * 4] / 255.0;     // R
        float32Data[i + TARGET_SIZE * TARGET_SIZE]       = imageData[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE]   = imageData[i * 4 + 2] / 255.0; // B
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_SIZE, TARGET_SIZE]);
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    // Execute Inference
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data; 
    
    // Parse the [1, 6, 8400] output tensor
    let rawBoxes = [];
    const elements = 8400;

    for (let i = 0; i < elements; i++) {
        const x = output[i];
        const y = output[i + elements];
        const w = output[i + 2 * elements];
        const h = output[i + 3 * elements];
        const scoreMask = output[i + 4 * elements];
        const scoreNoMask = output[i + 5 * elements];

        const maxScore = Math.max(scoreMask, scoreNoMask);

        if (maxScore > 0.5) {
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

    const finalBoxes = nonMaxSuppression(rawBoxes, 0.4);

    // Render operations
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    let isViolating = false;

    if (finalBoxes.length > 0) {
        finalBoxes.forEach(box => {
            // Scale coordinates from the 640x640 tensor model back to the 640x480 UI rendering resolution
            const scaleX = overlay.width / TARGET_SIZE;
            const scaleY = overlay.height / TARGET_SIZE;
            
            const scaledX = box.x * scaleX;
            const scaledY = box.y * scaleY;
            const scaledW = box.w * scaleX;
            const scaledH = box.h * scaleY;

            if (box.classId === 1) {
                isViolating = true;
                ctxOverlay.strokeStyle = '#ff0000';
                ctxOverlay.fillStyle = '#ff0000';
            } else {
                ctxOverlay.strokeStyle = '#00ff00';
                ctxOverlay.fillStyle = '#00ff00';
            }

            ctxOverlay.lineWidth = 3;
            ctxOverlay.strokeRect(scaledX, scaledY, scaledW, scaledH);
            
            const label = box.classId === 1 ? `VIOLATION [${box.score.toFixed(2)}]` : `COMPLIANT [${box.score.toFixed(2)}]`;
            ctxOverlay.fillRect(scaledX, scaledY - 25, label.length * 10, 25);
            
            ctxOverlay.fillStyle = '#ffffff';
            ctxOverlay.font = '16px monospace';
            ctxOverlay.fillText(label, scaledX + 5, scaledY - 8);
        });

        // Global Checkpoint Protocol Logic
        if (isViolating) {
            statusPanel.innerText = "🚨 ACCESS DENIED: STERILE PROTOCOL VIOLATED";
            statusPanel.style.backgroundColor = "#4a0000";
            statusPanel.style.borderColor = "#ff0000";
            statusPanel.style.color = "#ffcccc";
        } else {
            statusPanel.innerText = "✅ ACCESS GRANTED: PROCEED TO AIRLOCK";
            statusPanel.style.backgroundColor = "#003300";
            statusPanel.style.borderColor = "#00ff00";
            statusPanel.style.color = "#ccffcc";
        }
    } else {
        statusPanel.innerText = "AWAITING SUBJECT...";
        statusPanel.style.backgroundColor = "transparent";
        statusPanel.style.borderColor = "#555";
        statusPanel.style.color = "#ffffff";
    }

    // Loop
    requestAnimationFrame(processFrame);
}

// Initiate initialization sequence
window.onload = loadModel;
