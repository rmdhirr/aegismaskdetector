# 🛡️ AegisMask: Edge-Native Sterile Checkpoint

![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-WebAssembly-blue?style=for-the-badge&logo=onnx)
![YOLO11n](https://img.shields.io/badge/Model-YOLO11n-00FFFF?style=for-the-badge)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AegisMask** is a zero-latency, entirely client-side computer vision system designed to simulate a real-world sterile facility checkpoint. By leveraging WebAssembly (WASM) and ONNX Runtime Web, this system runs a YOLO11n object detection model natively within the browser—bypassing the need for cloud backends, WebSockets, or STUN/TURN servers.

> **💡 Note:** I built this end-to-end in a single day as a masterclass example for my AI Engineering students. It serves as a structural blueprint for deploying production-grade, edge-native computer vision applications.

---

## ✨ Core Features

* **Edge-Native Inference:** Executes YOLO11n entirely on local hardware via WebAssembly, guaranteeing zero network latency and absolute data privacy.
* **Smart Telemetry & Logging:** Dynamically extracts violation snapshots using HTML5 Canvas and pushes active Desktop Web Notifications to security personnel.
* **Hardware-Linked Deterrents:** Utilizes the Web Audio API with a built-in state machine to trigger audio alarms for protocol violations without audio spam.
* **Auto-Scaling Coordinate Normalization:** Automatically detects and scales normalized model outputs (`0.0 - 1.0`) to exact UI pixel dimensions for perfect bounding box alignment.

---

## 🧠 Model Architecture & Training

This project utilizes a highly optimized **YOLO11 nano (YOLO11n)** architecture, chosen for its exceptional balance of mean Average Precision (mAP) and lightweight computational overhead suitable for browser execution.

### 📊 Dataset
The model was trained on a custom Mask/No-Mask dataset hosted on my Roboflow account. 
🔗 **[View the Dataset on Roboflow Here](https://app.roboflow.com/ramadhirras-workspace/mask-wearing-ukiwr/1)** 

### 🚀 The "Yolo Trainer" Pipeline
To streamline the training process for educational purposes, the model was trained using a custom, UI-driven Google Colab environment that I made. This pipeline abstracts away complex boilerplate, allowing students to securely input their Roboflow credentials and execute rigorous training loops.

🔗 **[Access the YOLO Trainer (Google Colab)](https://colab.research.google.com/drive/1Iv1yoEpJ1GfXf5zGnmgV4_hbMj6v6Q2o?usp=sharing)**

**Evaluation Metrics:**
* **Precision (P):** *0.9173*
* **Recall (R):** *0.8355*
* **mAP50:** *0.9355*
* **F1-Score:** *0.8745*

---

## 🛠️ System Components & Codebase Explanation

The repository is intentionally kept lightweight, consisting of two primary files to demonstrate clean separation of concerns.

### 1. `index.html` (The Interface & Viewport)
Acts as the DOM structure and CSS styling engine.
* **Dual-Pane Dashboard:** Utilizes Flexbox to separate the live camera feed from the dynamic violation log.
* **Absolute Z-Index Stacking:** Strategically stacks a transparent `<canvas>` directly over the `<video>` element to ensure bounding boxes render with 1-to-1 pixel accuracy over the physical camera feed.
* **Processor Canvas:** Contains a hidden `640x640` canvas used strictly to downscale and format frames for the ONNX tensor.

### 2. `app.js` (The Inference Engine)
The core logic script, handling everything from hardware initialization to matrix math.
* **Tensor Formatting:** Converts standard interleaved RGBA pixel arrays into the planar NCHW (Batch, Channels, Height, Width) Float32 arrays required by YOLO.
* **Non-Maximum Suppression (NMS):** Includes a custom JavaScript implementation of NMS and Intersection over Union (IoU) to filter overlapping, redundant bounding boxes.
* **State Machine:** Manages the `currentState` variable to ensure audio deterrents and Web Notifications only trigger *once* per incident, rather than firing 30 times a second.

---

## 🚀 Deployment (GitHub Pages)

Because this application relies exclusively on static assets (HTML, JS, ONNX), it can be hosted for free anywhere.

1. Fork or clone this repository.
2. Ensure your trained model is exported to ONNX format and named `best.onnx`.
3. Place `best.onnx` in the root directory.
4. Go to your repository settings -> **Pages**.
5. Deploy from the `main` branch. 
6. Open the generated URL, grant camera/notification permissions, and initialize the system!

---

### 👩‍💻 Author
**Ramadhirra Azzahra Putri**  
*Mechatronics & AI Engineer | Specializing in NLP*

---
*Built with ❤️ and excessive amounts of tea in Kuningan, West Java.*
