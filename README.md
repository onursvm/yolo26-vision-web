# 🎯 YOLO26 Vision AI - Real-Time Web Object Detection

A real-time, AI-powered object detection web application using the **YOLO26 model**, with live webcam and camera support.

- 📷 **Webcam & Camera Support** – Real-time object detection directly through your browser or desktop camera.
- ⚡ **High-Speed Detection** – Ultra-fast and efficient object detection powered by YOLO26 Nano (`yolo26n.pt`).
- 🇹🇷 / 🇬🇧 **Multilingual Support** – Available in both Turkish and English.
- 🌙 / ☀️ **Dark & Light Mode** – Switch between modern dark and light themes.
- 📸 **Live HUD & Screenshot Capture** – Interactive real-time detection overlay with live latency, FPS counters, and instant snapshot download.

---

## 🚀 Quick Start (Localhost)

### 1. Requirements & Setup

Make sure you have Python 3.10+ installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/onursvm/yolo26-vision-web.git
cd yolo26-vision-web
pip install -r requirements.txt
```

### 2. Run the Web Application (Localhost)

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

### 3. Run the Desktop OpenCV Mode (Optional)

You can also run the direct OpenCV window version:

```bash
python main.py
```

- Press `q` to exit.
- Press `s` to save a screenshot.

---

## 📁 Project Structure

```
├── app.py                 # Flask web application & YOLO26 detection API
├── main.py                # Desktop OpenCV application
├── requirements.txt       # Python dependencies
├── yolo26n.pt             # YOLO26 Nano model weights
├── static/
│   ├── css/style.css      # Modern glassmorphism UI styles
│   └── js/app.js          # Client-side camera, HUD canvas, and state management
└── templates/
    └── index.html         # Web application template
```

---

## 🛠️ Tech Stack

- **Model:** Ultralytics YOLO26
- **Backend:** Python, Flask, Flask-SocketIO, OpenCV, PyTorch
- **Frontend:** Vanilla JavaScript (ES6+), Modern CSS3 with Design Tokens, HTML5 Canvas HUD