# Enhanced Human Activity Recognition Through a CNN-LSTM Hybrid Model 

This project implements a **Human Activity Recognition (HAR) system** trained on the **UCF101** dataset using a **CNN-LSTM** architecture with **attention**. It features a powerful **Flask-based web app** that allows users to:

- Upload video files or paste online video links 
- Predict **multiple actions** occurring across video segments
- View a **video player with real-time action display**
- See top-5 prediction chart per segment
- Automatically fallback to **CLIP-based zero-shot prediction** for unseen or unknown actions

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Flask App Features](#flask-app-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Overview

- **Goal**: Recognize and classify human activities in video clips.
- **Input**: Local video file or any online video link (YouTube, TikTok, etc.)
- **Output**: Segment-wise predicted actions with confidence and top-5 chart
- **Backup Prediction**: Uses CLIP zero-shot inference for unknown actions

## Dataset

- **UCF101**: 13,320 videos across 101 categories (sports, daily life, gestures, etc.)
- Used for training and evaluation
- Optional: Extend using `extended_label_map.json` for zero-shot support

## Model Architecture

- **CNN**: ResNet-18 for spatial feature extraction
- **LSTM**: Bidirectional LSTM with attention for temporal modeling
- **Attention**: Focus mechanism on relevant time steps
- **Input**: Sequences of resized frames (e.g., 16 frames per clip)
- **Output**: Top-1 and Top-5 predictions with probabilities

## Flask App Features

- Upload local video file
- Paste public video URL (supports many platforms)
- Automatically segments long video into overlapping clips
- Predict action for each segment
- Show timeline overlay if multiple actions are present
- Fallback to CLIP model for unknown/unseen actions

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/CheanBotum/Final-HAR-Model.git
cd Final-HAR-Model
```

### 2. Create and activate a virtual environment
```bash
conda create -p ./.conda python=3.9 -y
conda activate ./.conda
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Add pretrained model weights
```bash
# Place your trained UCF101 model here:
logs/best_model.pth
```

### 5. Run the Flask app
```bash
python app.py
```

## Usage

Launch the web interface at: [http://localhost:5000](http://localhost:5000)

You can upload a video or enter a video URL.

#### Example curl request
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "video=@examples/sample.mp4"
```

#### Example JSON Response
```json
{
  "segment_predictions": [
    {"label": "Running", "confidence": 0.88, "top5": [["Running", 0.88], ["Walking", 0.06], ...]},
    {"label": "JumpRope", "confidence": 0.91, "top5": [["JumpRope", 0.91], ["Skipping", 0.05], ...]}
  ],
  "video_url": "/static/uploads/unique_video.mp4"
}
```

## Project Structure
```
Final-HAR-Model/
├── app.py                   # Flask web app (video handling, inference, UI)
├── config.py                # Configurations (paths, hyperparams)
├── model.py                 # CNN-LSTM with attention
├── transforms.py            # Preprocessing transforms
├── zero_shot_clip_predict.py # CLIP-based fallback for unseen actions
├── extended_label_map.json  # Optional extended labels for zero-shot
├── static/
│   ├── uploads/             # Uploaded videos
│   └── processed_videos/   # Saved annotated outputs
├── templates/
│   └── index.html           # Web UI
├── logs/
│   └── best_model.pth       # Trained model checkpoint
└── requirements.txt
```

## Dependencies

```bash
pip install -r requirements.txt
```

### Key Libraries

- **Flask** – lightweight web framework
- **PyTorch** – model training and inference
- **OpenCV** – video handling
- **transformers** – CLIP-based zero-shot prediction
- **yt_dlp** – support for downloading videos from any platform
- **ffmpeg-python** – for format conversion

## Acknowledgments

- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)

---

Built for Final Year Thesis research 
