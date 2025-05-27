# Human Activity Recognition (UCF101)

This project implements a **Human Activity Recognition (HAR) model** trained on the **UCF101** dataset using a **CNN-LSTM architecture**. A lightweight **Flask API** enables users to upload video files or provide YouTube URLs and receive the top-5 predicted action classes with annotated visual output.

# Final-HAR-Model

This project implements a **Human Activity Recognition (HAR) model** trained on the **UCF101** dataset using a **CNN-LSTM architecture**. A lightweight **Flask API** is provided for inference, allowing users to submit videos (or YouTube URLs) and receive action predictions and visual outputs.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Flask API](#flask-api)
- [Installation](#installation)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Example Request](#example-request)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

## Overview

- **Goal**: Classify short video clips into one of 101 human activities.
- **Input**: Local video file or YouTube link
- **Output**: Top-5 predicted actions with confidence + labeled preview video
- **Deployment**: Flask REST API


## Dataset

- **UCF101**: 13,320 videos across 101 categories (sports, daily life, gestures, etc.)
- Used exclusively for training and evaluation

## Model Architecture

- **CNN Backbone**: ResNet-18 for frame-wise feature extraction
- **Temporal Modeling**: LSTM layer to capture sequential motion
- **Classifier**: Fully-connected layer mapping to 101 classes
- **Input**: Sampled and preprocessed frames from video
- **Output**: Top-5 predicted activity labels with confidence


## Flask API

A RESTful API interface is provided for inference using video files or YouTube links. It processes video, runs the model, and returns predictions + annotated previews.


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/CheanBotum/Final-HAR-Model.git
cd Final-HAR-Model
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Add pretrained model weights
Place your trained UCF101 model file as:
```
logs/best_model.pth
```

### 5. Run the Flask app
```bash
python app.py
```

## Usage

Once the API is running (default: `http://127.0.0.1:5000`), you can send requests using `curl`, Postman, or Python.

### API Endpoints

| Method | Endpoint        | Input Type     | Description                  |
|--------|------------------|----------------|------------------------------|
| POST   | `/predict`       | File upload    | Upload a video file (MP4, AVI, etc.) |
| POST   | `/predict_url`   | Form URL       | Submit a YouTube video URL  |


### Example Request

####  Upload a video file:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "video=@examples/sample_video.mp4"
```

####  Send a YouTube URL:
```bash
curl -X POST http://127.0.0.1:5000/predict_url \
  -F "url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

#### Sample Response:
```json
{
  "top_prediction": "Basketball",
  "top_5": [
    {"label": "Basketball", "score": 0.84},
    {"label": "VolleyballSpiking", "score": 0.07},
    {"label": "Diving", "score": 0.04},
    {"label": "JumpRope", "score": 0.02},
    {"label": "SoccerJuggling", "score": 0.01}
  ],
  "preview_video": "/static/results/preview.mp4"
}
```


## Project Structure

```
human-activity-recognition/
├── app.py                 # Main Flask API
├── config.py              # Config loader
├── model.py               # CNN-LSTM model definition
├── utils.py               # Frame sampling, preprocessing, YouTube/ffmpeg helpers
├── transforms.py               
├── templates/
│   └── index.html         
├── static/
│   ├── uploads/           # Store Uploaded videos
│   ├── results/          
├── logs/
│   └── best_model.pth      
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- [Flask](https://flask.palletsprojects.com/) - Web API
- [PyTorch](https://pytorch.org/) - Model definition and inference
- [OpenCV](https://opencv.org/) - Frame extraction and image handling
- [pytube](https://github.com/pytube/pytube) - YouTube video download
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) - Format conversion
- NumPy, Matplotlib, tqdm, Pillow

##  Acknowledgments

- **UCF101 dataset** from the [University of Central Florida](https://www.crcv.ucf.edu/data/UCF101.php)
- [PyTorch](https://pytorch.org/) for deep learning
- [pytube](https://pytube.io/) for YouTube video handling
- [OpenCV](https://opencv.org/) for video and image processing
- [ffmpeg](https://ffmpeg.org/) for multimedia handling


** Built for research, prototyping, and real-time human activity recognition**

## Overview

- **Goal**: Recognize and classify human activities in video clips.
- **Input**: Local video file or YouTube link.
- **Output**: Top-5 predicted activity labels with confidence scores + annotated preview image or frame sequence.
- **Deployment**: Inference through a RESTful Flask API.

## Dataset

- **UCF101**: A dataset of 13,320 video clips from 101 human action classes including sports, daily activities, and gestures.
- All training and testing are done **only on UCF101**.  
- No external datasets (e.g., HMDB) are used.

## Model Architecture

- **Backbone**: ResNet-18 (used for per-frame feature extraction)
- **Temporal Model**: LSTM for sequential temporal modeling
- **Input**: Extracted and resized frames from video
- **Output**: Probability distribution over 101 action classes

## Flask API

A minimal Flask API wraps the trained model to enable video-based prediction via HTTP requests.

### Features

- Upload local video files
- Provide YouTube URLs (auto-downloaded via `pytube`)
- Returns:
  - Top-5 predictions
  - Annotated preview frame
