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
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

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
