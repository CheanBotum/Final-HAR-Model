# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import os, uuid, torch, cv2, subprocess, json
# from PIL import Image, ImageDraw, ImageFont
# from torchvision import transforms
# from model import CNN_LSTM
# from config import Config
# import yt_dlp

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# # Load class names from label map
# with open(Config.label_map_path, "r") as f:
#     Config.class_names = json.load(f)

# # Load model
# model = CNN_LSTM(
#     num_classes=Config.num_classes,
#     sequence_length=Config.frames_per_clip,
#     hidden_dim=Config.hidden_dim,
#     lstm_layers=Config.lstm_layers,
#     dropout=Config.dropout
# ).to(Config.device)

# checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
# model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
# model.eval()

# # Image transform
# transform = transforms.Compose([
#     transforms.Resize((Config.img_size, Config.img_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# def convert_to_mp4(input_path):
#     output_path = input_path.rsplit('.', 1)[0] + '_converted.mp4'
#     try:
#         subprocess.run(
#             ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path],
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
#         )
#         return output_path if os.path.exists(output_path) else None
#     except Exception as e:
#         print(f"[FFMPEG ERROR] {e}")
#         return None

# def download_youtube_video(url):
#     output_pattern = os.path.join(UPLOAD_FOLDER, '%(id)s.%(ext)s')
#     ydl_opts = {
#         'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]/best',
#         'merge_output_format': 'mp4',
#         'outtmpl': output_pattern,
#         'quiet': True,
#         'noplaylist': True,
#     }
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             final_path = os.path.join(UPLOAD_FOLDER, f"{info['id']}.mp4")
#             return final_path if os.path.exists(final_path) else None
#     except Exception as e:
#         print(f"[YT_DLP ERROR] {e}")
#         return None

# def extract_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(frame_rgb)
#         tensor = transform(image)
#         frames.append(tensor)
#     cap.release()

#     if not frames:
#         raise ValueError("No frames found in video.")

#     total = len(frames)
#     if total < Config.frames_per_clip:
#         frames += [frames[-1]] * (Config.frames_per_clip - total)
#     else:
#         step = max(1, total // Config.frames_per_clip)
#         frames = frames[::step][:Config.frames_per_clip]

#     return torch.stack(frames).unsqueeze(0).to(Config.device)

# def predict_tensor_input(tensor):
#     with torch.no_grad():
#         output = model(tensor)
#         prob = torch.nn.functional.softmax(output, dim=1)
#         top5 = torch.topk(prob, 5)
#         top_preds = [
#             (Config.class_names[str(idx.item())], round(score.item(), 3))
#             for score, idx in zip(top5.values[0], top5.indices[0])
#         ]
#         return top_preds

# def save_prediction_image(base_image, predictions):
#     draw = ImageDraw.Draw(base_image)
#     try:
#         font = ImageFont.truetype("arial.ttf", 24)
#     except:
#         font = ImageFont.load_default()

#     for i, (label, score) in enumerate(predictions):
#         draw.text((10, 10 + i * 30), f"{i+1}. {label}: {score}", font=font, fill=(255, 255, 0))

#     path = os.path.join(UPLOAD_FOLDER, f"pred_{uuid.uuid4().hex}.jpg")
#     base_image.save(path)
#     return path

# def predict_video(video_path):
#     if not video_path.lower().endswith(".mp4"):
#         video_path = convert_to_mp4(video_path)
#         if not video_path:
#             raise RuntimeError("Video conversion failed")

#     tensor_input = extract_frames(video_path)
#     predictions = predict_tensor_input(tensor_input)

#     # Get preview frame
#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Failed to read video frame")

#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     image_path = save_prediction_image(image, predictions)

#     return predictions, image_path, video_path

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'video' in request.files and request.files['video'].filename:
#             file = request.files['video']
#             ext = os.path.splitext(file.filename)[1].lower()
#             raw_path = os.path.join(UPLOAD_FOLDER, secure_filename(f"{uuid.uuid4().hex}{ext}"))
#             file.save(raw_path)
#             predictions, image_path, final_video_path = predict_video(raw_path)

#         elif 'youtube_url' in request.form and request.form['youtube_url']:
#             video_path = download_youtube_video(request.form['youtube_url'])
#             if not video_path:
#                 return jsonify({'error': 'YouTube download failed'}), 500
#             predictions, image_path, final_video_path = predict_video(video_path)

#         else:
#             return jsonify({'error': 'No valid input'}), 400

#         return jsonify({
#             'prediction': predictions[0][0],
#             'predictions': predictions,
#             'topk': [{'label': label, 'score': score} for label, score in predictions],
#             'image_url': '/' + image_path.replace('\\', '/'),
#             'video_url': '/' + final_video_path.replace('\\', '/')
#         })

#     except Exception as e:
#         print("[ERROR]", str(e))
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# =============================================

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import torch
import cv2
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model import CNN_LSTM
from config import Config
import yt_dlp

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max 500 MB upload

# Load class names once at startup
with open(Config.label_map_path, "r") as f:
    Config.class_names = json.load(f)

# Load your trained model once
model = CNN_LSTM(
    num_classes=Config.num_classes,
    sequence_length=Config.frames_per_clip,
    hidden_dim=Config.hidden_dim,
    lstm_layers=Config.lstm_layers,
    dropout=Config.dropout
).to(Config.device)

checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
model.eval()

# Preprocessing transform for frames
transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def convert_to_mp4(input_path):
    output_path = input_path.rsplit('.', 1)[0] + '_converted.mp4'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"[FFMPEG ERROR] {e}")
        return None

def download_youtube_video(url):
    output_pattern = os.path.join(UPLOAD_FOLDER, '%(id)s.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_pattern,
        'quiet': True,
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_path = os.path.join(UPLOAD_FOLDER, f"{info['id']}.mp4")
            return final_path if os.path.exists(final_path) else None
    except Exception as e:
        print(f"[YT_DLP ERROR] {e}")
        return None

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        tensor = transform(image)
        frames.append(tensor)
    cap.release()

    if not frames:
        raise ValueError("No frames found in video.")

    total = len(frames)
    if total < Config.frames_per_clip:
        frames += [frames[-1]] * (Config.frames_per_clip - total)
    else:
        step = max(1, total // Config.frames_per_clip)
        frames = frames[::step][:Config.frames_per_clip]

    return torch.stack(frames).unsqueeze(0).to(Config.device)

def predict_tensor_input(tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        top5 = torch.topk(prob, 5)
        top_preds = [
            (Config.class_names[str(idx.item())], round(score.item(), 3))
            for score, idx in zip(top5.values[0], top5.indices[0])
        ]
        return top_preds

def save_prediction_image(base_image, predictions):
    draw = ImageDraw.Draw(base_image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for i, (label, score) in enumerate(predictions):
        draw.text((10, 10 + i * 30), f"{i+1}. {label}: {score}", font=font, fill=(255, 255, 0))

    path = os.path.join(UPLOAD_FOLDER, f"pred_{uuid.uuid4().hex}.jpg")
    base_image.save(path)
    return path

def predict_video(video_path):
    if not video_path.lower().endswith(".mp4"):
        video_path = convert_to_mp4(video_path)
        if not video_path:
            raise RuntimeError("Video conversion failed")

    tensor_input = extract_frames(video_path)
    predictions = predict_tensor_input(tensor_input)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read video frame")

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_path = save_prediction_image(image, predictions)

    return predictions, image_path, video_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Clear old uploaded files before new upload (optional)
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

        if 'video' in request.files and request.files['video'].filename:
            file = request.files['video']
            ext = os.path.splitext(file.filename)[1].lower()
            raw_path = os.path.join(UPLOAD_FOLDER, secure_filename(f"{uuid.uuid4().hex}{ext}"))
            file.save(raw_path)
            predictions, image_path, final_video_path = predict_video(raw_path)

        elif 'youtube_url' in request.form and request.form['youtube_url']:
            video_path = download_youtube_video(request.form['youtube_url'])
            if not video_path:
                return jsonify({'error': 'YouTube download failed'}), 500
            predictions, image_path, final_video_path = predict_video(video_path)

        else:
            return jsonify({'error': 'No valid input'}), 400

        return jsonify({
            'prediction': predictions[0][0],
            'predictions': predictions,
            'topk': [{'label': label, 'score': score} for label, score in predictions],
            'image_url': '/' + image_path.replace('\\', '/'),
            'video_url': '/' + final_video_path.replace('\\', '/')
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
