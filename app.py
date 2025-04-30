# import os
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename

# import torch
# import cv2
# from PIL import Image

# from config import Config
# from model import CNN_LSTM
# from transforms import get_transforms

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load model once at startup
# model = CNN_LSTM(
#     num_classes=Config.num_classes,
#     sequence_length=Config.frames_per_clip,
#     hidden_dim=Config.hidden_dim,
#     lstm_layers=Config.lstm_layers,
#     dropout=Config.dropout
# ).to(Config.device)

# checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Preprocess video function
# def preprocess_video(video_path):
#     transform = get_transforms(Config.img_size)
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = Image.fromarray(frame)
#         frames.append(transform(frame))
    
#     cap.release()

#     total_frames = len(frames)
#     if total_frames < Config.frames_per_clip:
#         frames += [frames[-1]] * (Config.frames_per_clip - total_frames)
#     else:
#         step = total_frames // Config.frames_per_clip
#         frames = frames[::step][:Config.frames_per_clip]

#     frames = torch.stack(frames, dim=0)  # (sequence_length, 3, img_size, img_size)
#     frames = frames.unsqueeze(0).to(Config.device)  # (1, sequence_length, 3, img_size, img_size)
#     return frames

# # Prediction function
# def predict(video_path):
#     frames = preprocess_video(video_path)
#     with torch.no_grad():
#         outputs = model(frames)
#         pred = outputs.argmax(dim=1).item()
#         pred_class = Config.class_names[pred]
#     return pred_class

# # Routes
# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400

#     video = request.files['video']
#     filename = secure_filename(video.filename)
#     video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     video.save(video_path)

#     try:
#         prediction = predict(video_path)
#         response = {'prediction': prediction}
#     except Exception as e:
#         response = {'error': str(e)}

#     # Remove video after prediction to save space
#     os.remove(video_path)
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



# mutli-action

import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import torch
from model import load_model, predict_video
from youtube_dl import YoutubeDL

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model once
model = load_model('model_logs/enhanced_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def convert_to_mp4(input_path):
    output_path = input_path.rsplit('.', 1)[0] + ".mp4"
    if not input_path.endswith(".mp4"):
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        clip.close()
        os.remove(input_path)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower()
    unique_name = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    # Convert and update path to .mp4
    mp4_path = convert_to_mp4(filepath)
    prediction = predict_video(model, mp4_path, device)

    return jsonify({
        'prediction': prediction,
        'video_url': f"/static/uploads/{os.path.basename(mp4_path)}"
    })

@app.route('/predict_youtube', methods=['POST'])
def predict_youtube():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No YouTube URL provided'}), 400

    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio/best',
            'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], '%(id)s.%(ext)s'),
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info_dict)
        
        mp4_path = convert_to_mp4(downloaded_path)
        prediction = predict_video(model, mp4_path, device)

        return jsonify({
            'prediction': prediction,
            'video_url': f"/static/uploads/{os.path.basename(mp4_path)}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
