import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor only once
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load class names from extended label map
with open("data/extended_label_map.json", "r") as f:
    EXTENDED_CLASSES = list(json.load(f).values())

def extract_frames_from_video(video_path, max_frames=16, resize=(224, 224)):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = vidcap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame).resize(resize)
        frames.append(pil_img)

    vidcap.release()
    return frames

def zero_shot_clip_predict(input_path, is_video=True, class_names=None):
    """
    Predict action label from image or video using zero-shot CLIP.

    Args:
        input_path (str): path to image or video file
        is_video (bool): True for video, False for image
        class_names (list): optional list of class names

    Returns:
        predicted_label (str)
        top5 (list of tuples): (label, probability)
    """
    classes = class_names or EXTENDED_CLASSES

    if is_video:
        frames = extract_frames_from_video(input_path)
    else:
        img = Image.open(input_path).convert("RGB")
        frames = [img]

    # Convert label names to readable text prompts
    text_inputs = [f"a photo of someone {label.replace('_', ' ').lower()}" for label in classes]
    inputs = clip_processor(text=text_inputs, images=frames, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # (num_frames, num_classes)
        probs = logits_per_image.softmax(dim=-1)

    mean_probs = probs.mean(dim=0)  # (num_classes,)
    top5_prob, top5_idx = mean_probs.topk(5)
    top5 = [(classes[idx], prob.item()) for idx, prob in zip(top5_idx, top5_prob)]
    predicted_label = top5[0][0]

    return predicted_label, top5
