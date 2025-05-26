from torchvision import transforms
from PIL import Image
import torch
import cv2
import os

def predict_image(model, image_path, config):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])
    input_tensor = transform(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    top5 = torch.topk(probs, 5)
    labels_scores = [(config.idx_to_label[str(i.item())], probs[i].item()) for i in top5.indices]
    return labels_scores[0][0], labels_scores

def predict_video(model, video_path, config):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < config.seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (config.img_size, config.img_size))
        frame = frame[:, :, ::-1]  # BGR to RGB
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize(mean=config.mean, std=config.std)(frame)
        frames.append(frame)
    cap.release()

    input_tensor = torch.stack(frames).unsqueeze(0).to(config.device)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    top5 = torch.topk(probs, 5)
    labels_scores = [(config.idx_to_label[str(i.item())], probs[i].item()) for i in top5.indices]

    # Save preview image
    preview_path = os.path.join("static", "preview.jpg")
    preview_frame = transforms.ToPILImage()(frames[min(3, len(frames) - 1)])
    preview_frame.save(preview_path)

    return labels_scores[0][0], labels_scores, preview_path

def extract_frames(video_path, out_dir, max_frames=12):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    idx = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 5 == 0:
            save_path = os.path.join(out_dir, f"frame_{frame_count:02d}.jpg")
            cv2.imwrite(save_path, frame)
            frame_count += 1
        idx += 1
    cap.release()
