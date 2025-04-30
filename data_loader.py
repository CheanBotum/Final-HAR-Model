import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage


class VideoDataset(Dataset):
    def __init__(self, dataset_path, label_map_path, cache_path, transform, sequence_length, img_size):
        self.dataset_path = dataset_path
        self.label_map_path = label_map_path
        self.cache_path = cache_path
        self.transform = transform
        self.sequence_length = sequence_length
        self.img_size = img_size

        self.label_map = self._load_labels()

        if os.path.exists(cache_path):
            self.table = pq.read_table(cache_path)
        else:
            self.table = self._build_arrow_table()
            pq.write_table(self.table, cache_path)

    def _load_labels(self):
        with open(self.label_map_path, 'r') as f:
            return {name.strip(): idx for idx, name in enumerate(f.readlines())}

    def _build_arrow_table(self):
        paths, labels = [], []
        for label in os.listdir(self.dataset_path):
            label_dir = os.path.join(self.dataset_path, label)
            if not os.path.isdir(label_dir):
                continue
            for video in os.listdir(label_dir):
                paths.append(os.path.join(label_dir, video))
                labels.append(self.label_map[label])
        return pa.table({"path": paths, "label": labels})

    def _load_video(self, index):
        row = self.table.slice(index, 1).to_pydict()
        video_path = row['path'][0]

        try:
            video, _, _ = read_video(video_path, pts_unit='sec')
            video = video.numpy()  # (T, H, W, C)
            return video
        except Exception as e:
            print(f"[Error] Failed to read video: {video_path}\n{e}")
            return []

    def _frame_to_pil(self, frame):
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]  # Trim to RGB
        assert frame.shape[-1] == 3, f"Expected 3-channel frame, got {frame.shape[-1]}"
        return ToPILImage()(frame)

    def _process_frames(self, video):
        frames = []
        for frame in video:
            if frame.shape[-1] != 3:
                frame = frame[:, :, :3]
            frame = self._frame_to_pil(frame)
            frame = self.transform(frame)
            frames.append(frame)

        # Pad or truncate
        if len(frames) < self.sequence_length:
            pad_count = self.sequence_length - len(frames)
            frames.extend([frames[-1]] * pad_count)
        else:
            frames = frames[:self.sequence_length]

        return torch.stack(frames)  # (T, C, H, W)

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        video = self._load_video(index)
        if len(video) == 0:
            print(f"[Warning] Empty video at index {index}, using dummy tensor")
            dummy = torch.zeros((self.sequence_length, 3, self.img_size, self.img_size))
            label = self.table.slice(index, 1).to_pydict()['label'][0]
            return dummy, torch.tensor(label)

        video_tensor = self._process_frames(video)
        label = self.table.slice(index, 1).to_pydict()['label'][0]
        return video_tensor, torch.tensor(label)
