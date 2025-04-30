import os
import torch

class Config:
    # Dataset & Cache
    dataset_path = "data/UCF101"
    cache_path = "data/cache/data.arrow"
    test_cache_dir = "data/cache/test"
    label_map_path = "data/classInd.txt"

    # Test annotations (make sure this file exists)
    test_annotation_path = "data/test_annotations.csv" 

    # Image & Video Processing
    img_size = 112
    frames_per_clip = 16
    step_between_clips = 1
    batch_size = 8

    # Model
    num_classes = 101
    learning_rate = 1e-4
    epochs = 20
    hidden_dim = 256
    lstm_layers = 2
    dropout = 0.5

    # Results
    results_dir = "results"
    model_save_path = "logs/best_model.pth"
    checkpoint_path = "logs/checkpoint.pth"

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2

    # Scheduler / Optimizer
    weight_decay = 1e-5
    scheduler_step_size = 5
    scheduler_gamma = 0.1

    # Class labels (loaded from classInd.txt)
    class_names = []
    with open(label_map_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or len(line.split()) != 2:
                print(f"Skipping invalid line: {line}")
                continue
            idx, label = line.split()
            class_names.append(label)
