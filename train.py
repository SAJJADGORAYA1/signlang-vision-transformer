import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from data import SignDataset, KeypointExtractor
from model import SignTransformer

# Config
VIDEO_DIR = "Words"
SINGLE_TEST_VIDEO = "body wash.mp4"
MAX_FRAMES = 30
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)  # 1662
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 2
MODEL_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 2
MODEL_PATH = "sign_transformer.pth"

def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, classes, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=classes))

def test_single_video(model, test_path, label_encoder):
    extractor = KeypointExtractor()
    cap = cv2.VideoCapture(test_path)
    frames = []
    while len(frames) < MAX_FRAMES and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extractor.extract(frame)
        frames.append(keypoints)
    cap.release()
    
    if len(frames) < MAX_FRAMES:
        padding = [np.zeros(KEYPOINT_DIM, dtype=np.float32) 
                  for _ in range(MAX_FRAMES - len(frames))]
        frames.extend(padding)
    frames = np.stack(frames)
    input_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        print(f"Predicted Label: {label_encoder.inverse_transform([pred_label])[0]}")

def test_all_videos(model, dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Testing on all videos in dataset...")
    true_labels, pred_labels = evaluate_model(model, dataloader)
    
    plot_confusion_matrix(true_labels, pred_labels, 
                         dataset.encoder.classes_,
                         title='Full Dataset Confusion Matrix')
    
    print("\nPer-video Results:")
    for i, video_path in enumerate(dataset.video_paths):
        true_label = dataset.labels[i]
        pred_label = dataset.encoder.inverse_transform([pred_labels[i]])[0]
        result = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"{os.path.basename(video_path)}: True={true_label}, Pred={pred_label} -> {result}")