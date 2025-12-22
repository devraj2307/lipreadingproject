import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List
import os
import glob
from tqdm import tqdm

try:
    from model import LipNet_Attn
except ImportError:
    print("Error: model.py not found. Please put it in the same folder.")
    exit()

FINETUNE_DATA_DIR = "finetune_data"
ORIGINAL_MODEL_PATH = "best_model.pth"
FINETUNED_MODEL_PATH = "best_finetuned_model(entire).pth"

RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 100 

BATCH_SIZE = 8 
NUM_WORKERS = 0 
EPOCHS = 30     
LEARNING_RATE = 1e-5 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "ACCESS", "ACCUSED", "ACTUALLY", "AFTERNOON", "AGAIN", "AGREE", "ALREADY", "AMONG", "ASKING", "ATTACKS", 
    "BECOME", "BEING", "BETWEEN", "BRITISH", "BUILD", "CHANGE", "CHARGE", "CHIEF", "CHINA", "CLAIMS", 
    "CLEAR", "COUNCIL", "COUPLE", "DESCRIBED", "DIFFICULT", "EVERY", "EVERYTHING", "EXACTLY", "EXAMPLE", "EXPECTED", 
    "FAMILIES", "FAMILY", "FOCUS", "FOOTBALL", "FORCE", "GIVEN", "GREAT", "GROWTH", "HAPPENING", "HOMES", 
    "INSIDE", "INVESTMENT", "ISLAMIC", "JUSTICE", "LABOUR", "LARGE", "LEAST", "LEAVE", "MARKET", "MIDDLE", 
    "MILLIONS", "NUMBERS", "OPPOSITION", "ORDER", "PATIENTS", "PERIOD", "PLACE", "PLANS", "POLICY", "POLITICS", 
    "POSSIBLE", "POWER", "PRESIDENT", "PROBABLY", "PROBLEMS", "PROTECT", "QUESTION", "QUESTIONS", "REMEMBER", "RETURN", 
    "RIGHT", "RUNNING", "SCHOOL", "SCHOOLS", "SECRETARY", "SEEMS", "SERIOUS", "SERVICES", "SOCIETY", "SPECIAL", 
    "SPEECH", "SPENT", "STATE", "STILL", "SUPPORT", "TAKING", "TERMS", "THINK", "TOWARDS", "UNION", 
    "USING", "WAITING", "WANTS", "WEAPONS", "WELFARE", "WESTMINSTER", "WORKING", "WORST", "WOULD", "WRONG"
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

class FinetuneDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.num_frames = 29
        self.img_size = (96, 96)
        
        self.samples = self._build_dataset()
        print(f"Found {len(self.samples)} clips for {len(CLASS_TO_IDX)} classes in {root_dir}")

    def _build_dataset(self) -> List[Tuple[Path, int]]:
        samples = []
        if not self.root_dir.exists():
            print(f"Error: Finetune data directory not found: {self.root_dir}")
            return []
            
        for word_dir in self.root_dir.iterdir():
            if word_dir.is_dir() and word_dir.name in CLASS_TO_IDX:
                class_idx = CLASS_TO_IDX[word_dir.name]
                for clip_dir in word_dir.iterdir():
                    if clip_dir.is_dir():
                        samples.append((clip_dir, class_idx))
        return samples

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        frame_files = sorted(glob.glob(str(video_path / "frame_*.png")))
        frames = []
        for frame_file in frame_files:
            try:
                with Image.open(frame_file) as img:
                    img = img.convert('RGB')
                    frame_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
                    frames.append(frame_np)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                frames.append(np.full((*self.img_size, 3), -1.0, dtype=np.float32))

        while len(frames) < self.num_frames:
            if not frames:
                return torch.full((3, self.num_frames, *self.img_size), -1.0, dtype=torch.float32)
            frames.append(frames[-1])

        frames_np = np.stack(frames[:self.num_frames], axis=0) 
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2) 
        return frames_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        return frames, label

def main():
    print(f"Starting fine-tuning process on {DEVICE}")

    dataset = FinetuneDataset(root_dir=FINETUNE_DATA_DIR)
    if len(dataset) == 0:
        print("No data found. Did you run the data collection script first?")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    print(f"Loading pre-trained model from {ORIGINAL_MODEL_PATH}")
    model = LipNet_Attn(
        num_classes=NUM_CLASSES,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        dropout=DROPOUT
    )
    try:
        model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Make sure 'best_model.pth' is in this folder.")
        return
        
    model.to(DEVICE)
    print("Model loaded.")

    for param in model.conv_layers.parameters():
        param.requires_grad = True
        
    for param in model.rnn.parameters():
        param.requires_grad = True
    for param in model.attention.parameters():
        param.requires_grad = True
    for param in model.fc_layers.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Fine-Tuning ---")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for frames, labels in progress_bar:
            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs.detach(), 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            current_loss = total_loss / total_samples
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Complete: Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            print(f"  -> New best model saved to {FINETUNED_MODEL_PATH}")

    print("--- Fine-tuning finished ---")

if __name__ == "__main__":
    main()