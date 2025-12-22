import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

try:
    from model import LipNet_Attn
    from dataloader import LRWDataset
except ImportError:
    print("Error: Make sure model.py and dataloader.py are in this folder.")
    exit()

ROOT_DIR = r"D:\lrw-v1\100_proc_rgb" 
MODEL_PATH = "best_model.pth"
WORD_LIST_PATH = r"D:\lrw-v1\100_proc_rgb\100_rgb.txt"

RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 100 

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_per_class(model, dataloader, device, num_classes):
    model.eval()
    
    class_correct = torch.zeros(num_classes, dtype=torch.float32, device=device)
    class_total = torch.zeros(num_classes, dtype=torch.float32, device=device)

    total_predictions = 0
    total_correct = 0

    progress_bar = tqdm(dataloader, desc="Evaluating on Validation Set", unit="batch")
    
    with torch.no_grad():
        for frames, labels in progress_bar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            total_predictions += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            for i in range(num_classes):
                class_mask = (labels == i)
                class_total[i] += class_mask.sum()
                class_correct[i] += ((predicted == i) & class_mask).sum()

    progress_bar.close()
    
    class_accuracies = class_correct / (class_total + 1e-6) 
    
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    return class_accuracies.cpu().numpy(), class_correct.cpu().numpy(), class_total.cpu().numpy(), overall_accuracy

def load_word_list(path):
    if not os.path.exists(path):
        print(f"Error: Word list not found at {path}")
        print("Please edit the WORD_LIST_PATH variable.")
        return None
    with open(path, 'r') as f:
        words = f.read().strip().split(',')
    if len(words) != NUM_CLASSES:
        print(f"Warning: Loaded {len(words)} words, but NUM_CLASSES is {NUM_CLASSES}.")
    return words

if __name__ == "__main__":
    
    print(f"--- Starting Per-Class Validation ---")
    print(f"Using device: {DEVICE}")
    
    print(f"Loading word list from: {WORD_LIST_PATH}")
    word_list = load_word_list(WORD_LIST_PATH)
    if word_list is None:
        exit()

    print(f"Loading validation data from: {ROOT_DIR}")
    try:
        val_dataset = LRWDataset(root_dir=ROOT_DIR, split='val', augment=False)
        
        num_classes_loaded = len(val_dataset.classes)
        if num_classes_loaded != NUM_CLASSES:
            print(f"Warning: Dataloader found {num_classes_loaded} classes, but config expects {NUM_CLASSES}.")
            NUM_CLASSES = num_classes_loaded
            
        if len(val_dataset) == 0:
            print("Error: The validation dataset is empty. Check your dataset 'val' split.")
            exit()
            
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
            
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    print(f"Initializing model (LipNet_Attn)")
    model = LipNet_Attn(
        num_classes=NUM_CLASSES,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        dropout=DROPOUT
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at: {MODEL_PATH}")
        print("Please edit the MODEL_PATH variable in this script.")
        exit()
        
    print(f"Loading weights from: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("This often means your model config (RNN_HIDDEN_SIZE, etc.)")
        print("does not match the saved .pth file.")
        exit()
        
    model.to(DEVICE)
    
    print("Running evaluation...")
    
    accuracies, corrects, totals, overall_acc = evaluate_per_class(model, val_loader, DEVICE, NUM_CLASSES)
    
    results = []
    for i in range(NUM_CLASSES):
        word = word_list[i] if i < len(word_list) else f"Class_{i}"
        acc = accuracies[i]
        correct_count = int(corrects[i])
        total_count = int(totals[i])
        results.append((word, acc, correct_count, total_count))
        
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Evaluation Complete (on Validation Set) ---")
    print(f"  Model:            {MODEL_PATH}")
    print(f"  Overall Val Acc:  {overall_acc * 100:.2f}%\n")
    
    print("--- Per-Class Accuracy Report (Sorted) ---")
    print(f"{'Rank':<5} | {'Word':<15} | {'Accuracy':<10} | {'Correct/Total':<15}")
    print("-" * 50)
    
    for rank, (word, acc, correct_count, total_count) in enumerate(results, 1):
        acc_str = f"{acc * 100:.2f}%"
        count_str = f"{correct_count}/{total_count}"
        print(f"{rank:<5} | {word:<15} | {acc_str:<10} | {count_str:<15}")
        
    print("--------------------------------------------------\n")