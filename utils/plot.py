import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm

try:
    from utils import plot_loss_curves, plot_roc_auc_curves
    from model import LipNet_Attn 
    from dataloader import create_dataloaders
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure model.py, dataloader.py, and utils.py are accessible.")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

ROOT_DIR = r"D:\lrw-v1\100_proc_rgb"
SAVE_DIR = SCRIPT_DIR

LATEST_CHECKPOINT = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
BEST_MODEL = os.path.join(SAVE_DIR, "best_model.pth")

BATCH_SIZE = 32
NUM_WORKERS = 0

RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
DROPOUT = 0.5


def plot_losses_from_history(checkpoint_path, save_path):
    print(f"Loading history from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if 'history' not in checkpoint:
        print("Error: 'history' key not found in the checkpoint.")
        return

    history = checkpoint['history']
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])

    if not train_losses or not val_losses:
        print("Error: Loss history is empty.")
        return

    val_losses_filtered = [v for v in val_losses if v is not None]
    train_losses_trimmed = train_losses[:len(val_losses_filtered)]
    
    print("Plotting loss curves...")
    plot_loss_curves(train_losses_trimmed, val_losses_filtered, save_path=save_path)


def get_predictions_for_roc(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(dataloader, desc="Getting Predictions for ROC", unit="batch")
    with torch.no_grad():
        for frames, labels in progress_bar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            
            probs = F.softmax(outputs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_roc_from_model(model_path, dataloader, num_classes, model_params, save_path):
    print(f"Loading best model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Best model file not found at {model_path}")
        return

    model = LipNet_Attn(
        num_classes=num_classes,
        rnn_hidden_size=model_params['rnn_hidden_size'],
        rnn_num_layers=model_params['rnn_num_layers'],
        dropout=model_params['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    true_labels, pred_probs = get_predictions_for_roc(model, dataloader, DEVICE)
    
    print("Plotting ROC-AUC curves...")
    plot_roc_auc_curves(true_labels, pred_probs, num_classes, save_path=save_path)


if __name__ == "__main__":
    loss_save_path = os.path.join(SAVE_DIR, "final_loss_curves.png")
    plot_losses_from_history(LATEST_CHECKPOINT, loss_save_path)

    print("\nLoading dataloaders to run new evaluation...")
    try:
        _, val_loader, _, num_classes = create_dataloaders(
            root_dir=ROOT_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}. Cannot plot ROC-AUC curve.")
        print(f"Check that the path '{ROOT_DIR}' is correct.")
        exit()

    if val_loader is None:
        print("Validation loader is None. Skipping ROC-AUC plot.")
    else:
        model_params = {
            'rnn_hidden_size': RNN_HIDDEN_SIZE,
            'rnn_num_layers': RNN_NUM_LAYERS,
            'dropout': DROPOUT
        }
        roc_save_path = os.path.join(SAVE_DIR, "final_roc_auc_curves.png")
        plot_roc_from_model(BEST_MODEL, val_loader, num_classes, model_params, roc_save_path)

    print(f"\nAll plotting tasks complete. Check {SAVE_DIR} for your .png files.")