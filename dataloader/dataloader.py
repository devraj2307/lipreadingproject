import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List

class LRWDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', augment: bool = False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment
        self.num_frames = 29
        self.img_size = (96, 96)

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"Creating {split} dataset...")
        self.samples = self._build_dataset()
        print(f"Found {len(self.samples)} samples in '{split}' split for {len(self.classes)} classes.")

    def _build_dataset(self) -> List[Tuple[Path, int]]:
        samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            split_dir = self.root_dir / class_name / self.split
            if not split_dir.exists():
                continue
            for video_dir in sorted(split_dir.iterdir()):
                if video_dir.is_dir():
                    if any(video_dir.glob('*.png')):
                        samples.append((video_dir, class_idx))
        return samples

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        frame_files = sorted(video_path.glob('*.png'))
        frames = []
        for frame_file in frame_files:
            try:
                with Image.open(frame_file) as img:
                    img = img.convert('RGB')
                    img = img.resize(self.img_size, Image.BILINEAR)
                    frame_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
                    frames.append(frame_np)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                frames.append(np.full((*self.img_size, 3), -1.0, dtype=np.float32))

        while len(frames) < self.num_frames:
            if not frames:
                print(f"Warning: No frames loaded for {video_path}, returning black video.")
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

        if self.augment and torch.rand(1) < 0.5:
            frames = torch.flip(frames, dims=[3])

        return frames, label

def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_dataset = LRWDataset(root_dir=root_dir, split='train', augment=True)
    val_dataset = LRWDataset(root_dir=root_dir, split='val', augment=False)
    test_dataset = LRWDataset(root_dir=root_dir, split='test', augment=False)

    val_loader = None
    if len(val_dataset) > 0:
         val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        print("Warning: Validation dataset is empty.")

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        print("Warning: Test dataset is empty.")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check the root_dir and split names.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, test_loader, num_classes