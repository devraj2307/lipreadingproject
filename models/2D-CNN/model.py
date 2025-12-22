import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Deep2DCNN(nn.Module):
    def __init__(self, input_channels=29, num_classes=100):
        super(Deep2DCNN, self).__init__()
        
        self.conv_initial = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(32)
        
        self.res_block1_1 = ResidualBlock(32, 32)
        self.res_block1_2 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_block2_1 = ResidualBlock(32, 64)
        self.res_block2_2 = ResidualBlock(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_block3_1 = ResidualBlock(64, 128)
        self.res_block3_2 = ResidualBlock(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten_size = 128 * 11 * 11
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        x = self.res_block1_1(x)
        x = self.res_block1_2(x)
        x = self.pool1(x)
        
        x = self.res_block2_1(x)
        x = self.res_block2_2(x)
        x = self.pool2(x)
        
        x = self.res_block3_1(x)
        x = self.res_block3_2(x)
        x = self.pool3(x)
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Deep2DCNN(input_channels=29, num_classes=100)
    
    print("=" * 50)
    print("Model Architecture")
    print("=" * 50)
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    print("\n" + "=" * 50)
    print("Testing Forward Pass")
    print("=" * 50)
    
    dummy_input = torch.randn(4, 29, 92, 92)
    print(f"Input shape:  {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 100)
    print("\nForward pass successful.")