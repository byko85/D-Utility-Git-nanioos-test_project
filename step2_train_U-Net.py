import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.up4 = Up(64, 32, 32)

        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g

    def create_window(self, window_size, channel):
        _1d = self.gaussian_window(window_size).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        return 1 - ssim_map.mean()


def add_gaussian_noise(x, std=0.05):
    noise = torch.randn_like(x) * std
    noisy = x + noise
    return torch.clamp(noisy, 0.0, 1.0)


def train_model():
    ROOT_DIR = "./mvtec_ad"
    CATEGORY = "bottle"
    SAVE_PATH = "unet_dae_bottle_best.pth"

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(220),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    model = UNetAutoencoder().to(device)

    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 80
    best_loss = float("inf")

    print("U-Net Denoising Autoencoder 학습 시작...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _, _ in train_loader:
            images = images.to(device)

            noisy_images = add_gaussian_noise(images, std=0.05)

            optimizer.zero_grad()
            outputs = model(noisy_images)

            loss_l1 = l1_loss(outputs, images)
            loss_ssim = ssim_loss(outputs, images)
            loss = 0.7 * loss_l1 + 0.3 * loss_ssim

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), SAVE_PATH)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    print(f"모델 저장 완료: {SAVE_PATH}")
    print(f"최저 Loss: {best_loss:.6f}")


if __name__ == "__main__":
    train_model()