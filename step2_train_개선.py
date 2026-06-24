import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 384 -> 192
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 192 -> 96
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 96 -> 48
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 48 -> 24
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 24 -> 12
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 12 -> 24
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 24 -> 48
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 48 -> 96
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 96 -> 192
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 192 -> 384
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_model():
    ROOT_DIR = "./mvtec_ad"
    CATEGORY = "bottle"
    SAVE_PATH = "autoencoder_model_final_v1.pth"

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(200),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    model = ConvAutoencoder().to(device)

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 80
    best_loss = float("inf")

    print("모델 학습 시작...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 국부 결함 + 전체 구조를 같이 반영
            loss = 0.7 * l1_loss(outputs, images) + 0.3 * mse_loss(outputs, images)

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