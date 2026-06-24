import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.defect_types = []

        if is_train:
            train_good_dir = os.path.join(root_dir, category, "train", "good")
            for fname in sorted(os.listdir(train_good_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.image_paths.append(os.path.join(train_good_dir, fname))
                    self.labels.append(0)
                    self.defect_types.append("good")
        else:
            test_dir = os.path.join(root_dir, category, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue

                for fname in sorted(os.listdir(defect_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.image_paths.append(os.path.join(defect_dir, fname))
                        self.labels.append(0 if defect_type == "good" else 1)
                        self.defect_types.append(defect_type)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, defect_type


if __name__ == "__main__":
    ROOT_DIR = "./mvtec_ad"
    CATEGORY = "bottle"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(220),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=transform)
    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)

    test_good = sum(1 for x in test_dataset.labels if x == 0)
    test_ng = sum(1 for x in test_dataset.labels if x == 1)

    print("=== MVTec AD bottle 데이터 확인 ===")
    print(f"학습 데이터 개수(정상): {len(train_dataset)}장")
    print(f"테스트 데이터 개수(전체): {len(test_dataset)}장")
    print(f"테스트 정상 개수: {test_good}장")
    print(f"테스트 불량 개수: {test_ng}장")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sample_indices = [0, min(1, len(test_dataset) - 1), min(2, len(test_dataset) - 1)]

    for ax, idx in zip(axes, sample_indices):
        img, label, defect_type = test_dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(f"Label: {label}\nType: {defect_type}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()