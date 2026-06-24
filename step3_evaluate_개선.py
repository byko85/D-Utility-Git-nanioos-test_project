import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from step1_data_eda import MVTecDataset
from step2_train import ConvAutoencoder


def get_anomaly_score(error_map, mode="max_mix"):
    flat = error_map.flatten()

    if mode == "max":
        return np.max(flat)

    elif mode == "max_mix":
        return 0.85 * np.max(flat) + 0.15 * np.mean(flat)

    elif mode == "top01_mean":
        top_k = max(1, int(len(flat) * 0.001))  # 상위 0.1%
        return np.mean(np.sort(flat)[-top_k:])

    elif mode == "top02_mean":
        top_k = max(1, int(len(flat) * 0.002))  # 상위 0.2%
        return np.mean(np.sort(flat)[-top_k:])

    else:
        return np.max(flat)


def evaluate_performance(model, test_loader, device):
    model.eval()
    y_true = []
    y_scores = []

    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")
    print("현재 anomaly score mode: max_mix / blur: (5,5)")

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)

            error = torch.mean((images - outputs) ** 2, dim=1)
            error_map = error.squeeze().cpu().numpy()

            # blur를 너무 크게 주면 미세 결함이 희석될 수 있음
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)

            anomaly_score = get_anomaly_score(error_map, mode="max_mix")

            y_scores.append(anomaly_score)
            y_true.append(labels.item())

    auroc = roc_auc_score(y_true, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]

    if len(thresholds) == 0:
        best_threshold = 0.0
    elif best_idx >= len(thresholds):
        best_threshold = thresholds[-1]
    else:
        best_threshold = thresholds[best_idx]

    print("-" * 40)
    print("[전체 평가 결과]")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)

    return best_threshold


def visualize_anomaly(model, test_loader, device, threshold, num_samples=3):
    model.eval()
    samples_shown = 0

    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")

    with torch.no_grad():
        for images, labels, _ in test_loader:
            if labels.item() == 0:
                continue

            images = images.to(device)
            outputs = model(images)

            error = torch.mean((images - outputs) ** 2, dim=1)
            error_map = error.squeeze().cpu().numpy()
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)

            anomaly_score = get_anomaly_score(error_map, mode="max_mix")

            prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"

            error_map_norm = cv2.normalize(
                error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            heatmap = cv2.applyColorMap(error_map_norm, cv2.COLORMAP_JET)

            img_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
            out_np = outputs.squeeze().cpu().permute(1, 2, 0).numpy()

            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(
                (img_np * 255).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0
            )

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nScore: {anomaly_score:.4f} -> {prediction}")
            axes[1].imshow(out_np)
            axes[1].set_title("Reconstructed")
            axes[2].imshow(error_map, cmap="hot")
            axes[2].set_title("Error Map")
            axes[3].imshow(overlay)
            axes[3].set_title("Overlay Heatmap")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

            samples_shown += 1
            if samples_shown >= num_samples:
                break


if __name__ == "__main__":
    ROOT_DIR = "./mvtec_ad"
    CATEGORY = "bottle"
    MODEL_PATH = "autoencoder_model_final_v1.pth"

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(200),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    optimal_thresh = evaluate_performance(model, test_loader, device)
    visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3)