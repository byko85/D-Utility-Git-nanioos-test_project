import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. 데이터 불러오기
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "11_data.csv")

print("현재 코드 위치:", BASE_DIR)
print("읽으려는 파일:", file_path)

try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="cp949")

print("\n데이터 불러오기 완료")
print("원본 데이터 크기:", df.shape)
print(df.head())


# ============================================================
# 2. 데이터 전처리
# ============================================================

# PCA 분석에 사용하지 않을 컬럼 제거
# id: 단순 식별번호
# diagnosis: 정답 라벨
# Unnamed: 32: 빈 컬럼
drop_cols = ["id", "diagnosis", "Unnamed: 32"]

df_features = df.drop(columns=drop_cols, errors="ignore")

# 숫자형 컬럼만 선택
X = df_features.select_dtypes(include=[np.number])

print("\nPCA에 사용할 데이터 크기:", X.shape)
print("PCA 사용 컬럼:")
print(X.columns.tolist())

# 결측치 처리
if X.isna().sum().sum() > 0:
    print("\n결측치가 있어 평균값으로 대체합니다.")
    X = X.fillna(X.mean())

print("\n최종 PCA 입력 데이터 크기:", X.shape)

if X.shape[1] < 2:
    raise ValueError("PCA를 수행하려면 숫자형 컬럼이 최소 2개 이상 필요합니다.")


# ============================================================
# 3. 표준화
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n표준화 완료")


# ============================================================
# 4. PCA 전체 수행
# ============================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

pca_result = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(variance_ratio))],
    "Explained Variance Ratio": variance_ratio,
    "Explained Variance Ratio (%)": variance_ratio * 100,
    "Cumulative Explained Variance": cumulative_variance,
    "Cumulative Explained Variance (%)": cumulative_variance * 100
})

print("\n[PCA 설명분산 결과]")
print(pca_result)


# ============================================================
# 5. 최종 PC 개수 선택
# ============================================================

threshold = 0.90

selected_pc_count = np.argmax(cumulative_variance >= threshold) + 1
selected_cumulative = cumulative_variance[selected_pc_count - 1]

print("\n==============================")
print("최종 선택 PC 개수:", selected_pc_count)
print(f"선택된 PC의 누적 설명분산: {selected_cumulative:.4f}")
print(f"선택된 PC의 누적 설명분산(%): {selected_cumulative * 100:.2f}%")
print("==============================")


# ============================================================
# 6. Scree Plot 생성
# ============================================================

x_components = range(1, len(variance_ratio) + 1)

plt.figure(figsize=(10, 6))

plt.bar(
    x_components,
    variance_ratio,
    alpha=0.6,
    label="Individual Explained Variance"
)

plt.plot(
    x_components,
    cumulative_variance,
    marker="o",
    linewidth=2,
    label="Cumulative Explained Variance"
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="90% Standard Line"
)

plt.axvline(
    x=selected_pc_count,
    linestyle="--",
    label=f"Selected PC Count = {selected_pc_count}"
)

plt.title("PCA Scree Plot")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.xticks(x_components)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

scree_plot_path = os.path.join(BASE_DIR, "pca_scree_plot.png")
plt.savefig(scree_plot_path, dpi=300)
plt.show()

# ============================================================
# 추가 1. PC1-PC2 2D 산점도
# ============================================================

# diagnosis 컬럼이 있으면 색상 구분용으로 사용
if "diagnosis" in df.columns:
    y_label = df["diagnosis"].map({"B": 0, "M": 1})
else:
    y_label = None

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_2d_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])

if y_label is not None:
    pca_2d_df["target"] = y_label.values

plt.figure(figsize=(8, 6))

if y_label is not None:
    for target_value in sorted(pca_2d_df["target"].unique()):
        temp = pca_2d_df[pca_2d_df["target"] == target_value]
        label_name = "Benign(B)" if target_value == 0 else "Malignant(M)"
        plt.scatter(temp["PC1"], temp["PC2"], label=label_name, alpha=0.7)
    plt.legend()
else:
    plt.scatter(pca_2d_df["PC1"], pca_2d_df["PC2"], alpha=0.7)

plt.title("PCA Projection: 30D to 2D")
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
plt.grid(True)
plt.tight_layout()

pca_2d_path = os.path.join(BASE_DIR, "pca_2d_scatter.png")
plt.savefig(pca_2d_path, dpi=300)
plt.show()

print("2D PCA 산점도 저장 완료:", pca_2d_path)


# ============================================================
# 추가 2. PC1-PC2-PC3 3D 산점도
# ============================================================

from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

pca_3d_df = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"])

if y_label is not None:
    pca_3d_df["target"] = y_label.values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

if y_label is not None:
    for target_value in sorted(pca_3d_df["target"].unique()):
        temp = pca_3d_df[pca_3d_df["target"] == target_value]
        label_name = "Benign(B)" if target_value == 0 else "Malignant(M)"
        ax.scatter(temp["PC1"], temp["PC2"], temp["PC3"], label=label_name, alpha=0.7)
    ax.legend()
else:
    ax.scatter(pca_3d_df["PC1"], pca_3d_df["PC2"], pca_3d_df["PC3"], alpha=0.7)

ax.set_title("PCA Projection: 30D to 3D")
ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})")
ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})")

plt.tight_layout()

pca_3d_path = os.path.join(BASE_DIR, "pca_3d_scatter.png")
plt.savefig(pca_3d_path, dpi=300)
plt.show()

print("3D PCA 산점도 저장 완료:", pca_3d_path)

print("\nScree Plot 저장 완료:", scree_plot_path)


# ============================================================
# 7. 선택한 PC 개수로 차원 축소
# ============================================================

pca_final = PCA(n_components=selected_pc_count)
X_reduced = pca_final.fit_transform(X_scaled)

reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(selected_pc_count)]
)

reduced_csv_path = os.path.join(BASE_DIR, "pca_dimension_reduced_data.csv")
reduced_df.to_csv(reduced_csv_path, index=False, encoding="utf-8-sig")

print("\n차원 축소 데이터 저장 완료:", reduced_csv_path)


# ============================================================
# 8. PCA 결과 저장
# ============================================================

pca_result_csv_path = os.path.join(BASE_DIR, "pca_scree_result.csv")
pca_result.to_csv(pca_result_csv_path, index=False, encoding="utf-8-sig")

print("PCA 설명분산 결과 CSV 저장 완료:", pca_result_csv_path)

try:
    pca_result_excel_path = os.path.join(BASE_DIR, "pca_scree_result.xlsx")
    pca_result.to_excel(pca_result_excel_path, index=False)
    print("PCA 설명분산 결과 Excel 저장 완료:", pca_result_excel_path)
except Exception as e:
    print("Excel 저장은 건너뜁니다.")
    print("사유:", e)


# ============================================================
# 9. 최종 요약 출력
# ============================================================

original_dim = X.shape[1]
reduced_dim = selected_pc_count
reduction_rate = (1 - reduced_dim / original_dim) * 100

print("\n[최종 요약]")
print(f"원본 변수 개수: {original_dim}개")
print(f"최종 선택 PC 개수: {selected_pc_count}개")
print(f"누적 설명분산: {selected_cumulative * 100:.2f}%")
print(f"차원 축소율: {reduction_rate:.2f}%")
print("\n분석 완료")