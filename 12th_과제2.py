import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. 현재 코드 위치 기준으로 데이터 파일 자동 찾기
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("현재 코드 위치:", BASE_DIR)
print("\n[현재 폴더 안의 파일 목록]")
for file in os.listdir(BASE_DIR):
    print(repr(file))

# csv, xlsx, xls 파일 자동 검색
data_files = (
    glob.glob(os.path.join(BASE_DIR, "*.csv")) +
    glob.glob(os.path.join(BASE_DIR, "*.xlsx")) +
    glob.glob(os.path.join(BASE_DIR, "*.xls"))
)

print("\n[찾은 데이터 파일 목록]")
for f in data_files:
    print(f)

if len(data_files) == 0:
    raise FileNotFoundError("현재 폴더에 csv, xlsx, xls 데이터 파일이 없습니다.")

# 11_data로 시작하는 파일 우선 선택
target_files = [
    f for f in data_files
    if os.path.basename(f).startswith("11_data")
]

if len(target_files) > 0:
    file_path = target_files[0]
else:
    file_path = data_files[0]

print("\n실제로 읽을 파일:", file_path)


# ============================================================
# 2. 데이터 불러오기 함수
# ============================================================
def load_data(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

        last_error = None

        for enc in encodings:
            try:
                print(f"CSV 읽기 시도 인코딩: {enc}")
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                last_error = e

        raise last_error

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


df = load_data(file_path)

print("\n데이터 불러오기 완료")
print("원본 데이터 크기:", df.shape)
print("\n[상위 5개 데이터]")
print(df.head())

print("\n[전체 컬럼명]")
print(df.columns.tolist())


# ============================================================
# 3. PCA에 사용하지 않을 컬럼 정리
# ============================================================
# Unnamed 컬럼은 대부분 빈 컬럼이므로 제거
unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
if len(unnamed_cols) > 0:
    print("\n[제거할 Unnamed 컬럼]")
    print(unnamed_cols)
    df = df.drop(columns=unnamed_cols)

# id는 고유번호라서 PCA 분석 변수로 부적절하므로 제거
if "id" in df.columns:
    print("\n'id' 컬럼 제거")
    df = df.drop(columns=["id"])

# diagnosis는 정답 라벨이므로 PCA 입력 변수에서는 제외
# 단, 나중에 PC1-PC2 산점도 색상 구분용으로 따로 보관
diagnosis = None

if "diagnosis" in df.columns:
    diagnosis = df["diagnosis"].copy()
    print("\n'diagnosis' 컬럼은 정답 라벨이므로 PCA 입력에서는 제외")
    df = df.drop(columns=["diagnosis"])


# ============================================================
# 4. 숫자형 데이터만 선택
# ============================================================
numeric_df = df.select_dtypes(include=[np.number]).copy()

print("\n숫자형 데이터 크기:", numeric_df.shape)
print("PCA 사용 숫자형 컬럼:")
print(numeric_df.columns.tolist())

if numeric_df.shape[1] < 2:
    raise ValueError("PCA를 수행하려면 숫자형 컬럼이 최소 2개 이상 필요합니다.")

# 모든 값이 결측치인 컬럼 제거
all_nan_cols = numeric_df.columns[numeric_df.isna().all()].tolist()

if len(all_nan_cols) > 0:
    print("\n[모든 값이 결측치라 제거할 컬럼]")
    print(all_nan_cols)
    numeric_df = numeric_df.drop(columns=all_nan_cols)

# 결측치가 일부 있는 경우 평균값으로 대체
missing_count = numeric_df.isna().sum().sum()

if missing_count > 0:
    print(f"\n결측치 총 개수: {missing_count}")
    print("결측치는 각 컬럼의 평균값으로 대체합니다.")
    numeric_df = numeric_df.fillna(numeric_df.mean())

print("\n최종 PCA 사용 데이터 크기:", numeric_df.shape)

if numeric_df.shape[0] < 2:
    raise ValueError("PCA를 수행하려면 데이터 행이 최소 2개 이상 필요합니다.")

if numeric_df.shape[1] < 2:
    raise ValueError("PCA를 수행하려면 숫자형 컬럼이 최소 2개 이상 필요합니다.")


# ============================================================
# 5. 표준화
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

print("\n표준화 완료")


# ============================================================
# 6. PCA 수행
# ============================================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)

pca_result = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_ratio))],
    "Explained Variance Ratio": explained_ratio,
    "Explained Variance Ratio (%)": explained_ratio * 100,
    "Cumulative Explained Variance": cumulative_ratio,
    "Cumulative Explained Variance (%)": cumulative_ratio * 100
})

print("\n[PCA 설명분산 결과]")
print(pca_result)


# ============================================================
# 7. PCA 결과 저장
# ============================================================
result_csv_path = os.path.join(BASE_DIR, "pca_scree_result.csv")
pca_result.to_csv(result_csv_path, index=False, encoding="utf-8-sig")

print("\nPCA 결과 CSV 저장 완료:", result_csv_path)

try:
    result_excel_path = os.path.join(BASE_DIR, "pca_scree_result.xlsx")
    pca_result.to_excel(result_excel_path, index=False)
    print("PCA 결과 Excel 저장 완료:", result_excel_path)
except Exception as e:
    print("Excel 저장은 건너뜁니다.")
    print("사유:", e)


# ============================================================
# 8. Scree Plot 생성
# ============================================================
plt.figure(figsize=(9, 6))

plt.plot(
    range(1, len(explained_ratio) + 1),
    explained_ratio,
    marker="o",
    label="Explained Variance Ratio"
)

plt.plot(
    range(1, len(cumulative_ratio) + 1),
    cumulative_ratio,
    marker="s",
    label="Cumulative Explained Variance"
)

plt.axhline(y=0.90, linestyle="--", label="90% Standard Line")

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.xticks(range(1, len(explained_ratio) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(BASE_DIR, "pca_scree_plot.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print("\nScree Plot 저장 완료:", plot_path)


# ============================================================
# 9. 최종 PC 개수 선택
# ============================================================
selected_pc_count = np.argmax(cumulative_ratio >= 0.90) + 1
selected_cumulative = cumulative_ratio[selected_pc_count - 1]

print("\n==============================")
print("최종 선택 PC 개수:", selected_pc_count)
print(f"선택된 PC의 누적 설명분산: {selected_cumulative:.4f}")
print(f"선택된 PC의 누적 설명분산(%): {selected_cumulative * 100:.2f}%")
print("==============================")


# ============================================================
# 10. 선택한 PC 개수로 차원 축소
# ============================================================
pca_final = PCA(n_components=selected_pc_count)
X_pca_final = pca_final.fit_transform(X_scaled)

pca_final_df = pd.DataFrame(
    X_pca_final,
    columns=[f"PC{i+1}" for i in range(selected_pc_count)]
)

reduced_csv_path = os.path.join(BASE_DIR, "pca_dimension_reduced_data.csv")
pca_final_df.to_csv(reduced_csv_path, index=False, encoding="utf-8-sig")

print("\n차원 축소 데이터 CSV 저장 완료:", reduced_csv_path)

try:
    reduced_excel_path = os.path.join(BASE_DIR, "pca_dimension_reduced_data.xlsx")
    pca_final_df.to_excel(reduced_excel_path, index=False)
    print("차원 축소 데이터 Excel 저장 완료:", reduced_excel_path)
except Exception as e:
    print("Excel 저장은 건너뜁니다.")
    print("사유:", e)


# ============================================================
# 11. PC1-PC2 산점도 생성
# ============================================================
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(8, 6))

    if diagnosis is not None:
        labels = diagnosis.astype(str)

        for label in sorted(labels.unique()):
            idx = labels == label
            plt.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                label=f"diagnosis={label}",
                alpha=0.7
            )

        plt.legend()
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Result: PC1 vs PC2")
    plt.grid(True)
    plt.tight_layout()

    scatter_path = os.path.join(BASE_DIR, "pca_pc1_pc2_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.show()

    print("PC1-PC2 산점도 저장 완료:", scatter_path)


# ============================================================
# 12. 보고서용 문장 자동 생성
# ============================================================
original_dim = numeric_df.shape[1]
reduced_dim = selected_pc_count
reduction_rate = (1 - reduced_dim / original_dim) * 100

report_text = f"""
[보고서에 사용할 문장]

본 분석에서는 PCA를 활용하여 다차원 데이터를 저차원 주성분으로 축소하였다.
분석에 사용한 원본 데이터는 총 {original_dim}개의 숫자형 변수로 구성되어 있었다.

PCA 수행 전 변수 간 단위 차이로 인한 영향을 줄이기 위해 StandardScaler를 이용하여 표준화를 수행하였다.
이후 전체 숫자형 변수에 대해 PCA를 수행하고, 각 주성분의 설명분산비율과 누적 설명분산비율을 확인하였다.

Scree Plot 분석 결과, 주성분이 증가할수록 개별 설명분산비율은 점차 감소하는 경향을 보였다.
본 과제에서는 누적 설명분산 90% 이상을 기준으로 최종 PC 개수를 선정하였다.

분석 결과 PC1부터 PC{selected_pc_count}까지의 누적 설명분산은 {selected_cumulative * 100:.2f}%로 나타났다.
따라서 최종 주성분 개수는 {selected_pc_count}개로 선정하였다.

결과적으로 원본 {original_dim}개의 변수를 {reduced_dim}개의 주성분으로 축소하였으며,
차원 축소율은 약 {reduction_rate:.2f}%이다.
이는 원본 데이터의 주요 정보를 대부분 유지하면서 변수 개수를 줄인 결과라고 판단할 수 있다.

따라서 본 분석에서는 PC{selected_pc_count}까지를 최종 선택하여 차원 축소를 진행하였다.
"""

print(report_text)

report_path = os.path.join(BASE_DIR, "pca_report_text.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print("보고서용 문장 저장 완료:", report_path)