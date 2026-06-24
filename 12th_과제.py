import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, "11_data.csv")

print("현재 코드 위치:", BASE_DIR)
print("읽으려는 파일:", file_path)

df = pd.read_csv(file_path)

print("데이터 불러오기 완료")
print("데이터 크기:", df.shape)
print(df.head())

# =========================
# 2. 숫자형 컬럼만 선택
# =========================
numeric_df = df.select_dtypes(include=[np.number])

print("숫자형 데이터 크기:", numeric_df.shape)
print("사용 컬럼:", numeric_df.columns.tolist())

# 결측치 제거
numeric_df = numeric_df.dropna()

# =========================
# 3. 표준화
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# =========================
# 4. PCA 수행
# =========================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)

# =========================
# 5. 결과표 생성
# =========================
pca_result = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_ratio))],
    "Explained Variance Ratio": explained_ratio,
    "Cumulative Explained Variance": cumulative_ratio
})

print("\n[PCA 설명분산 결과]")
print(pca_result)

# 엑셀 저장
pca_result.to_excel("pca_scree_result.xlsx", index=False)

# =========================
# 6. Scree Plot 그리기
# =========================
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(explained_ratio) + 1),
    explained_ratio,
    marker='o',
    label="Explained Variance Ratio"
)

plt.plot(
    range(1, len(cumulative_ratio) + 1),
    cumulative_ratio,
    marker='s',
    label="Cumulative Explained Variance"
)

plt.axhline(y=0.90, linestyle='--', label="90% 기준선")

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.xticks(range(1, len(explained_ratio) + 1))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=300)
plt.show()

# =========================
# 7. 90% 이상 되는 PC 개수 자동 선택
# =========================
selected_pc_count = np.argmax(cumulative_ratio >= 0.90) + 1

print(f"\n최종 선택 PC 개수: {selected_pc_count}개")
print(f"선택된 PC의 누적 설명분산: {cumulative_ratio[selected_pc_count-1]:.4f}")

# =========================
# 8. 선택한 PC 개수로 차원 축소
# =========================
pca_final = PCA(n_components=selected_pc_count)
X_pca_final = pca_final.fit_transform(X_scaled)

pca_final_df = pd.DataFrame(
    X_pca_final,
    columns=[f"PC{i+1}" for i in range(selected_pc_count)]
)

pca_final_df.to_excel("pca_dimension_reduced_data.xlsx", index=False)

print("\n차원 축소 데이터 저장 완료: pca_dimension_reduced_data.xlsx")