import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로
file_path = r"C:\Users\User\Desktop\충북대 강의\머신런닝 프로그래밍\13th\11_data.csv"

# CSV 파일 불러오기
df = pd.read_csv(file_path)

# 문자형 컬럼 제외하고 숫자형 데이터만 추출
X = df.select_dtypes(include=[np.number])

# 식별 번호(id) 등 군집화에 방해되는 컬럼이 있다면 제거
if 'id' in X.columns:
    X = X.drop(columns=['id'])

# 통째로 비어있는 열 제거
X = X.dropna(axis=1, how='all')

# 남은 결측치는 해당 변수의 평균값으로 채움
X = X.fillna(X.mean())

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# K값 변화에 따른 K-Means 군집화 성능 비교
# =====================================================

k_range = range(2, 11)

sse_list = []
silhouette_list = []

print("=" * 50)
print("K값 변화에 따른 K-Means 분석 결과")
print("=" * 50)

for k_value in k_range:
    kmeans = KMeans(
        n_clusters=k_value,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 평가 지표 계산
    sil_score = silhouette_score(X_scaled, kmeans_labels)
    sse = kmeans.inertia_

    sse_list.append(sse)
    silhouette_list.append(sil_score)

    print("-" * 40)
    print(f"🎯 K-Means 군집화 결과 (K={k_value})")
    print("-" * 40)
    print(f"▶ 실루엣 점수 (Silhouette Score): {sil_score:.4f}")
    print(f"▶ SSE (오차제곱합, Inertia): {sse:.4f}")

print("-" * 40)

# 결과를 표 형태로 정리
result_df = pd.DataFrame({
    "K값": list(k_range),
    "SSE": sse_list,
    "Silhouette Score": silhouette_list
})

print("\n[K값별 비교 결과]")
print(result_df)

# =====================================================
# K값별 비교 결과표를 이미지로 시각화
# =====================================================

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 보기 좋게 반올림한 데이터프레임 생성
table_df = result_df.copy()
table_df["SSE"] = table_df["SSE"].round(4)
table_df["Silhouette Score"] = table_df["Silhouette Score"].round(4)

# 최적 K값 행 찾기
best_row_index = table_df["Silhouette Score"].idxmax()

# 표 그리기
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axis('off')

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc='center',
    loc='center'
)

# 표 스타일 조정
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# 헤더 스타일
for col in range(len(table_df.columns)):
    table[(0, col)].set_facecolor('#D9EAF7')
    table[(0, col)].set_text_props(weight='bold')

# 최적 K값 행 강조
# matplotlib table은 헤더가 0행, 데이터는 1행부터 시작
highlight_row = best_row_index + 1
for col in range(len(table_df.columns)):
    table[(highlight_row, col)].set_facecolor('#FFF2CC')
    table[(highlight_row, col)].set_text_props(weight='bold')

plt.title("K값별 비교 결과표", fontsize=14, fontweight='bold', pad=20)

# 이미지 저장
table_image_path = r"C:\Users\User\Desktop\충북대 강의\머신런닝 프로그래밍\13th\kmeans_k_value_table.png"
plt.savefig(table_image_path, dpi=300, bbox_inches='tight')

plt.show()

print(f"\n결과표 이미지 저장 완료: {table_image_path}")

# 실루엣 점수가 가장 높은 K값 찾기
best_k = result_df.loc[result_df["Silhouette Score"].idxmax(), "K값"]
best_silhouette = result_df["Silhouette Score"].max()

print("\n" + "=" * 50)
print(f"실루엣 점수 기준 최적 K값: {best_k}")
print(f"최고 실루엣 점수: {best_silhouette:.4f}")
print("=" * 50)

# =====================================================
# SSE 그래프 - Elbow Method
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(
    result_df["K값"],
    result_df["SSE"],
    marker='o'
)

plt.title("Elbow Method - SSE by K", fontsize=14, fontweight='bold')
plt.xlabel("K Value")
plt.ylabel("SSE")
plt.xticks(list(k_range))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================================
# Silhouette Score 그래프
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(
    result_df["K값"],
    result_df["Silhouette Score"],
    marker='o'
)

plt.title("Silhouette Score by K", fontsize=14, fontweight='bold')
plt.xlabel("K Value")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_range))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =====================================================
# 최적 K값으로 최종 K-Means 군집화
# =====================================================

final_kmeans = KMeans(
    n_clusters=int(best_k),
    random_state=42,
    n_init=10
)

final_labels = final_kmeans.fit_predict(X_scaled)

# PCA를 이용한 2차원 축소
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 최적 K값 기준 군집 결과 시각화
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=final_labels,
    cmap='viridis',
    s=30,
    edgecolor='k'
)

plt.title(f"K-Means Clustering Result - Best K={best_k}", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 결과 엑셀 저장
save_path = r"C:\Users\User\Desktop\충북대 강의\머신런닝 프로그래밍\13th\kmeans_k_value_result.xlsx"
result_df.to_excel(save_path, index=False)

print(f"\n결과 파일 저장 완료: {save_path}")