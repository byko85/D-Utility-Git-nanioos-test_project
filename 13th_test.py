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

# 남은 결측치는 각 변수의 평균값으로 채움
X = X.fillna(X.mean())

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 모델 학습
k_value = 3
kmeans = KMeans(
    n_clusters=k_value,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# 평가 지표 계산
sil_score = silhouette_score(X_scaled, kmeans_labels)
sse = kmeans.inertia_

print("-" * 40)
print(f"🎯 K-Means 군집화 결과 (K={k_value})")
print("-" * 40)
print(f"▶ 실루엣 점수 (Silhouette Score): {sil_score:.4f}")
print(f"▶ SSE (오차제곱합, Inertia): {sse:.4f}")
print("-" * 40)

# PCA를 이용한 2차원 축소
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 그래프 시각화
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap='viridis',
    s=30,
    edgecolor='k'
)

plt.title(f"K-Means Clustering Result (K={k_value})", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()