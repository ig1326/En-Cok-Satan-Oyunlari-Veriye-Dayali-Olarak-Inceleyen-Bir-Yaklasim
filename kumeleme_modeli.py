import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Veriyi oku
df = pd.read_csv("bestSelling_games.csv")

# 2. Temizleme
df = df.dropna(subset=["price", "rating", "difficulty", "length", "supported_languages"])
df["language_count"] = df["supported_languages"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0)

# 3. Özellikleri seç
features = df[["price", "rating", "difficulty", "length", "language_count"]]

# 4. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 5. Elbow metodu ile ideal küme sayısını bul
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), sse, marker="o")
plt.title("Elbow Yöntemi: Optimum Küme Sayısı")
plt.xlabel("Küme Sayısı")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.grid(True)
plt.show()

# 6. En iyi küme sayısı seçildikten sonra (örneğin k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_scaled)

# 7. Küme içeriğini analiz et
cluster_summary = df.groupby("cluster")[["price", "rating", "difficulty", "length", "language_count"]].mean()
print("\nKüme Özellik Ortalamaları:\n")
print(cluster_summary)

# 8. PCA ile görselleştirme (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster"], palette="Set2")
plt.title("Kümeleme Sonuçlarının 2D Görselleştirmesi (PCA)")
plt.xlabel("Bileşen 1")
plt.ylabel("Bileşen 2")
plt.legend(title="Küme")
plt.tight_layout()
plt.show()
