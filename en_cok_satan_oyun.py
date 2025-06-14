import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. CSV dosyasını oku
df = pd.read_csv("bestSelling_games.csv")

# 2. Gerekli sütunlar
df = df[["game_name", "user_defined_tags", "estimated_downloads"]].dropna()

# 3. estimated_downloads sayıya çevrilmişse gerek yok, ama emin olmak için dönüştürelim
df["estimated_downloads"] = df["estimated_downloads"].astype(int)

# 4. Tür bazlı indirme sayısı biriktirme
tag_downloads = defaultdict(int)

for _, row in df.iterrows():
    tags = [tag.strip() for tag in row["user_defined_tags"].split(",") if tag.strip()]
    for tag in tags:
        tag_downloads[tag] += row["estimated_downloads"]

# 5. En çok satan ilk 20 tür
top_tags = sorted(tag_downloads.items(), key=lambda x: x[1], reverse=True)[:20]
tags, downloads = zip(*top_tags)

# 6. Grafik
plt.figure(figsize=(12, 6))
plt.barh(tags[::-1], downloads[::-1], color="steelblue")
plt.xlabel("Toplam Tahmini İndirme Sayısı")
plt.title("En Çok Satan Oyun Türleri (Kümülatif Tahmini İndirmeye Göre)")
plt.tight_layout()
plt.show()
