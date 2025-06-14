import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Veriyi yükle
df = pd.read_csv("bestSelling_games.csv")

# 2. Eksik değerleri kontrol et
print("Eksik değerler:\n", df.isnull().sum())

# 3. Kritik sütunlardaki eksik verileri temizle
df = df.dropna(subset=["user_defined_tags", "estimated_downloads", "price"])

# 4. estimated_downloads sayıya çevrilmiş mi kontrol et
df["estimated_downloads"] = df["estimated_downloads"].astype(int)

# 5. Desteklenen dil sayısını hesapla
df["language_count"] = df["supported_languages"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0)

# 6. Türleri liste olarak sakla
df["tags_list"] = df["user_defined_tags"].apply(lambda x: [tag.strip() for tag in str(x).split(",")])

# 7. Aykırı değerleri görselleştir (örnek: downloads)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["estimated_downloads"])
plt.title("Aykırı İndirme Sayıları")
plt.show()

# 8. Sayısal sütunlar için istatistiksel özet
print(df[["price", "rating", "difficulty", "length", "estimated_downloads", "language_count"]].describe())

# 9. Rating dağılım grafiği
plt.figure(figsize=(6, 4))
sns.histplot(df["rating"], bins=20, kde=True)
plt.title("Oyun Puanı Dağılımı (Rating)")
plt.xlabel("Rating")
plt.tight_layout()
plt.show()

# 10. Temizlik öncesi-sonrası karşılaştırma
print("Veri satır sayısı (temizlenmiş):", df.shape[0])
