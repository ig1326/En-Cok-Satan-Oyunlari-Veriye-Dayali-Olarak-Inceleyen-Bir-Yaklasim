import pandas as pd
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv("bestSelling_games.csv")

# Dil sayısını hesapla
df["language_count"] = df["supported_languages"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0)
df["estimated_downloads"] = df["estimated_downloads"].astype(int)

# Dil aralıkları oluştur
def dil_grubu(sayi):
    if sayi <= 3:
        return "1-3 dil"
    elif sayi <= 6:
        return "4-6 dil"
    elif sayi <= 10:
        return "7-10 dil"
    else:
        return "11+ dil"

df["dil_grubu"] = df["language_count"].apply(dil_grubu)

# Ortalama indirme sayısını grupla
dil_df = df.groupby("dil_grubu")["estimated_downloads"].mean().reindex(["1-3 dil", "4-6 dil", "7-10 dil", "11+ dil"])

# Grafik
plt.figure(figsize=(8, 5))
dil_df.plot(kind="bar", color="teal")
plt.title("Dil Sayısına Göre Ortalama İndirme Sayısı")
plt.ylabel("Ortalama Tahmini İndirme")
plt.xlabel("Desteklenen Dil Aralığı")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

import seaborn as sns

# Fiyat aralığı etiketleme
def fiyat_araligi(fiyat):
    if fiyat == 0:
        return "Ücretsiz"
    elif fiyat <= 5:
        return "0-5 USD"
    elif fiyat <= 15:
        return "5-15 USD"
    else:
        return "15+ USD"

df["fiyat_araligi"] = df["price"].apply(fiyat_araligi)

# Ortalama indirme ve rating hesapla
price_group = df.groupby("fiyat_araligi")[["estimated_downloads", "rating"]].mean().reindex(["Ücretsiz", "0-5 USD", "5-15 USD", "15+ USD"])

# Grafik: Ortalama İndirme Sayısı
plt.figure(figsize=(8, 5))
sns.barplot(x=price_group.index, y=price_group["estimated_downloads"], palette="Set2")
plt.title("Fiyat Aralığına Göre Ortalama İndirme Sayısı")
plt.ylabel("Ortalama Tahmini İndirme")
plt.xlabel("Fiyat Aralığı")
plt.tight_layout()
plt.show()

# Grafik: Ortalama Rating
plt.figure(figsize=(8, 5))
sns.barplot(x=price_group.index, y=price_group["rating"], palette="Set3")
plt.title("Fiyat Aralığına Göre Ortalama Kullanıcı Puanı")
plt.ylabel("Ortalama Rating")
plt.xlabel("Fiyat Aralığı")
plt.ylim(0, 10)
plt.tight_layout()
plt.show()
