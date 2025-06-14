import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# 1. Veriyi yükle
df = pd.read_csv("bestSelling_games.csv")

# 2. Eksik verileri temizle
df = df.dropna(subset=["user_defined_tags", "estimated_downloads", "price"])
df["estimated_downloads"] = df["estimated_downloads"].astype(int)

# 3. Özellik oluşturma
df["free_to_play"] = df["price"].apply(lambda x: 1 if x == 0 else 0)
df["language_count"] = df["supported_languages"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0)
df["tags_list"] = df["user_defined_tags"].apply(lambda x: [tag.strip() for tag in str(x).split(",")])

# 4. Başarılı oyunları etiketle (indirme sayısına göre üst %20)
threshold = df["estimated_downloads"].quantile(0.80)
df["successful"] = df["estimated_downloads"].apply(lambda x: 1 if x >= threshold else 0)

# 5. Özellik matrisini oluştur
mlb = MultiLabelBinarizer()
tags_encoded = pd.DataFrame(mlb.fit_transform(df["tags_list"]), columns=mlb.classes_)
features = pd.concat([df[["price", "rating", "difficulty", "length", "language_count", "free_to_play"]], tags_encoded], axis=1)

# 6. Eğitim/test ayrımı (%80 / %20)
X = features
y = df["successful"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Decision Tree modeli
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Başarı ölçütleri
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

# 9. Konfüzyon matrisi
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Konfüzyon Matrisi")
plt.show()

# 10. Karar ağacını görselleştir
plt.figure(figsize=(20, 8))
plot_tree(model, feature_names=X.columns, class_names=["Not Successful", "Successful"], filled=True, fontsize=7)
plt.title("Karar Ağacı: Başarılı Oyun Tahmini")
plt.show()

