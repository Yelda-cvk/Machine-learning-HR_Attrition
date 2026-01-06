# Veri işleme
import pandas as pd
import numpy as np

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Modelleme
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CSV dosyasını oku
df = pd.read_csv("HR_Attrition.csv")

# İlk 5 satıra bakalım
df.head()

# Genel yapı
df.info()

# İstatistiksel özet
df.describe()

# Çalışan ayrılmış mı ayrılmamış mı
sns.countplot(x="Attrition", data=df)
plt.title("Çalışan Ayrılma Dağılımı")
plt.show()

# Model için anlamsız sütunlar
df.drop(["EmployeeNumber", "Over18", "StandardHours"], axis=1, inplace=True)

le = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# X: özellikler
X = df.drop("Attrition", axis=1)

# y: hedef
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Doğruluk
accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

