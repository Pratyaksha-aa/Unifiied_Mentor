import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("Iris.csv")

print(df.head())
print(df.info())
print(df.describe())

print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Boxplot before
sns.boxplot(x='SepalWidthCm', data=df)
plt.title("Before Outlier Removal")
plt.show()

# IQR Method
Q1 = df['SepalWidthCm'].quantile(0.25)
Q3 = df['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['SepalWidthCm'] >= lower) & (df['SepalWidthCm'] <= upper)]

# Boxplot after
sns.boxplot(x='SepalWidthCm', data=df)
plt.title("After Outlier Removal")
plt.show()

# Count Plot
sns.countplot(x='Species', data=df)
plt.title("Class Distribution")
plt.show()

# Pairplot
sns.pairplot(df, hue='Species')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Feature Importance - Random Forest")
plt.show()
