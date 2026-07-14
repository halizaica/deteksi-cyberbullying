# ==========================================================
# IMPORT LIBRARY
# ==========================================================
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# LOAD DATASET
# ==========================================================
df = pd.read_csv("dataset_percobaan.csv")
print(df.columns.tolist())
print("="*60)
print("DATASET BERHASIL DIMUAT")
print("="*60)
print(df.head())

print("\nJumlah Data :", len(df))

print("\nDistribusi Label")
print(df["encoded_label"].value_counts())

# ==========================================================
# DATA
# ==========================================================
X = df["clean_text"].astype(str)
y = df["encoded_label"]

# ==========================================================
# SPLIT DATA
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah Training :", len(X_train))
print("Jumlah Testing  :", len(X_test))

# ==========================================================
# MEMBUAT FOLDER MODEL
# ==========================================================
os.makedirs("model", exist_ok=True)

# ==========================================================
# FUNGSI EVALUASI
# ==========================================================
def evaluasi_model(model, X_test, y_test):

    pred = model.predict(X_test)

    print("\nClassification Report")
    print(classification_report(y_test, pred))

    print("Accuracy  :", round(accuracy_score(y_test, pred),4))
    print("Precision :", round(precision_score(y_test, pred),4))
    print("Recall    :", round(recall_score(y_test, pred),4))
    print("F1 Score  :", round(f1_score(y_test, pred),4))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, pred))

# ==========================================================
# BAG OF WORDS
# ==========================================================
print("\n")
print("="*60)
print("TRAINING MODEL BAG OF WORDS + SVM")
print("="*60)

bow_vectorizer = CountVectorizer(

    ngram_range=(1,2),
    min_df=2

)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

model_bow = SVC(

    kernel="linear",
    class_weight="balanced",
    probability=True,
    random_state=42

)

model_bow.fit(

    X_train_bow,
    y_train

)

evaluasi_model(

    model_bow,
    X_test_bow,
    y_test

)

joblib.dump(

    model_bow,
    "model/svm_bow.pkl"

)

joblib.dump(

    bow_vectorizer,
    "model/bow_vectorizer.pkl"

)

print("\nModel BoW berhasil disimpan.")

# ==========================================================
# TF-IDF
# ==========================================================
print("\n")
print("="*60)
print("TRAINING MODEL TF-IDF + SVM")
print("="*60)

tfidf_vectorizer = TfidfVectorizer(

    ngram_range=(1,2),
    min_df=2

)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_tfidf = SVC(

    kernel="linear",
    class_weight="balanced",
    probability=True,
    random_state=42

)

model_tfidf.fit(

    X_train_tfidf,
    y_train

)

evaluasi_model(

    model_tfidf,
    X_test_tfidf,
    y_test

)

joblib.dump(

    model_tfidf,
    "model/svm_tfidf.pkl"

)

joblib.dump(

    tfidf_vectorizer,
    "model/tfidf_vectorizer.pkl"

)

print("\nModel TF-IDF berhasil disimpan.")

# ==========================================================
# SELESAI
# ==========================================================
print("\n")
print("="*60)
print("SEMUA MODEL BERHASIL DILATIH")
print("="*60)

print("File yang dihasilkan :")
print("✔ model/svm_bow.pkl")
print("✔ model/bow_vectorizer.pkl")
print("✔ model/svm_tfidf.pkl")
print("✔ model/tfidf_vectorizer.pkl")