# teori_fix_final.py
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ===============================
# 1. LOAD DATA UTAMA
# ===============================
df = pd.read_csv("labeling spreadsheet - netflix_reviews_cleaned_final.csv")
print(">>> Total data utama:", len(df))

# pilih kolom teks
text_cols = ["cleaned_review", "final_review", "content"]
text_col = None
for c in text_cols:
    if c in df.columns:
        text_col = c
        break

if text_col is None:
    raise ValueError("Tidak ada kolom teks ditemukan!")

# ===============================
# 2. LOAD DATA MANUAL (FILE YANG SAMA)
# ===============================
df_manual = pd.read_csv("labeling spreadsheet - netflix_reviews_cleaned_final.csv")

# handle kolom dengan spasi + lowercase
df_manual.columns = df_manual.columns.str.strip().str.lower()

# mapping otomatis
if "cleaned_review" in df_manual.columns:
    df_manual = df_manual.rename(columns={"cleaned_review": "teks"})

if "labeling" in df_manual.columns:
    df_manual = df_manual.rename(columns={"labeling": "label"})

# cek kolom wajib
required = {"teks", "label"}
if not required.issubset(set(df_manual.columns)):
    raise ValueError(f"File manual harus punya kolom: {required}")

# ===============================
# 3. NORMALISASI LABEL
# ===============================
def normalize(x):
    if pd.isna(x): return ""
    x = str(x).lower().strip()
    if x in ["positif", "pos", "positive"]: return "positif"
    if x in ["negatif", "neg", "negative"]: return "negatif"
    if x in ["netral", "neutral"]: return "netral"
    return ""

df_manual["label_norm"] = df_manual["label"].apply(normalize)

print("\n>>> Ringkasan label manual:")
print(df_manual["label_norm"].value_counts())

# cek lengkap
if not {"positif", "negatif", "netral"}.issubset(set(df_manual["label_norm"].unique())):
    raise ValueError("Label tidak lengkap! Harus ada positif, negatif, netral.")

# ===============================
# 4. SELEKSI 250 DATA MANUAL
# ===============================
pos = df_manual[df_manual["label_norm"]=="positif"].head(100)
neg = df_manual[df_manual["label_norm"]=="negatif"].head(100)
net = df_manual[df_manual["label_norm"]=="netral"].head(50)

train_df = pd.concat([pos, neg, net]).copy()
train_df = train_df.rename(columns={"teks": "text"})

# pastikan folder ada sebelum save
os.makedirs("Labeling Data", exist_ok=True)

train_df.to_csv("Labeling Data/data_manual_cleaned.csv", index=False)

print("\n>>> 250 data manual siap training.")

# ===============================
# 5. TRAINING MODEL
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df["text"])
y = train_df["label_norm"]

model = LogisticRegression(max_iter=300)
model.fit(X, y)

print(">>> Model berhasil dilatih.")

# ===============================
# 6. LABELI SISA DATA
# ===============================
remaining = df.copy()
remaining = remaining[~remaining[text_col].isin(train_df["text"])]

X_rem = vectorizer.transform(remaining[text_col])
remaining["predicted_label"] = model.predict(X_rem)

# pastikan folder ada sebelum save
os.makedirs("Labeling Data", exist_ok=True)

remaining.to_csv("Labeling Data/data_self_training.csv", index=False)

print("\n>>> Selesai. Hasil disimpan:")
print("1. Labeling Data/data_manual_cleaned.csv")
print("2. Labeling Data/data_self_training.csv")
