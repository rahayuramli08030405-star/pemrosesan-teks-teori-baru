# === Import Library ===
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === 1. Baca File Hasil Cleaning ===
df = pd.read_csv("netflix_reviews_cleaned.csv")

# Pastikan kolom yang digunakan ada
texts = df['final_review'].dropna().astype(str)

# === 2. Buat Bigram ===
vectorizer_bi = CountVectorizer(ngram_range=(2, 2))
bi_matrix = vectorizer_bi.fit_transform(texts)
bi_sum = bi_matrix.sum(axis=0)
bi_freq = [(word, bi_sum[0, idx]) for word, idx in vectorizer_bi.vocabulary_.items()]
bi_freq = sorted(bi_freq, key=lambda x: x[1], reverse=True)

print("🔹 Top 20 Bigram:")
for w, c in bi_freq[:20]:
    print(f"{w} : {c}")

# === 3. Buat Trigram ===
vectorizer_tri = CountVectorizer(ngram_range=(3, 3))
tri_matrix = vectorizer_tri.fit_transform(texts)
tri_sum = tri_matrix.sum(axis=0)
tri_freq = [(word, tri_sum[0, idx]) for word, idx in vectorizer_tri.vocabulary_.items()]
tri_freq = sorted(tri_freq, key=lambda x: x[1], reverse=True)

print("\n🔹 Top 20 Trigram:")
for w, c in tri_freq[:20]:
    print(f"{w} : {c}")

# === 4. Buat WordCloud ===
all_text = ' '.join(texts)

# Ganti dengan font TrueType yang kamu punya, misalnya Arial
# Jika pakai Windows, bisa arahkan ke:
# font_path = "C:\\Windows\\Fonts\\arial.ttf"
font_path = "C:\\Windows\\Fonts\\arial.ttf"

# === 4. Buat WordCloud ===
all_text = ' '.join(texts)
font_path = "C:\\Windows\\Fonts\\arial.ttf"

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='plasma',
    font_path=font_path
).generate(all_text)

# === 5. Tampilkan WordCloud ===
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud Review Netflix", fontsize=16)

# 🟢 Tambahkan baris ini agar gambar tersimpan otomatis:
plt.savefig("wordcloud_netflix.png", dpi=300, bbox_inches='tight')

plt.show()
