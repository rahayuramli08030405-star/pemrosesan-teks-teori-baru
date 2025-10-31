# === Import Library ===
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Unduh stopwords (jika belum)
nltk.download('stopwords')

# === 1. Baca File Hasil Cleaning ===
df = pd.read_csv("netflix_reviews_cleaned.csv")

# Pastikan kolom yang digunakan ada
texts = df['final_review'].dropna().astype(str)

# === 2. Definisikan daftar kata tidak penting & normalisasi ===
stop_words = set(stopwords.words('indonesian'))

# Kata gaul, singkatan, typo, dan kata objek
custom_stopwords = {
    'netflix','aplikasi', 'app', 'apk', 'apkny', 'nya', 'nyaa',
    'dong', 'nih', 'lho', 'eh', 'aja', 'ya', 'deh', 'lah', 'kok', 'weh', 'woi',
    'gue', 'gua', 'gw', 'loe', 'lo', 'org', 'yg', 'bro', 'sis', 'anj', 'anjir',
    'anjay', 'haha', 'wkwk', 'wkwkwk', 'lol', 'tolol', 'tai', 'bodoh'
}

# Kamus normalisasi kata tidak baku → baku
normalisasi_kata = {
    'gk': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'tdk': 'tidak',
    'bgt': 'banget', 'bngt': 'banget', 'trus': 'terus', 'dr': 'dari', 'tp': 'tapi',
    'blm': 'belum', 'udh': 'sudah', 'udah': 'sudah', 'pdhl': 'padahal', 'jg': 'juga',
    'sm': 'sama', 'klo': 'kalau', 'kl': 'kalau', 'dgn': 'dengan', 'msh': 'masih',
    'bsk': 'besok', 'krn': 'karena', 'trs': 'terus', 'pls': 'tolong',
    'makasih': 'terima kasih', 'makasi': 'terima kasih', 'thx': 'terima kasih',
    'tq': 'terima kasih', 'ok': 'oke', 'okeey': 'oke', 'okey': 'oke'
}

# === 3. Fungsi Pembersihan Lengkap ===
def clean_text(text):
    text = text.lower()                      # ubah huruf kecil semua
    text = re.sub(r'\d+', '', text)          # hapus angka
    text = re.sub(r'[^\w\s]', ' ', text)     # hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip() # hapus spasi berlebih

    tokens = []
    for w in text.split():
        if w in normalisasi_kata:            # ubah kata tidak baku ke baku
            w = normalisasi_kata[w]
        if w not in stop_words and w not in custom_stopwords:
            tokens.append(w)
    return ' '.join(tokens)

# Terapkan fungsi ke semua teks
df['cleaned_review'] = texts.apply(clean_text)

# 🔹 Hapus baris kosong setelah cleaning
df = df[df['cleaned_review'].notnull() & (df['cleaned_review'].str.strip() != '')]

# === 4. Simpan hasil pembersihan ke file baru ===
output_file = "netflix_reviews_cleaned_final.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ File hasil pembersihan telah disimpan sebagai: {output_file}")

# === 5. Buat Bigram ===
vectorizer_bi = CountVectorizer(ngram_range=(2, 2))
bi_matrix = vectorizer_bi.fit_transform(df['cleaned_review'])
bi_sum = bi_matrix.sum(axis=0)
bi_freq = [(word, bi_sum[0, idx]) for word, idx in vectorizer_bi.vocabulary_.items()]
bi_freq = sorted(bi_freq, key=lambda x: x[1], reverse=True)

print("\n🔹 Top 20 Bigram:")
for w, c in bi_freq[:20]:
    print(f"{w} : {c}")

# === 6. Buat Trigram ===
vectorizer_tri = CountVectorizer(ngram_range=(3, 3))
tri_matrix = vectorizer_tri.fit_transform(df['cleaned_review'])
tri_sum = tri_matrix.sum(axis=0)
tri_freq = [(word, tri_sum[0, idx]) for word, idx in vectorizer_tri.vocabulary_.items()]
tri_freq = sorted(tri_freq, key=lambda x: x[1], reverse=True)

print("\n🔹 Top 20 Trigram:")
for w, c in tri_freq[:20]:
    print(f"{w} : {c}")

# === 7. Buat WordCloud ===
all_text = ' '.join(df['cleaned_review'])
font_path = "C:\\Windows\\Fonts\\arial.ttf"

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='plasma',
    font_path=font_path
).generate(all_text)

# === 8. Tampilkan dan Simpan WordCloud ===
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud Review Netflix (Setelah Normalisasi & Pembersihan Lengkap)", fontsize=16)
plt.savefig("wordcloud_netflix_final.png", dpi=300, bbox_inches='tight')
plt.show()