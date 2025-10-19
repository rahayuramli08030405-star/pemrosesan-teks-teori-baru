from google_play_scraper import Sort, reviews
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh stopwords (hanya pertama kali)
nltk.download('stopwords')

# === 1. SCRAPING DATA DARI GOOGLE PLAY ===
app_id = 'com.netflix.mediaclient'

result, _ = reviews(
    app_id,
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=1000
)

df = pd.DataFrame(result)

# Hapus kolom gambar profil jika ada
if 'userImage' in df.columns:
    df = df.drop(columns=['userImage'])

# === 2. PREPROCESSING DASAR ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # hapus URL
    text = re.sub(r'\d+', '', text)                      # hapus angka
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)             # hapus tanda baca
    text = re.sub(r'\b(build|versi|version|v)\b', '', text)  # hapus kata build/version
    text = re.sub(r'\s+', ' ', text).strip()             # hapus spasi berlebih
    return text

df['cleaned_review'] = df['content'].apply(clean_text)

# === 3. NORMALISASI TEKS ===
normalization_dict = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "gak": "tidak",
    "bgt": "banget", "bngt": "banget", "tp": "tapi", "tpi": "tapi",
    "trs": "terus", "yg": "yang", "klo": "kalau", "klu": "kalau",
    "kl": "kalau", "dlm": "dalam", "sm": "sama", "dr": "dari",
    "udh": "sudah", "aja": "saja", "aj": "saja", "jd": "jadi",
    "jg": "juga", "ni": "ini", "nih": "ini", "ny": "nya", "kmu": "kamu",
    "pls": "tolong", "tlg": "tolong", "mkn": "makan", "nontonin": "menonton",
    "aplikasinya": "aplikasi", "bikin": "membuat", "baguss": "bagus",
    "parahh": "parah", "bgtu": "begitu", "bangettt": "banget",
    "ok": "oke", "okee": "oke", "mantapp": "mantap", "mantappp": "mantap"
}

def normalize_text(text):
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # hapus huruf berulang
    words = text.split()
    normalized_words = [normalization_dict.get(w, w) for w in words]
    normalized_words = [w for w in normalized_words if len(w) >= 3]
    return ' '.join(normalized_words)

df['normalized_review'] = df['cleaned_review'].apply(normalize_text)

# === 4. FINAL CLEANING: STOPWORDS + STEMMING ===
def final_process(text):
    words = [w for w in text.split() if w not in stop_words]
    text = ' '.join(words)
    text = stemmer.stem(text)
    return text

df['final_review'] = df['normalized_review'].apply(final_process)

# === 5. REMOVE DUPLIKAT ===
# Hapus duplikat berdasarkan nama pengguna dan isi review
df = df.drop_duplicates(subset=['userName', 'content'], keep='first')

# === 6. SIMPAN HASIL AKHIR ===
df_final = df[['userName', 'score', 'content', 'final_review']]
df_final.to_csv('netflix_reviews_cleaned.csv', index=False, encoding='utf-8-sig')

print(f"✅ Berhasil membersihkan, menormalisasi, dan menghapus duplikat dari {len(df_final)} review Netflix!")
print("📁 File tersimpan: netflix_reviews_cleaned.csv")
