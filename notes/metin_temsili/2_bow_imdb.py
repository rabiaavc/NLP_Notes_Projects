"""
    Amaç:
    -- IMDB yorumları içeren veri seti ile çalış
            -- cs dosyasından veriyi oku
            -- text cleaning (küçük harf çevirme, rakam ve özel karakterleden kurtul,stopwordslerden kurtul)
            -- metinleri sayısal vektörlere dönüştür (Bag of words)
            --kelime frekanslarını hesapla ve en sık geçen 5 kelimeyi bul
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re #regular expression
from collections import Counter
#stop wordleri kaldırmak  için
import nltk
from nltk.corpus import stopwords

#read csv

"""
Amaç:
-- IMDB yorumları içeren veri seti ile çalış
-- CSV dosyasından veriyi oku
-- Text cleaning yap (küçük harf, rakam ve özel karakter temizleme, stopwords kaldırma)
-- Metinleri sayısal vektörlere dönüştür (Bag of Words)
-- Kelime frekanslarını hesapla ve en sık geçen 5 kelimeyi bul
"""

# Bag of Words için gerekli sınıf
from sklearn.feature_extraction.text import CountVectorizer

# Veri okuma ve veri işleme için
import pandas as pd

# Regex (düzenli ifadeler) için
import re

# Frekans hesaplamak için
from collections import Counter

# Stopwords kaldırmak için NLTK
import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------
# STOPWORDS İNDİRME (ilk çalıştırmada gerekli)
# ---------------------------------------------------
nltk.download('stopwords')

# Stopwords listesini set yapıyoruz (daha hızlı arama için)
stopwords_english = set(stopwords.words('english'))

# ---------------------------------------------------
# CSV DOSYASINI OKUMA
# ---------------------------------------------------
df = pd.read_csv("IMDB Dataset.csv")

# İlk 5 satırı kontrol edelim
print(df.head(5))

# Yorumları al
comments = df["review"]

# Etiketleri al (positive / negative)
labels = df["sentiment"]


# ---------------------------------------------------
# METİN TEMİZLEME FONKSİYONU
# ---------------------------------------------------
def data_clean(text):
    # 1️⃣ Küçük harfe çevir
    text = text.lower()

    # 2️⃣ Rakamları kaldır
    text = re.sub(r"\d+", " ", text)

    # 3️⃣ Özel karakterleri kaldır
    text = re.sub(r"[^\w\s]", " ", text)

    # 4️⃣ 2 harften kısa kelimeleri kaldır
    text = " ".join([word for word in text.split() if len(word) > 2])

    # 5️⃣ Stopwords kaldır
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_english]

    # Tekrar string haline getir
    return " ".join(filtered_words)


# ---------------------------------------------------
# TÜM YORUMLARA TEMİZLEME UYGULA
# ---------------------------------------------------
cleaned_comments = [data_clean(y) for y in comments]

# ---------------------------------------------------
# BAG OF WORDS MODELİ
# ---------------------------------------------------

# CountVectorizer nesnesi oluştur
bow_model = CountVectorizer()

# İlk 75 yorumu sayısal vektöre çevir
comment_vectors = bow_model.fit_transform(cleaned_comments[:75])

# Kelime kümesini (vocabulary) al
vocabulary = bow_model.get_feature_names_out()

# Sparse matrisi numpy array'e çevir
vector_representation = comment_vectors.toarray()

print("Vektör Temsili:")
print(vector_representation)

#vektör temsillerini dataframe e dönüşütürlerim
df_bow = pd.DataFrame(vector_representation, columns=vocabulary)
print(df_bow.head(5))

# ---------------------------------------------------
# KELİME FREKANSLARINI HESAPLA
# ---------------------------------------------------

# Her kelimenin toplam frekansını hesapla
word_counts = comment_vectors.sum(axis=0).A1

# Kelime ve frekanslarını bir sözlükte eşleştir
word_freq = dict(zip(vocabulary, word_counts))

# Frekansa göre büyükten küçüğe sırala
five = Counter(word_freq).most_common(5)

# En sık geçen 5 kelimeyi yazdır
print("En sık geçen 5 kelime:",five)







