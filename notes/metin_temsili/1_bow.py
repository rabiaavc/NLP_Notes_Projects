"""
Amaç:
    - metin temsili (Bag of Words): metin listesi --> sayısal vektörler
    - sklearn CountVectorizer: kelimelerin kaç defa geçtiğini sayar ve vektör temsillerine dönüştürür.

Sonuç:
    - kelime kümesi (vocabulary)
    - her metin listesi sayısal vektörler ile temsil edilecek

    pip install scikit-learn
"""

#Scikit-learn içindeki CountVectorizer sınıfını projene dahil eder.
from sklearn.feature_extraction.text import CountVectorizer

# Örnek metinlerden oluşan veri seti
dokumanlar = [
    "kedi bahçede",
    "kedi evde",
]

# CountVectorizer nesnesi oluştur
kelime_sayac = CountVectorizer()

# Dokümanları sayısal vektörlere çevir (Bag of Words)
dokuman_vektorleri = kelime_sayac.fit_transform(dokumanlar)
print(f"dokuman_vektörleri",dokuman_vektorleri)

# Kelime kümesini al
kelime_kumesi = kelime_sayac.get_feature_names_out()
print(f"Kelime kümesi: {kelime_kumesi}")

# Sayısal vektörleri yazdır
print("BoW Matrisi:")
print(dokuman_vektorleri.toarray())