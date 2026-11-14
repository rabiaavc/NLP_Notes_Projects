"""
Stopwords
Anlam taşımayan veya model için bilgi değeri düşük olan kelimelerdir.
Türkçe: ve, ile, ama, için, bu, şu, o, da, de, mi, mı, mu, mü, gibi, çok, en
İngilizce : the, is, am, are, was, were,
            a, an, and, or, but,
            in, on, at, to, of, for,
            this, that, it, be, as
"""

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords') #stopwords setini indir

stopwords_english = stopwords.words('english') #ingilizce stop words listesi

#örnek metin

eng_text = "This is just a simple example to show how stop words are removed."
eng_text = eng_text.split()
print(eng_text)
filtered_words_eng = [word for word in eng_text if word.lower() not in stopwords_english]
print(filtered_words_eng)

#türkçe stopwords analizi
stop_words_tr = set(stopwords.words('turkish'))

tr_text = "Merhaba bugün sizlerle birlikte NLP eğitimi gerçekleştiriyoruz. Bu eğitimi sizler için çok faydalı olacaktır."
tr_text_list = tr_text.split()
print(tr_text_list)

filtered_words_tr = [word for word in tr_text_list if word.lower() not in stop_words_tr]
print(filtered_words_tr)

# kütüphane kullanmadan stop words çıkarma işlemi

custom_tr_stopwords = ["bu", "ile", "de", "da", "mi", "ki"]

custom_text = "Bu bir denemedir, bunun için amacımız metinlerde ki bazı kelimeleri çıkartmak."

custom_text_list = custom_text.split()

filtered_custom_words_tr = [
    word for word in custom_text_list
    if word.lower() not in custom_tr_stopwords
]

print(f"Orijinal: {custom_text}")
print(f"Custom stop words çıkarılmış hali: {filtered_custom_words_tr}")