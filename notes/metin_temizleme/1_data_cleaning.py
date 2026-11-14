"""
Stemming:Kelimenin sonundaki ekleri mekanik olarak keser.Yani dilbilgisi düşünmez, kurallara bakmaz.
Lemmatization:Kelimeyi sözlükteki gerçek temel formuna indirir.Dilbilgisel analiz yapar.

"""

## Veri Temizleme Adımları:
"""
• Boşlukların Temizlenmesi: "Hello World" → "Hello World"

• Büyük-Küçük Harf Dönüşümleri: "HeLLo WorlD" → "hello world"

• Noktalama İşaretlerinin Kaldırılması: "Hello, World!" → "Hello World"

• Özel Karakterlerin Kaldırılması: "Bu yazı $100 dolara mal oldu!" → "Bu yazı 100 dolara mal oldu"

• Yazım Hatalarının Düzeltilmesi: "Hllo Wrld" → "Hello World"

• HTML ve URL Temizleme: "<div>Hello World!</div>" → "Hello World!"""

"""
Stemming: Kelimenin sonundaki ekleri mekanik olarak keser.
Lemmatization: Kelimeyi sözlükteki gerçek temel formuna indirir.
"""

## Veri Temizleme Adımları

# 1️⃣ Boşluk temizleme
raw_text = "Python,   Google      NLP    dersi."
print(raw_text.split())
normalized_text_1 = " ".join(raw_text.split())
print(normalized_text_1)


# 2️⃣ Küçük harfe çevirme
raw_text = "PYTHON, GOogle NLP"
normalized_text_2 = raw_text.lower()
print(normalized_text_2)


# 3️⃣ Noktalama işaretlerini kaldırma
import string
raw_text = "Hello, World!"
normalized_text_3 = raw_text.translate(str.maketrans("", "", string.punctuation))
print(normalized_text_3)


# 4️⃣ Özel karakter temizleme
import re
raw_text = "Natural @ Language $ Processing"
normalized_text_4 = re.sub(r"[^A-Za-z0-9\s]", '', raw_text)
print(normalized_text_4)


# 5️⃣ Yazım hatalarını düzeltme
from textblob import TextBlob
raw_text = "ıt is amazig 2026"
normalized_text_5 = TextBlob(raw_text).correct()
print(normalized_text_5)


# 6️⃣ HTML temizleme
from bs4 import BeautifulSoup
raw_html = "<div> 2045 Google </div>"
normalized_text_6 = BeautifulSoup(raw_html, "html.parser").get_text()
print(normalized_text_6)