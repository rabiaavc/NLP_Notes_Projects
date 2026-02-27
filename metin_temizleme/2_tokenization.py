"""

Tokenization yapacağız 2 yöntem var
1- Kelime Tokenizasyonu
2- Cümle Tokenizasyonu
"""

# pip install nltk
import nltk #natural language tool kit
nltk.download('punkt') #kelime ve cümle tokenizasyonu içib gerekli veri

#örnek text tanımla
raw_text = "Merhaba Google! Bu bir NLP eğitimidir. NLP eğitiminde ilerleyen aşamalarda LLM konusunda öğrenelim."

#word tokenization
word_tokens = nltk.word_tokenize(raw_text)
print(f"Kelime Tokenizasyonu: {word_tokens}")

#sentence tokenization
sentence_tokens = nltk.sent_tokenize(raw_text)
print(f"Cümle Tokenizasyonu: {sentence_tokens}")
