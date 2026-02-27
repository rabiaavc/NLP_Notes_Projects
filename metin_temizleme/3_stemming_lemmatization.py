"""
Kök ve Gövde Bulma

kök -- stemming
gövde -- lemma

Stemming:Kelimenin sonundaki ekleri mekanik olarak keser.Yani dilbilgisi düşünmez, kurallara bakmaz.
stemming --> porter stemmer

Lemmatization:Kelimeyi sözlükteki gerçek temel formuna indirir.Dilbilgisel analiz yapar.
lemmatization --> porter lemmatizer

"""
import nltk
nltk.download("wordnet") #lemma bulmak için gerekli wordnet veri tabanı
nltk.download("omw-1.4")

#stemming
from nltk.stem import PorterStemmer  #ingilizce için popüler stemmer algoritması
stemmer = PorterStemmer()  #porter stemmer nesnesi oluştu
words_stem = ["playing","played","plays","happily","happier","studying","studies"]

stems = [stemmer.stem(w) for w in words_stem]
print(f"Original: {words_stem}")
print(f"Stemming: {stems}")

#lemmatization

from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words_lemma = ["running","ran","gone","better","children"]

lemmas = [lemmatizer.lemmatize(w) for w in words_lemma]
print(f"Original: {words_lemma}")
print(f"Lemmas: {lemmas}")



