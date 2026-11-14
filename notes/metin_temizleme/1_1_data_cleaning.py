import re
import string
from textblob import TextBlob
from bs4 import BeautifulSoup


class TextCleaner:

    def __init__(self, text):
        self.text = text

    def remove_extra_spaces(self):
        self.text = " ".join(self.text.split())
        return self

    def to_lowercase(self):
        self.text = self.text.lower()
        return self

    def remove_punctuation(self):
        self.text = self.text.translate(
            str.maketrans("", "", string.punctuation)
        )
        return self

    def remove_special_characters(self):
        self.text = re.sub(r"[^a-zA-Z0-9\s]", "", self.text)
        return self

    def correct_spelling(self):
        self.text = str(TextBlob(self.text).correct())
        return self

    def remove_html(self):
        self.text = BeautifulSoup(self.text, "html.parser").get_text()
        return self

    def get_text(self):
        return self.text


# 🔥 Kullanım
raw_text = "<div> PYTHON,   GOogle NLP @ 2026 amazig </div>"

cleaned_text = (
    TextCleaner(raw_text)
        .remove_html()
        .remove_extra_spaces()
        .to_lowercase()
        .remove_punctuation()
        .remove_special_characters()
        .correct_spelling()
        .get_text()
)

print(cleaned_text)

#Zincirleme Mantığı
#Bir metodun döndürdüğü şeyin, bir sonraki metodun üzerinde çalıştırılmasıdır.