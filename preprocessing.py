import nltk
import spacy
import re
from nltk.corpus import stopwords
from spacy.cli import download as spacy_download

nltk.download('stopwords')
spacy_download('pt_core_news_sm')

spacy_nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
stopwords_nltk = set(stopwords.words('portuguese'))

stopwords_list = {
    'após', 'várias', 'principalmente', 'vez', 'ficar',
    'ser', 'ter', 'receber', 'nosso', 'meu', 'minha'
}

stopwords_pt = stopwords_nltk.union(stopwords_list)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zãâáàêéíôóõúç0-9\s]', ' ', text)
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_pt and len(t) > 2]
    cleaned = ' '.join(tokens)
    doc = spacy_nlp(cleaned)
    
    lemmas = []
    for tok in doc:
        if tok.is_alpha and tok.text not in stopwords_pt and not tok.is_stop and len(tok.lemma_) > 2:
            lemmas.append(tok.lemma_)
    
    return " ".join(lemmas)