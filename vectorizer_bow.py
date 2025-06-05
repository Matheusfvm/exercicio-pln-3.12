from sklearn.feature_extraction.text import CountVectorizer

def vectorizer(documents):
    vectorizer = CountVectorizer(
        min_df=0.05,
        max_df=0.9
    )
    bow = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    return bow, vocab
    
