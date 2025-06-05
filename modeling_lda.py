from sklearn.decomposition import LatentDirichletAllocation

def fit_lda_model(bow_matrix, n_topics):
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=1000,
        learning_method='batch',
        random_state=42,
        doc_topic_prior=0.5,
        topic_word_prior=0.01,
    )
    dt_matrix = lda_model.fit_transform(bow_matrix)
    topic_word_matrix = lda_model.components_
    return dt_matrix, topic_word_matrix