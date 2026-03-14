import numpy as np
import pandas as pd


def create_vocabulary(words):
    vocab_df = pd.DataFrame({
        "word": words,
        "id": range(len(words))
    })
    word_to_id = dict(zip(vocab_df["word"], vocab_df["id"]))
    return vocab_df, word_to_id


def tokenize(sentence, word_to_id):
    tokens = sentence.lower().split()
    return [word_to_id[token] for token in tokens]


def create_embedding_table(vocab_size, d_model):
    return np.random.randn(vocab_size, d_model)


def get_embeddings(token_ids, embedding_table):
    embeddings = embedding_table[token_ids]
    return np.expand_dims(embeddings, axis=0)
