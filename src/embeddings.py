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


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.power(10000.0, np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)

    return pe
