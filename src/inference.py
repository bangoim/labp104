import numpy as np

from src.transformer import transformer_forward


def autoregressive_generate(encoder_input_ids, model, word_to_id, id_to_word, max_len=20):
    start_id = word_to_id["<START>"]
    eos_id = word_to_id["<EOS>"]

    decoder_ids = [start_id]

    for _ in range(max_len):
        probs = transformer_forward(
            np.array(encoder_input_ids),
            np.array(decoder_ids),
            model,
        )

        next_token_probs = probs[0, -1, :]
        next_id = int(np.argmax(next_token_probs))
        decoder_ids.append(next_id)

        if next_id == eos_id:
            break

    return [id_to_word[tid] for tid in decoder_ids]
