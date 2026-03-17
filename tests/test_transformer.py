import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import softmax, layer_norm, residual_add_norm
from src.attention import scaled_dot_product_attention, init_attention_weights, project_qkv
from src.ffn import init_ffn_weights, feed_forward
from src.masks import create_causal_mask
from src.embeddings import (
    create_vocabulary, tokenize, create_embedding_table,
    get_embeddings, positional_encoding,
)
from src.encoder import init_encoder_block, encoder_block, init_encoder_stack, encoder
from src.decoder import (
    decoder_masked_self_attention, init_decoder_block, decoder_block,
    init_decoder_stack, decoder, output_projection,
)
from src.transformer import init_transformer, transformer_forward
from src.inference import autoregressive_generate


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


def test_softmax():
    print("\n-- softmax --")
    x = np.random.randn(2, 3, 4)
    s = softmax(x)
    check("shape preservada", s.shape == x.shape)
    check("soma=1 por linha", np.allclose(s.sum(axis=-1), 1.0, atol=1e-6))
    check("valores >= 0", np.all(s >= 0))


def test_layer_norm():
    print("\n-- layer_norm --")
    x = np.random.randn(2, 5, 64)
    ln = layer_norm(x)
    check("shape preservada", ln.shape == x.shape)
    check("media ~0", np.allclose(ln.mean(axis=-1), 0.0, atol=1e-5))
    check("var ~1", np.allclose(ln.var(axis=-1), 1.0, atol=1e-3))


def test_residual_add_norm():
    print("\n-- residual_add_norm --")
    x = np.random.randn(1, 4, 64)
    sub = np.random.randn(1, 4, 64)
    out = residual_add_norm(x, sub)
    check("shape preservada", out.shape == x.shape)
    check("media ~0", np.allclose(out.mean(axis=-1), 0.0, atol=1e-5))


def test_attention():
    print("\n-- scaled_dot_product_attention --")
    Q = np.random.randn(1, 5, 64)
    K = np.random.randn(1, 5, 64)
    V = np.random.randn(1, 5, 64)
    out = scaled_dot_product_attention(Q, K, V)
    check("shape (1,5,64)", out.shape == (1, 5, 64))

    mask = create_causal_mask(5)
    out_m = scaled_dot_product_attention(Q, K, V, mask)
    check("shape com mask (1,5,64)", out_m.shape == (1, 5, 64))


def test_attention_weights_and_projections():
    print("\n-- init_attention_weights / project_qkv --")
    Wq, Wk, Wv = init_attention_weights(64)
    check("Wq shape (64,64)", Wq.shape == (64, 64))

    X = np.random.randn(1, 5, 64)
    Q, K, V = project_qkv(X, Wq, Wk, Wv)
    check("Q shape (1,5,64)", Q.shape == (1, 5, 64))


def test_ffn():
    print("\n-- feed_forward --")
    W1, b1, W2, b2 = init_ffn_weights(64, 256)
    X = np.random.randn(1, 5, 64)
    out = feed_forward(X, W1, b1, W2, b2)
    check("shape preservada (1,5,64)", out.shape == (1, 5, 64))


def test_causal_mask():
    print("\n-- create_causal_mask --")
    mask = create_causal_mask(5)
    check("shape (5,5)", mask.shape == (5, 5))
    check("diagonal = 0", np.all(np.diag(mask) == 0))
    check("futuro = -inf", np.all(mask[0, 1:] == -np.inf))
    check("passado = 0", np.all(mask[4, :5] == 0))


def test_embeddings():
    print("\n-- embeddings --")
    words = ["<START>", "<EOS>", "thinking", "machines"]
    vocab_df, w2i = create_vocabulary(words)
    check("vocab size", len(w2i) == 4)

    ids = tokenize("thinking machines", w2i)
    check("tokenize", ids == [2, 3])

    table = create_embedding_table(4, 64)
    emb = get_embeddings(ids, table)
    check("embedding shape (1,2,64)", emb.shape == (1, 2, 64))


def test_positional_encoding():
    print("\n-- positional_encoding --")
    pe = positional_encoding(10, 64)
    check("shape (10,64)", pe.shape == (10, 64))
    check("valores entre -1 e 1", np.all(np.abs(pe) <= 1.0 + 1e-9))


def test_encoder_block():
    print("\n-- encoder_block --")
    weights = init_encoder_block(64, 256)
    X = np.random.randn(1, 5, 64)
    out = encoder_block(X, *weights)
    check("shape preservada (1,5,64)", out.shape == (1, 5, 64))


def test_encoder_stack():
    print("\n-- encoder stack --")
    layers = init_encoder_stack(3, 64, 256)
    X = np.random.randn(1, 5, 64)
    Z = encoder(X, layers)
    check("shape preservada (1,5,64)", Z.shape == (1, 5, 64))
    check("3 camadas", len(layers) == 3)


def test_decoder_block():
    print("\n-- decoder_block --")
    sa, ca, ffn = init_decoder_block(64, 256)
    Y = np.random.randn(1, 4, 64)
    Z = np.random.randn(1, 5, 64)
    out = decoder_block(Y, Z, sa, ca, ffn)
    check("shape preservada (1,4,64)", out.shape == (1, 4, 64))


def test_decoder_stack():
    print("\n-- decoder stack --")
    layers = init_decoder_stack(3, 64, 256)
    Y = np.random.randn(1, 4, 64)
    Z = np.random.randn(1, 5, 64)
    mask = create_causal_mask(4)
    out = decoder(Y, Z, layers, mask)
    check("shape preservada (1,4,64)", out.shape == (1, 4, 64))


def test_output_projection():
    print("\n-- output_projection --")
    W_out = np.random.randn(64, 100)
    dec_out = np.random.randn(1, 4, 64)
    probs = output_projection(dec_out, W_out)
    check("shape (1,4,100)", probs.shape == (1, 4, 100))
    check("soma=1", np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6))


def test_transformer_forward():
    print("\n-- transformer_forward --")
    np.random.seed(42)
    model = init_transformer(vocab_size=50, d_model=64, d_ff=256, n_layers=2)
    enc_ids = np.array([1, 2, 3])
    dec_ids = np.array([0, 4])
    probs = transformer_forward(enc_ids, dec_ids, model)
    check("shape (1,2,50)", probs.shape == (1, 2, 50))
    check("probabilidades somam 1", np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6))


def test_autoregressive():
    print("\n-- autoregressive_generate --")
    np.random.seed(42)
    words = ["<START>", "<EOS>", "thinking", "machines", "maquinas", "pensantes"]
    _, w2i = create_vocabulary(words)
    id2w = {v: k for k, v in w2i.items()}
    model = init_transformer(len(words), d_model=64, d_ff=256, n_layers=2)
    enc_ids = [w2i["thinking"], w2i["machines"]]
    result = autoregressive_generate(enc_ids, model, w2i, id2w, max_len=5)
    check("comeca com <START>", result[0] == "<START>")
    check("gera tokens", len(result) >= 2)


if __name__ == "__main__":
    np.random.seed(42)

    test_softmax()
    test_layer_norm()
    test_residual_add_norm()
    test_attention()
    test_attention_weights_and_projections()
    test_ffn()
    test_causal_mask()
    test_embeddings()
    test_positional_encoding()
    test_encoder_block()
    test_encoder_stack()
    test_decoder_block()
    test_decoder_stack()
    test_output_projection()
    test_transformer_forward()
    test_autoregressive()

    print(f"\n{'='*40}")
    print(f"Resultado: {passed} passed, {failed} failed")
    if failed == 0:
        print("Todos os testes passaram!")
    else:
        print("Alguns testes falharam.")
        sys.exit(1)
