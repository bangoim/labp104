from src.attention import scaled_dot_product_attention, init_attention_weights, project_qkv
from src.utils import residual_add_norm


def decoder_masked_self_attention(Y, Wq, Wk, Wv, mask, eps=1e-6):
    Q, K, V = project_qkv(Y, Wq, Wk, Wv)
    attn_out = scaled_dot_product_attention(Q, K, V, mask)
    return residual_add_norm(Y, attn_out, eps)
