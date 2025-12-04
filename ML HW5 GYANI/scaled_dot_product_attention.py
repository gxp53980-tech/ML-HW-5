import numpy as np

def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    # subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (num_queries, d_k)
        K: Key matrix of shape (num_keys, d_k)
        V: Value matrix of shape (num_keys, d_v)
        mask: Optional boolean mask of shape (num_queries, num_keys)
              True = keep, False = mask out

    Returns:
        attention_weights: (num_queries, num_keys)
        context: (num_queries, d_v)
    """
    d_k = K.shape[-1]

    # 1. Compute raw attention scores: QK^T
    scores = Q @ K.T  # shape: (num_queries, num_keys)

    # 2. Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # 3. Apply mask if given (set masked positions to very negative number)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # 4. Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # 5. Multiply by V to get context vectors
    context = attention_weights @ V  # shape: (num_queries, d_v)

    return attention_weights, context


if __name__ == "__main__":
    # Example values to test the function
    # You can change these to your own matrices

    # Suppose d_k = d_v = 4, num_queries = 2, num_keys = 3
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0]
    ])

    K = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ])

    V = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 2.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    ])

    # Optional: no mask here (mask=None)
    attn_weights, context = scaled_dot_product_attention(Q, K, V)

    print("Query (Q):")
    print(Q)
    print("\nKey (K):")
    print(K)
    print("\nValue (V):")
    print(V)

    print("\nAttention Weights:")
    print(attn_weights)

    print("\nContext Vectors:")
    print(context)
