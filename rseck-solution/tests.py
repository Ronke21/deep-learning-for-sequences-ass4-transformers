import torch
from torch import nn
import numpy as np

import attention

DEBUG = False


def test_attention_scores():
    test_and_answers_values = [(1, 2, 3, 4, [[[5, 11, 17], [11, 25, 39], [17, 39, 61], [23, 53, 83]]]),
                               (2, 3, 1, 2, [[[14], [32]], [[122], [167]]])]
    for batch_size, d, n, m, expected_array in test_and_answers_values:
        total_elements_a = batch_size * n * d
        sequential_tensor_a = torch.arange(1, total_elements_a + 1)
        total_elements_b = batch_size * m * d
        sequential_tensor_b = torch.arange(1, total_elements_b + 1)

        # fill in values for the a, b and expected_output tensor.
        a = sequential_tensor_a.view(batch_size, n, d)  # a three-dim tensor
        b = sequential_tensor_b.view(batch_size, m, d)  # a three-dim tensor
        #result for first batch, calculated by b@a_t manual
        expected_output = torch.tensor(expected_array) / np.sqrt(d)  # a three-dim tensor
        A = attention.attention_scores(a, b)
        # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
        assert torch.allclose(A, expected_output)
        dim_0, dim_1, dim_2 = A.shape
        assert dim_0 == batch_size
        assert dim_1 == m
        assert dim_2 == n


def test_kqv():
    n_heads = 2
    b, n, d = 3, 4, n_heads * 3
    x = torch.ones(b, n, d)
    kqv_matrices = nn.ModuleList([attention.create_kqv_matrix(d, n_heads) for i in range(n_heads)])
    for kqv_matrix in kqv_matrices:
        k, q, v = attention.kqv(x, kqv_matrix)
        assert k.shape == (b, n, int(d / n_heads))
        assert q.shape == (b, n, int(d / n_heads))
        assert v.shape == (b, n, int(d / n_heads))
        if DEBUG:
            expected_output = torch.ones(b, n, int(d / n_heads)) * d
            assert torch.equal(expected_output, k)
            assert torch.equal(expected_output, q)
            assert torch.equal(expected_output, v)


def test_self_attention():
    v = torch.tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]])
    A = torch.tensor([[[0.1, 0.2],
                       [0.3, 0.4]]])
    output = attention.self_attention(v, A)
    expected_output = torch.tensor([[[2.4283, 3.4283, 4.4283],
                                     [3.5130, 4.5130, 5.5130]]])
    torch.allclose(output, expected_output)

    mask = attention.create_causal_mask(3, 2, 6)
    masked_output = attention.self_attention(v, A, mask)


def test_multi_head_attention_layer():
    n_heads = 2
    b, n, d = 3, 4, n_heads * 3
    x = torch.ones(b, n, d)
    kqv_matrices = nn.ModuleList([attention.create_kqv_matrix(d, n_heads) for i in range(n_heads)])
    mask = attention.create_causal_mask(d, n_heads, 6)
    attention.multi_head_attention_layer(x, kqv_matrices, mask)


def test_causal_self_attention():
    n_heads = 2
    b, n, d = 3, 4, n_heads * 3
    max_context_len = 6
    causual_self_attention_model = attention.CausalSelfAttention(d, n_heads, max_context_len)
    x = torch.ones(b, n, d)
    attention_on_x = causual_self_attention_model(x)


if __name__ == '__main__':
    test_kqv()
    test_attention_scores()
    test_self_attention()
    test_multi_head_attention_layer()
    test_causal_self_attention()
