import torch
import attention
import numpy as np

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

