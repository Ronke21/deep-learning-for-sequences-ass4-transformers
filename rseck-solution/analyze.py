from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from attention import CausalSelfAttention, kqv, attention_scores
from transformer import TransformerLM


def heat_map_kqv(x: Tensor, kqv_matrix: Tensor):
    k, q, v = kqv(x, kqv_matrix)
    return attention_scores(k, q).detach().cpu()


def heat_map_attention(x: Tensor, attention_layer: CausalSelfAttention):
    for kqv_matrix in attention_layer.kqv_matrices:
        heat_map = heat_map_kqv(x, kqv_matrix)
        plt.imshow(heat_map[0].detach().cpu().numpy(), cmap="hot", interpolation="nearest")
        plt.show()


def create_heatmap_plot_for_model(model: TransformerLM, x: Tensor):
    num_layers = len(model.layers)
    num_heads = len(model.layers[0].causal_attention.kqv_matrices)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(20, 20))

    for i, layer in enumerate(model.layers):
        for j, head in enumerate(layer.causal_attention.kqv_matrices):
            heat_map = heat_map_kqv(x, head)

            ax = axes[i, j]
            cax = ax.matshow(heat_map.numpy()[0], cmap="viridis")
            fig.colorbar(cax, ax=ax)

            ax.set_title(f"Layer {i + 1} Head {j + 1}")
            ax.axis("off")  # Hide axes for a cleaner look

    plt.tight_layout()
    plt.show()


def main(model_path: str, tokenizer_path: str, text: str):
    embed_size = 192
    max_content_len = 128
    tokenizer = torch.load(tokenizer_path)
    tokens = torch.tensor([tokenizer.tokenize(text)])
    model = TransformerLM(6, 6, embed_size, max_content_len, 66, 4 * embed_size, True)
    model.load_state_dict(torch.load(model_path))
    embeddings = model.embed(tokens)
    create_heatmap_plot_for_model(model, embeddings)


if __name__ == "__main__":
    path = (
        Path()
        / "results for yedidia"
        / "loss_0.25_llm_model_25000_s_128_b_64_dp_data_nl_6_nh_6_es_192_mhs_768_lr_0.0005_gc_1.0_wd_false_nbtt_25000_us_False.pth"
    )
    main(path, "tokenizer.pth", "This is my attempt")
