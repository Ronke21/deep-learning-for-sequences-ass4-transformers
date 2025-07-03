from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from attention import CausalSelfAttention, kqv, attention_scores
from transformer import TransformerLM

import torch.serialization
from data import CharTokenizer  # Make sure this matches the actual location and name

def heat_map_kqv(x: Tensor, kqv_matrix: Tensor):
    k, q, v = kqv(x, kqv_matrix)
    return attention_scores(k, q).detach().cpu()


def heat_map_attention(x: Tensor, attention_layer: CausalSelfAttention):
    for kqv_matrix in attention_layer.kqv_matrices:
        heat_map = heat_map_kqv(x, kqv_matrix)
        plt.imshow(heat_map[0].detach().cpu().numpy(), cmap="hot", interpolation="nearest")
        plt.show()


def create_heatmap_plot_for_model(model: TransformerLM, x: Tensor, dir_path: str):
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
    plt.savefig(dir_path + "attention_heatmaps.png")
    print(f"Attention heatmaps saved to {dir_path}attention_heatmaps.png")
    plt.close()


def main(text: str, dir_path: str, embed_size: int = 192, max_content_len: int = 128, n_layers: int = 6, n_heads: int = 6):

    # model_path - find file in dir that starts with llm_model_50000 and ends with .pth
    model_path = Path(dir_path).glob("llm_model_50000*.pth")
    model_path = next(model_path, None)
    if model_path is None:
        raise FileNotFoundError("Model file not found in the specified directory.")
    
    # tokenizer_path - find file in dir that ends with _tokenizer.pth
    tokenizer_path = Path(dir_path).glob("*_tokenizer.pth")
    tokenizer_path = next(tokenizer_path, None)
    if tokenizer_path is None:
        raise FileNotFoundError("Tokenizer file not found in the specified directory.")
    torch.serialization.add_safe_globals([CharTokenizer])
    tokenizer = torch.load(tokenizer_path, weights_only=False)    
    tokens = torch.tensor([tokenizer.tokenize(text)])
    model = TransformerLM(n_layers, n_heads, embed_size, max_content_len, 66, 4 * embed_size, True)
    model.load_state_dict(torch.load(model_path))
    embeddings = model.embed(tokens)
    create_heatmap_plot_for_model(model, embeddings, dir_path)


if __name__ == "__main__":
    dir_path = "results_eng_regular/"
    model_path = dir_path + "llm_model_50000_seq-len_128_batch-size_64_data-path_data_n-layers_6_n-heads_6_embed-size_192_mlp-hidden-size768_learning-rate_0.0005_gradient-clipping_1.0_weight-decay_0.01_num-batches-to-train_50000_use-scheduler_True.pth"
    tokenizer_path = dir_path + "seq-len_128_batch-size_64_data-path_data_n-layers_6_n-heads_6_embed-size_192_mlp-hidden-size768_learning-rate_0.0005_gradient-clipping_1.0_weight-decay_0.01_num-batches-to-train_50000_use-scheduler_True_tokenizer.pth"
    sentence = "This is my attempt to visualize the attention in a transformer model."
    embed_size = 192
    max_content_len = 128
    n_layers = 6
    n_heads = 6
    main(sentence, dir_path, embed_size, max_content_len, n_layers, n_heads)