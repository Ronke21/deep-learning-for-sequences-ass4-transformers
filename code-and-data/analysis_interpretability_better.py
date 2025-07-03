from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import re
import os

from attention import CausalSelfAttention, kqv, attention_scores
from transformer import TransformerLM
from data import CharTokenizer

import torch.serialization

SPECIAL_ANALYSIS_TOKENS = {
    "space": " ",
    "vowels": set("aeiouAEIOU"),
}

def heat_map_kqv(x: Tensor, kqv_matrix: Tensor):
    k, q, v = kqv(x, kqv_matrix)
    return attention_scores(k, q).detach().cpu()

def analyze_attention_pattern(attn_matrix: Tensor, token_ids: list[int], tokenizer: CharTokenizer):
    scores = {}

    # Previous-token attention
    prev_diag = torch.diagonal(attn_matrix[0, 1:, :-1])  # Skip first position
    scores["prev_token_attention"] = prev_diag.mean().item()

    # Space token attention
    space_id = tokenizer.char_to_idx.get(" ", None)
    if space_id is not None:
        space_pos = [i for i, tok in enumerate(token_ids) if tok == space_id]
        if space_pos:
            scores["space_attention"] = attn_matrix[0, :, space_pos].mean().item()

    # Vowel attention
    vowel_ids = {tokenizer.char_to_idx[c] for c in tokenizer.char_to_idx if c in SPECIAL_ANALYSIS_TOKENS["vowels"]}
    vowel_pos = [i for i, tok in enumerate(token_ids) if tok in vowel_ids]
    if vowel_pos:
        scores["vowel_attention"] = attn_matrix[0, :, vowel_pos].mean().item()

    return scores

# --- NEW ---
def save_heatmap(attn_matrix: Tensor, token_ids: list[int], tokenizer: CharTokenizer,
                 text: str, layer: int, head: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tokens = [tokenizer.decode([t]) for t in token_ids]
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attn_matrix[0].numpy(), cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_title(f"Layer {layer + 1} Head {head + 1}", fontsize=12)
    plt.tight_layout()

    safe_text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())[:50]
    filename = f"layer{layer + 1}_head{head + 1}_{safe_text}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    print(f"🖼️ Saved heatmap: {filename}")

def print_if_interesting(text: str, layer: int, head: int, score_dict: dict,
                         thresholds: dict, attn_matrix: Tensor, token_ids: list[int],
                         tokenizer: CharTokenizer, out_dir: str):
    findings = []
    for key, val in score_dict.items():
        if val > thresholds.get(key, 0.5):
            findings.append(f"{key}: {val:.2f}")
    if findings:
        print(f"[Interesting] Layer {layer+1}, Head {head+1} — {text}")
        for f in findings:
            print("   🔍", f)
        save_heatmap(attn_matrix, token_ids, tokenizer, text, layer, head, out_dir)  # --- NEW ---

def main(texts: list[str], dir_path: str, embed_size: int = 192, max_content_len: int = 128,
         n_layers: int = 6, n_heads: int = 6):

    model_path = next(Path(dir_path).glob("llm_model_50000*.pth"), None)
    tokenizer_path = next(Path(dir_path).glob("*_tokenizer.pth"), None)
    if model_path is None or tokenizer_path is None:
        raise FileNotFoundError("Model/tokenizer file not found.")
    torch.serialization.add_safe_globals([CharTokenizer])
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    model = TransformerLM(n_layers, n_heads, embed_size, max_content_len, 66, 4 * embed_size, True)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    thresholds = {"prev_token_attention": 0.9, "space_attention": 0.4, "vowel_attention": 0.4}
    out_dir = "attn_vis"  # --- NEW ---

    for text in texts:
        token_ids = tokenizer.tokenize(text)
        tokens_tensor = torch.tensor([token_ids])
        embeddings = model.embed(tokens_tensor)

        for layer_idx, layer in enumerate(model.layers):
            for head_idx, head in enumerate(layer.causal_attention.kqv_matrices):
                attn_matrix = heat_map_kqv(embeddings, head)
                scores = analyze_attention_pattern(attn_matrix, token_ids, tokenizer)
                print_if_interesting(
                    text, layer_idx, head_idx, scores, thresholds,
                    attn_matrix, token_ids, tokenizer, out_dir
                )

if __name__ == "__main__":
    dir_path = "results_eng_all_better/"
    texts = [
        "The cat sat on the mat.",
        "This is an example sentence.",
        "How many vowels are in this phrase?",
        "Spaces separate words.",
        "Testing patterns in transformers!"
    ]
    main(texts, dir_path, embed_size=384, max_content_len=256, n_layers=12, n_heads=12)
