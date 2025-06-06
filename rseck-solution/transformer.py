from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp
from utils import get_module_device


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            n_heads: int,
            embed_size: int,
            mlp_hidden_size: int,
            max_context_len,
            with_residuals: bool = False,
    ):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(
            embed_size, n_heads, max_context_len
        )
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals

    def forward(self, inputs):
        if self.with_residuals:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.layer_norm_2(x) + inputs
            x = self.mlp(x) + x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
        return x


class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(torch.arange(x.shape[-1], device=x.device))
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        # tok_embeddings =
        # pos_embeddings = ...
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
    ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params / 1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        for name, module in self.named_modules():
            # print(f"Module name: {name}")
            init_module(module)
            # print(module)
            if isinstance(module, nn.ModuleList):
                for sub_name, sub_module in module.named_modules():
                    # print(f"sub_Module name: {sub_name}")
                    init_module(sub_module)
                    # print(sub_module)
                    if isinstance(sub_module, nn.ModuleList):
                        for sub_sub_name, sub_sub_module in sub_module.named_modules():
                            # print(f"sub_sub_Module name: {sub_sub_name}")
                            init_module(sub_sub_module)
                            # print(sub_sub_module)

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=get_module_device(self)))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    # Temperature should be the temperature in which you sample.
    # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
    # for the given position.
    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float,
                                   topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=get_module_device(self)))
                logits_for_last_token = logits[0][-1]
                top_values, top_indices_to_tokens = torch.topk(logits_for_last_token, k=topK)
                top_values_in_temperature = top_values / temperature
                top_k_distribution_for_last_token = F.softmax(top_values_in_temperature, dim=0)
                sampled_token = top_indices_to_tokens[torch.multinomial(
                    top_k_distribution_for_last_token, num_samples=1).item()].item()
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated


def init_module(p):
    # if isinstance(p, torch.nn.LayerNorm) or isinstance(p, nn.Linear) or isinstance(p, nn.Embedding):
    # print("something actually got custom initialized")
    if isinstance(p, nn.LayerNorm):
        torch.nn.init.zeros_(p.bias)
        torch.nn.init.ones_(p.weight)
    elif isinstance(p, nn.Linear):
        torch.nn.init.xavier_normal_(p.weight)
        if p.bias is not None:
            torch.nn.init.normal_(p.bias)
    elif isinstance(p, nn.Embedding):
        torch.nn.init.xavier_normal_(p.weight)
