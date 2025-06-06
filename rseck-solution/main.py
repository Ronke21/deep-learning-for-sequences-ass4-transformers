from __future__ import annotations

import json
import os.path

import torch
from torch import optim

import data
import lm
from transformer import TransformerLM
import matplotlib.pyplot as plt

DEBUG = False


def get_file_name(
    seq_len,
    batch_size,
    data_path,
    n_layers,
    n_heads,
    embed_size,
    mlp_hidden_size,
    learning_rate,
    gradient_clipping,
    weight_decay,
    num_batches_to_train,
    use_scheduler,
):
    s = seq_len
    b = batch_size
    dp = data_path
    nl = n_layers
    nh = n_heads
    es = embed_size
    mhs = mlp_hidden_size

    lr = learning_rate
    gc = gradient_clipping
    wd = weight_decay

    nbtt = num_batches_to_train
    us = use_scheduler

    return f"with_dropout_s_{s}_b_{b}_dp_{dp}_nl_{nl}_nh_{nh}_es_{es}_mhs_{mhs}_lr_{lr}_gc_{gc}_wd_{wd}_nbtt_{nbtt}_us_{us}".replace(
        "/", ""
    )


def main():
    seq_len = 128
    batch_size = 128
    data_path = "data/"
    results_path = "results"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.01

    num_batches_to_train = 10 if DEBUG else 50000

    use_scheduler = True

    run_file_name = get_file_name(
        seq_len,
        batch_size,
        data_path,
        n_layers,
        n_heads,
        embed_size,
        mlp_hidden_size,
        learning_rate,
        gradient_clipping,
        weight_decay,
        num_batches_to_train,
        use_scheduler,
    )
    loss_file_name = os.path.join(results_path, "losses" + run_file_name + ".json")
    sampling_file_name = os.path.join(results_path, "sampling" + run_file_name + ".json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, tokenized_data = data.load_data(data_path)
    tokenizer_file_path = f"{run_file_name}_tokenizer.pth"
    torch.save(tokenizer, tokenizer_file_path)

    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=num_batches_to_train // 5, gamma=0.8
    )

    model.train()

    num_batches = 0
    losses = []
    samples = []
    while num_batches < num_batches_to_train:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits = model(batch_x.to(device))

            loss = lm.compute_loss(logits, batch_y.to(device))

            # parameters update
            model.zero_grad()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        # temperature( < 1) makes the model return more confident results (reducing entropy),
                        # while increasing the temperature ( > 1) makes the model return less confident results
                        # (increasing entropy).
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(
                                tokenizer.tokenize("Hello"), 500, 0.7, 5
                            )
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                        samples.append(sampled)
                        if num_batches % 10000 == 0:
                            torch.save(
                                model.state_dict(),
                                os.path.join(
                                    results_path, f"llm_model_{num_batches}_{run_file_name}.pth"
                                ),
                            )
                    print("")
    torch.save(
        model.state_dict(),
        os.path.join(results_path, f"llm_model_{num_batches}_{run_file_name}.pth"),
    )
    plt.plot(losses)  # noqa
    plt.savefig(os.path.join(results_path, f"llm_losses_{run_file_name}.png"))
    with open(loss_file_name, "a") as output:
        json.dump(losses, output, indent=4)
    with open(sampling_file_name, mode="a", encoding="utf-8") as output:
        json.dump(samples, output, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
