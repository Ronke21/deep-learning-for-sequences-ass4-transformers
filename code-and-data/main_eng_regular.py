# CUDA_VISIBLE_DEVICES=1 python main_eng_regular.py 2>&1 | tee training_eng_regular.log


from __future__ import annotations

import json
import os.path

import torch
from torch import optim

import data
import lm
from transformer import TransformerLM
import matplotlib.pyplot as plt

EPOCHS = 50000


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

    return (f"seq-len_{seq_len}_batch-size_{batch_size}_data-path_{data_path}_"
            f"n-layers_{n_layers}_n-heads_{n_heads}_embed-size_{embed_size}_mlp-hidden-size{mlp_hidden_size}"
            f"_learning-rate_{learning_rate}_gradient-clipping_{gradient_clipping}_weight-decay_{weight_decay}"
            f"_num-batches-to-train_{num_batches_to_train}_use-scheduler_{use_scheduler}".replace(
        "/", ""
    ))


def main():
    seq_len = 128
    batch_size = 64
    data_path = "data/"
    results_path = "results_eng_regular"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    weight_decay = 0.01

    num_batches_to_train = EPOCHS

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
    print(f"Using device: {device}")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    tokenizer, tokenized_data = data.load_data(data_path)
    tokenizer_file_path = os.path.join(results_path, f"{run_file_name}_tokenizer.pth")
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
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=num_batches_to_train // 5, gamma=0.8
    )

    model.train()

    num_batches = 0
    losses = []
    samples = []
    print(
        f"Training with seq_len={seq_len}, batch_size={batch_size}, n_layers={n_layers}, "
        f"n_heads={n_heads}, embed_size={embed_size}, mlp_hidden_size={mlp_hidden_size}, "
        f"learning_rate={learning_rate}, gradient_clipping={gradient_clipping}, "
        f"weight_decay={weight_decay}, num_batches_to_train={num_batches_to_train}, "
        f"use_scheduler={use_scheduler}")
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
