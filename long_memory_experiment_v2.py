from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# AYARLAR
# ============================================================
SEEDS = [11, 22, 33]

SEQ_LEN = 120
N_SAMPLES = 4500
HIDDEN_SIZE = 32
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

OUTPUT_DIR = Path("outputs_long_memory_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DAHA GERÇEKÇİ UZUN HAFIZA VERİSİ
# ============================================================
def create_dataset(n_samples, seq_len, seed):
    """
    Input shape: [N, T, 2]

    Kanal 0:
        Gürültü/sinyal değeri

    Kanal 1:
        Marker (hangi zaman adımındaki bilginin önemli olduğunu gösterir)

    Önemli bilgi her örnekte ilk çeyrekte RASTGELE bir konumda bulunur.

    Örnek:
        t=17 -> value = +1, marker = 1
        diğer adımlar -> noise, marker = 0

    Hedef:
        +1 ise sınıf 1
        -1 ise sınıf 0

    Dolayısıyla model:
        1) marker'ı görmeli,
        2) o andaki işareti okumalı,
        3) uzun süre boyunca bilgiyi taşımalı,
        4) dizinin sonunda sınıflandırmalı.
    """

    rng = np.random.default_rng(seed)

    labels = rng.integers(0, 2, size=n_samples)

    x = np.zeros(
        (n_samples, seq_len, 2),
        dtype=np.float32,
    )

    # Birinci kanal dikkat dağıtıcı gürültü.
    x[:, :, 0] = rng.normal(
        0.0,
        0.20,
        size=(n_samples, seq_len),
    )

    # Bilgi sabit t=0'da değil.
    max_signal_pos = max(2, seq_len // 4)

    signal_positions = rng.integers(
        0,
        max_signal_pos,
        size=n_samples,
    )

    for i, (label, pos) in enumerate(
        zip(labels, signal_positions)
    ):
        x[i, pos, 0] = 1.0 if label == 1 else -1.0
        x[i, pos, 1] = 1.0

    y = labels.astype(np.float32).reshape(-1, 1)

    return x, y


# ============================================================
# MODELLER
# ============================================================
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type,
        input_size=2,
        hidden_size=HIDDEN_SIZE,
    ):
        super().__init__()

        self.model_type = model_type.upper()
        self.hidden_size = hidden_size

        common = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        if self.model_type == "RNN":
            self.recurrent = nn.RNN(
                **common,
                nonlinearity="tanh",
            )

        elif self.model_type == "LSTM":
            self.recurrent = nn.LSTM(**common)
            self._set_lstm_total_forget_bias(1.0)

        elif self.model_type == "GRU":
            self.recurrent = nn.GRU(**common)
            self._set_gru_total_update_bias(1.0)

        else:
            raise ValueError(
                f"Desteklenmeyen model: {model_type}"
            )

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def _set_lstm_total_forget_bias(self, value):
        """
        PyTorch LSTM gate sırası:
            input, forget, cell, output

        PyTorch'ta bias_ih ve bias_hh ayrı ayrı vardır.
        Toplam forget bias yaklaşık +1 olsun diye:
            bias_ih forget = +1
            bias_hh forget = 0
        """
        h = self.hidden_size

        with torch.no_grad():
            self.recurrent.bias_ih_l0[h:2*h].fill_(value)
            self.recurrent.bias_hh_l0[h:2*h].zero_()

    def _set_gru_total_update_bias(self, value):
        """
        PyTorch GRU gate sırası:
            reset, update, new

        Pozitif update-gate bias başlangıçta geçmiş hidden state'i
        korumayı kolaylaştırabilir.

        Toplam update bias yaklaşık +1:
            bias_ih update = +1
            bias_hh update = 0
        """
        h = self.hidden_size

        with torch.no_grad():
            self.recurrent.bias_ih_l0[h:2*h].fill_(value)
            self.recurrent.bias_hh_l0[h:2*h].zero_()

    def forward(self, x):
        sequence_output, _ = self.recurrent(x)

        # Many-to-One:
        last_hidden = sequence_output[:, -1, :]

        return self.classifier(last_hidden)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    probs_all = []
    targets_all = []

    for xb, yb in loader:
        xb = xb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits)

        probs_all.append(
            probs.cpu().numpy().reshape(-1)
        )
        targets_all.append(
            yb.numpy().reshape(-1)
        )

    probs = np.concatenate(probs_all)
    targets = np.concatenate(targets_all).astype(int)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)

    return acc, f1


def train_model(
    model,
    train_loader,
    val_loader,
    device,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss()

    train_loss_history = []
    val_acc_history = []

    best_val_acc = -1.0
    best_state = None
    epoch_95 = None

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            # Exploding gradient'a karşı.
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            n = xb.size(0)
            total_loss += loss.item() * n
            total_n += n

        train_loss = total_loss / total_n
        val_acc, _ = evaluate(
            model,
            val_loader,
            device,
        )

        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)

        if epoch_95 is None and val_acc >= 0.95:
            epoch_95 = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"    Epoch {epoch:02d}/{EPOCHS}"
                f" | loss={train_loss:.5f}"
                f" | val_acc={val_acc:.4f}"
            )

    elapsed = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss": train_loss_history,
        "val_acc": val_acc_history,
        "best_val_acc": best_val_acc,
        "epoch_95": epoch_95,
        "seconds": elapsed,
    }


def make_loaders(seed):
    x, y = create_dataset(
        N_SAMPLES,
        SEQ_LEN,
        seed,
    )

    rng = np.random.default_rng(seed + 999)

    idx = np.arange(len(x))
    rng.shuffle(idx)

    x = x[idx]
    y = y[idx]

    n_train = int(
        len(x) * TRAIN_RATIO
    )

    n_val = int(
        len(x) * VAL_RATIO
    )

    train_end = n_train
    val_end = n_train + n_val

    splits = {
        "train": (
            x[:train_end],
            y[:train_end],
        ),
        "val": (
            x[train_end:val_end],
            y[train_end:val_end],
        ),
        "test": (
            x[val_end:],
            y[val_end:],
        ),
    }

    loaders = {}

    for split_name, (xs, ys) in splits.items():
        dataset = TensorDataset(
            torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32),
        )

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split_name == "train"),
        )

    return loaders


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "UZUN HAFIZA DENEYİ V2 - RNN vs LSTM vs GRU"
    )
    print("=" * 80)

    print("PyTorch :", torch.__version__)
    print("Device  :", device)

    if torch.cuda.is_available():
        print(
            "GPU     :",
            torch.cuda.get_device_name(0),
        )

    print()
    print(
        f"Sequence uzunluğu : {SEQ_LEN}"
    )
    print(
        "Önemli bilgi      : İlk çeyrekte rastgele bir konum"
    )
    print(
        "Marker kanalı     : Hangi zaman adımının önemli olduğunu gösterir"
    )
    print(
        f"Seed'ler          : {SEEDS}"
    )
    print()

    model_types = [
        "RNN",
        "LSTM",
        "GRU",
    ]

    rows = []

    # İlk seed'in eğrilerini ayrıca çizelim.
    first_seed_histories = {}

    for seed in SEEDS:
        print("#" * 80)
        print(f"SEED = {seed}")
        print("#" * 80)

        loaders = make_loaders(seed)

        for model_type in model_types:
            print()
            print(
                f"{model_type} eğitiliyor..."
            )

            set_seed(seed)

            model = SequenceClassifier(
                model_type
            ).to(device)

            params = count_parameters(model)

            print(
                f"  Parametre sayısı: {params:,}"
            )

            history = train_model(
                model,
                loaders["train"],
                loaders["val"],
                device,
            )

            test_acc, test_f1 = evaluate(
                model,
                loaders["test"],
                device,
            )

            print(
                f"  Test Acc={test_acc:.4f}"
                f" | F1={test_f1:.4f}"
                f" | Epoch95={history['epoch_95']}"
            )

            rows.append(
                {
                    "Seed": seed,
                    "Model": model_type,
                    "Parameters": params,
                    "TrainSeconds": history["seconds"],
                    "BestValAccuracy": history["best_val_acc"],
                    "Epoch95": (
                        history["epoch_95"]
                        if history["epoch_95"] is not None
                        else np.nan
                    ),
                    "TestAccuracy": test_acc,
                    "TestF1": test_f1,
                }
            )

            if seed == SEEDS[0]:
                first_seed_histories[
                    model_type
                ] = history

    detailed = pd.DataFrame(rows)

    detailed.to_csv(
        OUTPUT_DIR / "detailed_results.csv",
        index=False,
    )

    summary = (
        detailed
        .groupby("Model")
        .agg(
            TestAccuracyMean=("TestAccuracy", "mean"),
            TestAccuracyStd=("TestAccuracy", "std"),
            TestF1Mean=("TestF1", "mean"),
            TestF1Std=("TestF1", "std"),
            TrainSecondsMean=("TrainSeconds", "mean"),
            Epoch95Mean=("Epoch95", "mean"),
            Success95Count=("Epoch95", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "summary_mean_std.csv",
        index=False,
    )

    # --------------------------------------------------------
    # İlk seed için validation accuracy eğrisi
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))

    for model_type, history in first_seed_histories.items():
        plt.plot(
            range(1, EPOCHS + 1),
            history["val_acc"],
            label=model_type,
        )

    plt.axhline(
        0.50,
        linestyle="--",
        label="Şans seviyesi",
    )

    plt.axhline(
        0.95,
        linestyle=":",
        label="%95",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        f"Uzun Hafıza V2 - Seed {SEEDS[0]}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_accuracy_seed11.png",
        dpi=160,
    )
    plt.close()

    # --------------------------------------------------------
    # Ortalama test accuracy bar grafiği
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["Model"],
        summary["TestAccuracyMean"],
        yerr=summary["TestAccuracyStd"].fillna(0.0),
        capsize=5,
    )

    plt.ylim(0.0, 1.05)
    plt.ylabel("Test Accuracy")
    plt.title(
        "3 Seed Ortalama ± Standart Sapma"
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_mean_std.png",
        dpi=160,
    )
    plt.close()

    print()
    print("=" * 80)
    print("DETAYLI SONUÇLAR")
    print("=" * 80)

    print(
        detailed.to_string(index=False)
    )

    print()
    print("=" * 80)
    print("3 SEED ÖZETİ")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    print()
    print("YORUMLAMA:")
    print(
        "- TestAccuracyMean: tek koşu yerine ortalama başarı."
    )
    print(
        "- TestAccuracyStd: seed değişince ne kadar oynadığını gösterir."
    )
    print(
        "- Epoch95Mean: %95 validation doğruluğuna ne kadar hızlı ulaştı."
    )
    print(
        "- Success95Count: 3 koşudan kaçında %95'e ulaştı."
    )
    print(
        "- Bir model %50 civarında kalıyorsa uzun bilgiyi öğrenememiş sayılır."
    )

    print()
    print(
        f"Çıktılar: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
