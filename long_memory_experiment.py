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
SEED = 42

SEQ_LEN = 100
N_SAMPLES = 6000
HIDDEN_SIZE = 32
NUM_LAYERS = 1

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

NOISE_STD = 0.15

OUTPUT_DIR = Path("outputs_long_memory")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TEKRARLANABİLİRLİK
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# VERİ
# ============================================================
def create_delayed_memory_dataset(
    n_samples=N_SAMPLES,
    seq_len=SEQ_LEN,
    noise_std=NOISE_STD,
    seed=SEED,
):
    """
    Görev:
    İlk zaman adımında -1 veya +1 verilir.

    x[0] = -1  -> sınıf 0
    x[0] = +1  -> sınıf 1

    Sonraki 99 zaman adımı dikkat dağıtıcı küçük gürültüdür.

    Model, dizinin EN SONUNDA ilk adımda ne olduğunu hatırlamalıdır.

    Bu nedenle problem:
        Many-to-One
        + uzun dönem bağımlılığı
    """

    rng = np.random.default_rng(seed)

    labels = rng.integers(0, 2, size=n_samples).astype(np.float32)

    x = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(n_samples, seq_len, 1),
    ).astype(np.float32)

    # Asıl bilgi SADECE ilk zaman adımında.
    x[:, 0, 0] = np.where(labels == 1, 1.0, -1.0)

    y = labels.reshape(-1, 1)

    return x, y


# ============================================================
# MODELLER
# ============================================================
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        model_type,
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ):
        super().__init__()

        self.model_type = model_type.upper()
        self.hidden_size = hidden_size

        common = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        if self.model_type == "RNN":
            self.recurrent = nn.RNN(
                **common,
                nonlinearity="tanh",
            )

        elif self.model_type == "LSTM":
            self.recurrent = nn.LSTM(**common)

            # Eğitimde uzun hafızayı destekleyen yaygın bir başlangıç:
            # forget gate bias'ını pozitif başlatıyoruz.
            #
            # PyTorch LSTM gate sırası:
            # input, forget, cell, output
            self._set_lstm_forget_bias(1.0)

        elif self.model_type == "GRU":
            self.recurrent = nn.GRU(**common)

        else:
            raise ValueError(f"Desteklenmeyen model: {model_type}")

        self.classifier = nn.Linear(hidden_size, 1)

    def _set_lstm_forget_bias(self, value):
        """
        Forget gate'in başlangıçta bilgiyi hemen silmemesini sağlar.
        Bu, LSTM'lerde kullanılan bilinen bir başlangıç yaklaşımıdır.
        """
        h = self.hidden_size

        for name, parameter in self.recurrent.named_parameters():
            if "bias" in name:
                with torch.no_grad():
                    parameter[h:2*h].fill_(value)

    def forward(self, x):
        recurrent_output, _ = self.recurrent(x)

        # Many-to-One:
        # 100 zaman adımı okunuyor, yalnızca son temsil sınıflandırılıyor.
        last_hidden = recurrent_output[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_probabilities = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)

        logits = model(x_batch)
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu().numpy().reshape(-1)
        )

        all_targets.append(
            y_batch.numpy().reshape(-1)
        )

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)

    predictions = (probabilities >= 0.5).astype(np.int64)

    accuracy = accuracy_score(
        targets.astype(np.int64),
        predictions,
    )

    f1 = f1_score(
        targets.astype(np.int64),
        predictions,
    )

    return accuracy, f1


def train_model(
    model,
    train_loader,
    val_loader,
    device,
):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            # Exploding gradient'a karşı koruma.
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = x_batch.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            n_seen += batch_size

        train_loss = running_loss / n_seen

        val_acc, _ = evaluate(
            model,
            val_loader,
            device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_acc"].append(
            val_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"  Epoch {epoch:02d}/{EPOCHS}"
                f" | loss={train_loss:.5f}"
                f" | val_acc={val_acc:.4f}"
            )

    elapsed = (
        time.perf_counter() - start
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, elapsed, best_val_acc


# ============================================================
# GRADYANIN ZAMAN İÇİNDE YAŞAMASI
# ============================================================
def plot_conceptual_gradient_survival():
    """
    Bu bölüm eğitim sonucu değildir.

    Amaç, uzun zincirde sürekli çarpmanın etkisini
    matematiksel olarak görselleştirmektir.

    RNN örneği:
        yaklaşık 0.8^d

    LSTM-benzeri hafıza yolu:
        forget gate yaklaşık 0.99 ise 0.99^d

    GRU-benzeri güncelleme yolu:
        yaklaşık 0.98^d

    Bu grafik "LSTM her zaman bu değeri üretir" demek değildir.
    Yalnızca vanishing gradient mekanizmasını sezgisel gösterir.
    """

    distance = np.arange(0, SEQ_LEN)

    rnn_survival = 0.80 ** distance
    lstm_survival = 0.99 ** distance
    gru_survival = 0.98 ** distance

    plt.figure(figsize=(10, 5))

    plt.semilogy(
        distance,
        rnn_survival,
        label="Simple RNN benzetimi (0.80^d)",
    )

    plt.semilogy(
        distance,
        lstm_survival,
        label="LSTM cell-state benzetimi (0.99^d)",
    )

    plt.semilogy(
        distance,
        gru_survival,
        label="GRU benzetimi (0.98^d)",
    )

    plt.xlabel(
        "Bilginin geçmesi gereken zaman adımı sayısı"
    )

    plt.ylabel(
        "Göreli gradient büyüklüğü (log ölçek)"
    )

    plt.title(
        "Uzun Bağımlılıkta Gradient'in Yaşaması - Kavramsal Gösterim"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gradient_survival_concept.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# ANA DENEY
# ============================================================
def main():
    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(
        "UZUN HAFIZA DENEYİ: RNN vs LSTM vs GRU"
    )
    print("=" * 72)

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
        "Hatırlanacak bilgi : SADECE t=0 zaman adımında"
    )
    print(
        f"Dikkat dağıtıcı    : {SEQ_LEN - 1} zaman adımı"
    )
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    x, y = create_delayed_memory_dataset()

    indices = np.arange(
        len(x)
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(indices)

    x = x[indices]
    y = y[indices]

    n_train = int(
        len(x) * TRAIN_RATIO
    )

    n_val = int(
        len(x) * VAL_RATIO
    )

    train_end = n_train
    val_end = n_train + n_val

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_val = x[
        train_end:val_end
    ]

    y_val = y[
        train_end:val_end
    ]

    x_test = x[val_end:]
    y_test = y[val_end:]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.float32,
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x_val,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_val,
                dtype=torch.float32,
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x_test,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_test,
                dtype=torch.float32,
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"Train / Val / Test: "
        f"{len(x_train)} / "
        f"{len(x_val)} / "
        f"{len(x_test)}"
    )

    print()

    model_types = [
        "RNN",
        "LSTM",
        "GRU",
    ]

    histories = {}
    results = []

    # --------------------------------------------------------
    # Eğitim
    # --------------------------------------------------------
    for model_type in model_types:
        print("-" * 72)
        print(
            f"{model_type} eğitiliyor..."
        )
        print("-" * 72)

        # Modeller aynı seed'den başlasın.
        set_seed()

        model = SequenceClassifier(
            model_type
        ).to(device)

        params = count_parameters(
            model
        )

        print(
            f"Parametre sayısı: {params:,}"
        )

        history, elapsed, best_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            device,
        )

        histories[model_type] = history

        test_acc, test_f1 = evaluate(
            model,
            test_loader,
            device,
        )

        results.append(
            {
                "Model": model_type,
                "Parameters": params,
                "TrainSeconds": elapsed,
                "BestValAccuracy": best_val_acc,
                "TestAccuracy": test_acc,
                "TestF1": test_f1,
            }
        )

        print(
            f"{model_type}"
            f" | Test Acc={test_acc:.4f}"
            f" | F1={test_f1:.4f}"
            f" | Süre={elapsed:.2f} sn"
        )

        print()

    # --------------------------------------------------------
    # Loss grafiği
    # --------------------------------------------------------
    plt.figure(
        figsize=(10, 5)
    )

    for model_type, history in histories.items():
        plt.plot(
            range(
                1,
                len(
                    history[
                        "train_loss"
                    ]
                ) + 1,
            ),
            history[
                "train_loss"
            ],
            label=model_type,
        )

    plt.xlabel("Epoch")
    plt.ylabel(
        "BCE Training Loss"
    )
    plt.title(
        "Uzun Hafıza Deneyi - Training Loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # Validation accuracy grafiği
    # --------------------------------------------------------
    plt.figure(
        figsize=(10, 5)
    )

    for model_type, history in histories.items():
        plt.plot(
            range(
                1,
                len(
                    history[
                        "val_acc"
                    ]
                ) + 1,
            ),
            history[
                "val_acc"
            ],
            label=model_type,
        )

    plt.axhline(
        y=0.5,
        linestyle="--",
        label="Şans seviyesi (0.50)",
    )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Validation Accuracy"
    )
    plt.title(
        "100 Adım Önceki Bilgiyi Hatırlama"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_accuracy.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # Kavramsal gradient grafiği
    # --------------------------------------------------------
    plot_conceptual_gradient_survival()

    # --------------------------------------------------------
    # Sonuç tablosu
    # --------------------------------------------------------
    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_DIR / "summary_long_memory.csv",
        index=False,
    )

    print("=" * 72)
    print("SONUÇ TABLOSU")
    print("=" * 72)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        "NOT:"
    )

    print(
        "- %50 civarı doğruluk = model bilgiyi koruyamamış,"
        " yaklaşık rastgele tahmin ediyor."
    )

    print(
        "- Daha yüksek doğruluk = ilk zaman adımındaki bilgiyi"
        " daha başarılı taşımış."
    )

    print(
        "- Tek koşu kesin bilimsel hüküm değildir."
        " Seed, hiperparametre ve optimizasyon sonucu etkiler."
    )

    print()
    print(
        f"Çıktılar: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
