from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
SEQ_LEN = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.80

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_series(n_points=3500):
    """
    Biraz karmaşık ama öğrenilebilir sentetik zaman serisi.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(n_points, dtype=np.float32)

    series = (
        np.sin(0.035 * t)
        + 0.35 * np.sin(0.09 * t + 0.8)
        + 0.15 * np.sin(0.007 * t)
        + rng.normal(0.0, 0.05, size=n_points)
    )

    return series.astype(np.float32)


def make_sequences(series, seq_len):
    xs = []
    ys = []

    for i in range(len(series) - seq_len):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len])

    x = np.asarray(xs, dtype=np.float32)[..., None]
    y = np.asarray(ys, dtype=np.float32)[..., None]

    return x, y


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        model_type,
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ):
        super().__init__()

        self.model_type = model_type.upper()

        common_args = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        if self.model_type == "RNN":
            self.recurrent = nn.RNN(**common_args)
        elif self.model_type == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif self.model_type == "GRU":
            self.recurrent = nn.GRU(**common_args)
        else:
            raise ValueError(f"Desteklenmeyen model: {model_type}")

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        recurrent_output, _ = self.recurrent(x)

        # Many-to-One:
        # Tüm zaman adımlarını okuyoruz, son adımın temsilini kullanıyoruz.
        last_output = recurrent_output[:, -1, :]

        prediction = self.output_layer(last_output)
        return prediction


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()

            # Exploding gradient riskine karşı tipik bir önlem:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        history.append(epoch_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.6f}")

    elapsed = time.perf_counter() - start

    return history, elapsed


@torch.no_grad()
def predict(model, x_test, device):
    model.eval()

    tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    predictions = model(tensor)

    return predictions.cpu().numpy().reshape(-1)


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("RNN - LSTM - GRU ZAMAN SERİSİ KARŞILAŞTIRMASI")
    print("=" * 65)
    print("PyTorch:", torch.__version__)
    print("Device :", device)

    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # 1. Veri üret
    # ---------------------------------------------------------
    raw_series = create_series()

    split_raw = int(len(raw_series) * TRAIN_RATIO)

    # Normalizasyon istatistikleri SADECE train bölümünden hesaplanır.
    train_mean = raw_series[:split_raw].mean()
    train_std = raw_series[:split_raw].std()

    normalized_series = (raw_series - train_mean) / train_std

    x_all, y_all = make_sequences(normalized_series, SEQ_LEN)

    # Sequence oluşturulduğu için split indeksini yeniden ayarla.
    split_seq = split_raw - SEQ_LEN

    x_train = x_all[:split_seq]
    y_train = y_all[:split_seq]

    x_test = x_all[split_seq:]
    y_test = y_all[split_seq:]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print()
    print(f"Train örneği : {len(x_train)}")
    print(f"Test örneği  : {len(x_test)}")
    print(f"Sequence len : {SEQ_LEN}")
    print()

    model_types = ["RNN", "LSTM", "GRU"]

    histories = {}
    results = []

    for model_type in model_types:
        print("-" * 65)
        print(f"{model_type} eğitiliyor")
        print("-" * 65)

        # Her model karşılaştırmaya aynı seed ile başlasın.
        set_seed()

        model = RecurrentRegressor(model_type).to(device)

        n_params = count_parameters(model)
        print(f"Parametre sayısı: {n_params:,}")

        history, elapsed = train_model(model, train_loader, device)
        histories[model_type] = history

        pred_norm = predict(model, x_test, device)

        # Normalize edilmiş değerleri gerçek ölçeğe geri çevir.
        y_true = y_test.reshape(-1) * train_std + train_mean
        y_pred = pred_norm * train_std + train_mean

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append(
            {
                "Model": model_type,
                "Parameters": n_params,
                "TrainSeconds": elapsed,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
            }
        )

        print(
            f"{model_type} | "
            f"MAE={mae:.5f} | "
            f"MSE={mse:.5f} | "
            f"R²={r2:.5f} | "
            f"Süre={elapsed:.2f} sn"
        )

        # İlk 250 test noktası için tahmin grafiği
        plt.figure(figsize=(12, 5))
        plt.plot(y_true[:250], label="Gerçek")
        plt.plot(y_pred[:250], label="Tahmin")
        plt.title(f"{model_type} - Gerçek ve Tahmin")
        plt.xlabel("Test zaman adımı")
        plt.ylabel("Değer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"predictions_{model_type}.png", dpi=150)
        plt.close()

    # ---------------------------------------------------------
    # Loss karşılaştırması
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))

    for model_type, losses in histories.items():
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=model_type,
        )

    plt.title("Eğitim Loss Karşılaştırması")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_comparison.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # Sonuç tablosu
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    df = df.sort_values("MAE").reset_index(drop=True)

    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print()
    print("=" * 65)
    print("SONUÇLAR")
    print("=" * 65)
    print(df.to_string(index=False))
    print()
    print("Dosyalar outputs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
