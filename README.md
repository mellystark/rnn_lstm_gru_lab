# RNN - LSTM - GRU Mini Laboratuvarı

Bu proje, aynı zaman serisi problemi üzerinde üç recurrent mimariyi karşılaştırır:

- Simple RNN
- LSTM
- GRU

Amaç, mimarilerin kullanımını PyTorch ile pratik olarak görmek ve aynı veri / aynı eğitim koşullarında sonuçlarını kıyaslamaktır.

## 1) Windows + Miniconda ortamını oluştur

Anaconda Prompt veya Miniconda Prompt aç:

```bat
cd /d C:\Users\Melike\Desktop\myFolds\YL\DERSLER\LLM

conda create -n llm_ders python=3.11 -y
conda activate llm_ders

python -m pip install --upgrade pip
pip install torch numpy matplotlib pandas scikit-learn jupyterlab
```

## 2) PyTorch ve GPU kontrolü

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA: False` çıkarsa deney yine CPU üzerinde çalışır. GPU kurulumu daha sonra ayrıca düzeltilebilir.

## 3) Projeyi çalıştır

Bu klasörün içeriğini şu dizine kopyala:

```text
C:\Users\Melike\Desktop\myFolds\YL\DERSLER\LLM\rnn_lstm_gru_lab
```

Sonra:

```bat
cd /d C:\Users\Melike\Desktop\myFolds\YL\DERSLER\LLM\rnn_lstm_gru_lab
python run_experiment.py
```

Alternatif olarak Jupyter:

```bat
jupyter lab
```

ve `rnn_lstm_gru_lab.ipynb` dosyasını aç.

## 4) Deney ne yapıyor?

Sentetik bir zaman serisi oluşturulur:

```text
sinüs + ikinci bir periyodik bileşen + az miktarda gürültü
```

Model son 30 zaman adımını görür:

```text
x(t-29), ..., x(t-2), x(t-1), x(t)
```

ve bir sonraki değeri tahmin eder:

```text
x(t+1)
```

Aynı veri üzerinde sırayla:

```text
RNN
LSTM
GRU
```

eğitilir.

## 5) Çıktılar

`outputs` klasöründe:

- `summary.csv`
- `loss_comparison.png`
- `predictions_RNN.png`
- `predictions_LSTM.png`
- `predictions_GRU.png`

oluşur.

Terminalde ayrıca her model için:

- parametre sayısı
- eğitim süresi
- MAE
- MSE
- R²

gösterilir.

## 6) Derste dikkat edeceğin yer

Üç modelin Python sınıfında değişen temel satır şudur:

```python
nn.RNN(...)
nn.LSTM(...)
nn.GRU(...)
```

Geri kalan eğitim mantığı büyük ölçüde aynıdır. Böylece mimarilerin farkını kod seviyesinde çok daha rahat görebilirsin.

Sonraki laboratuvarlar için uygun genişletmeler:

1. BiRNN / BiLSTM / BiGRU
2. Çok katmanlı (stacked) RNN
3. Dropout
4. Türkçe duygu analizi: Many-to-One
5. Kelime dizisi tahmini: Many-to-Many
6. Uzun bağımlılık deneyi: RNN vs LSTM vs GRU
