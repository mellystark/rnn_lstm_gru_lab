# RNN, LSTM and GRU Sequence Modeling Lab

A small PyTorch laboratory project for understanding and comparing recurrent neural network architectures:

- Simple RNN
- LSTM
- GRU

The project was created as part of an LLM / deep learning course to study sequence modeling, long-term dependencies, vanishing gradients, and gated recurrent architectures through practical experiments.

## Objectives

This repository focuses on answering a few core questions:

- How does a basic RNN process sequential data?
- Why do RNNs struggle with long-term dependencies?
- How do LSTM and GRU architectures address this limitation?
- How do different recurrent models behave under the same training conditions?
- How sensitive are recurrent models to initialization and random seeds?

## Experiments

### 1. Time-Series Prediction

File:

```text
run_experiment.py
```

A synthetic periodic time series is generated using sinusoidal components and noise.

The models receive the previous 30 time steps and predict the next value.

```text
30 time steps
      |
      v
RNN / LSTM / GRU
      |
      v
next value
```

This is a **many-to-one regression** problem.

Metrics:

- MAE
- MSE
- R²
- Training time
- Number of trainable parameters

This experiment demonstrates an important point: a more complex recurrent architecture is not automatically better when the task only requires short-term dependencies.

### 2. Delayed Memory Experiment

File:

```text
long_memory_experiment.py
```

The model receives a sequence in which the class information appears only at the beginning of the sequence.

Example:

```text
+1, noise, noise, noise, ... , noise
```

or:

```text
-1, noise, noise, noise, ... , noise
```

After processing the complete sequence, the model must determine whether the initial value was `+1` or `-1`.

The purpose is to demonstrate:

- Long-term dependency learning
- Vanishing-gradient behavior
- Differences between RNN, LSTM and GRU memory mechanisms

### 3. Multi-Seed Long-Term Memory Experiment

File:

```text
long_memory_experiment_v2.py
```

This is the improved long-memory experiment.

Instead of always placing the important information at the first time step, the relevant signal appears at a random position within the first quarter of the sequence.

Each input contains two channels:

```text
[value, marker]
```

Example:

```text
[ 0.08, 0]
[-0.11, 0]
[ 1.00, 1]   <- important value
[ 0.03, 0]
...
```

The marker identifies which time step contains the value that must be remembered.

The sequence length is 120 steps.

To reduce dependence on a single initialization, the experiment is repeated with multiple random seeds.

The experiment reports:

- Test accuracy
- F1 score
- Mean accuracy across seeds
- Standard deviation
- Training time
- Best validation accuracy
- Epoch at which 95% validation accuracy is reached

This experiment provides a more reliable comparison than a single training run.

## Key Concepts

### Simple RNN

A recurrent neural network updates its hidden state at every time step:

```text
h(t) = f(x(t), h(t-1))
```

The hidden state acts as short-term memory.

However, during backpropagation through time, gradients may become extremely small over long sequences. This is known as the **vanishing gradient problem**.

### LSTM

Long Short-Term Memory networks introduce a cell state and gating mechanisms.

Main gates:

- Input gate
- Forget gate
- Output gate

These gates control what information should be written, retained, forgotten, and exposed.

### GRU

Gated Recurrent Units use a simpler gated architecture than LSTM.

Main gates:

- Update gate
- Reset gate

GRU typically has fewer parameters than LSTM while still being designed for long-term sequence modeling.

## Project Structure

```text
rnn-lstm-gru-lab/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- run_experiment.py
|-- long_memory_experiment.py
|-- long_memory_experiment_v2.py
|
|-- rnn_lstm_gru_lab.ipynb
|
|-- outputs/                  # generated, ignored by Git
|-- outputs_long_memory/      # generated, ignored by Git
`-- outputs_long_memory_v2/   # generated, ignored by Git
```

## Environment

Recommended:

```text
Python 3.11
Miniconda / Conda
PyTorch
NumPy
Pandas
Matplotlib
scikit-learn
JupyterLab
```

Create the environment:

```bash
conda create -n llm_ders python=3.11 -y
conda activate llm_ders
```

Install the general dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately according to your hardware. For NVIDIA GPUs, use a CUDA-enabled PyTorch build compatible with your system rather than a CPU-only build.

Verify the installation:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Running the Experiments

### Time-series experiment

```bash
python run_experiment.py
```

### First long-memory experiment

```bash
python long_memory_experiment.py
```

### Improved multi-seed experiment

```bash
python long_memory_experiment_v2.py
```

## Generated Outputs

The scripts generate CSV summaries and plots such as:

```text
summary.csv
summary_long_memory.csv
summary_mean_std.csv

loss_comparison.png
training_loss.png
validation_accuracy.png
accuracy_mean_std.png
```

Generated output folders are excluded from Git by default so that the repository remains clean.

Selected figures or final experiment results can later be added manually under a dedicated `results/` directory.

## Observations From the Experiments

The experiments illustrate several useful lessons:

1. A Simple RNN can perform very well when only short-term dependencies are required.
2. Long sequences make recurrent optimization much harder.
3. A model that remains near `0.693` binary cross-entropy loss is effectively behaving close to random guessing.
4. LSTM can learn dependencies that a Simple RNN may fail to preserve over long sequences.
5. Recurrent model training can be highly sensitive to initialization and random seeds.
6. A single training run is not sufficient for making strong model comparisons.
7. Accuracy alone may be misleading, so additional metrics such as macro-F1 and confusion matrices are useful.

## Why This Matters for LLMs

Modern LLMs are primarily based on Transformer architectures, but recurrent networks are important for understanding the historical development of sequence modeling.

```text
RNN
 |
 v
LSTM / GRU
 |
 v
Encoder-Decoder models
 |
 v
Attention
 |
 v
Transformer
 |
 v
Modern LLMs
```

Understanding RNN, LSTM and GRU makes the motivation behind attention and Transformers much easier to understand.

## Future Work

Planned extensions:

- Bidirectional RNN
- Bidirectional LSTM
- Bidirectional GRU
- Stacked recurrent networks
- Dropout experiments
- Sequence-length sensitivity analysis
- Gradient norm visualization
- Confusion matrices
- Macro-F1 evaluation
- Turkish sentiment classification
- Comparison with Transformer-based sequence models

## Notes

This repository is intended for educational experimentation.

Results may vary depending on:

- random seed
- initialization
- sequence length
- hidden size
- optimizer
- learning rate
- CPU/GPU backend
- PyTorch version
