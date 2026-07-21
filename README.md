# Turkish Movie Review Sentiment Analysis

A comparative study of transformer architectures for **binary sentiment classification on Turkish movie reviews**.

This project fine-tunes and evaluates three different language-model architectures:

- **BERTurk** — encoder-only transformer
- **Turkish GPT-2** — decoder-only transformer
- **mT5-small** — encoder-decoder transformer

The complete workflow, including exploratory data analysis, preprocessing, hyperparameter experiments, training, evaluation, and discussion, is available in the project notebook.

---

## Project Overview

Sentiment analysis aims to determine whether a text expresses a positive or negative opinion. In this project, Turkish movie reviews are classified into two sentiment categories:

- `0` — Negative
- `1` — Positive

The main objective is to compare how different transformer architectures perform under the same dataset and evaluation setting.

The project includes:

- Dataset loading and cleaning
- Exploratory data analysis
- Stratified train, validation, and test splitting
- Fine-tuning of three transformer architectures
- Three hyperparameter configurations for each model
- Validation-based model selection
- Final evaluation on a held-out test set
- Performance and architecture comparison

---

## Models

### BERTurk

[`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) is used as the encoder-only model.

BERTurk is pre-trained specifically on Turkish text and processes tokens bidirectionally. Its architecture is well suited to sequence-classification tasks where the complete context of a review is important.

### Turkish GPT-2

[`ytu-ce-cosmos/turkish-gpt2`](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2) is used as the decoder-only model.

The pre-trained autoregressive model is adapted for binary sequence classification by adding a classification head. Because GPT-2 tokenizers generally do not define a padding token, the end-of-sequence token is also used for padding.

### mT5-small

[`google/mt5-small`](https://huggingface.co/google/mt5-small) is used as the encoder-decoder model.

For mT5, sentiment classification is formulated as a text-generation task:

```text
Input:  duygu analizi: <movie review>
Output: pozitif | negatif
```

Generated labels are normalized and converted into binary predictions during evaluation.

---

## Dataset

The project uses the **Turkish Movie Sentiment Dataset**, containing positive and negative Turkish movie reviews.

The original dataset is associated with:

> Erkin Demirtas and Mykola Pechenizkiy,  
> *Cross-Lingual Polarity Detection with Machine Translation*,  
> Proceedings of the Second International Workshop on Issues of Sentiment Discovery and Opinion Mining, 2013.

Dataset source:

- [Turkish Movie Sentiment Dataset](https://mpechen.win.tue.nl/projects/smm/Turkish_Movie_Sentiment.zip)

The repository contains the sentiment files used by the notebook:

```text
data/
├── tr_polarity.neg
└── tr_polarity.pos
```

After duplicate removal, the dataset contains **10,638 reviews**. The data is divided using stratified sampling:

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The test set is kept separate during training and hyperparameter selection.

---

## Preprocessing

The preprocessing pipeline includes:

- Whitespace normalization
- Missing-value inspection
- Duplicate detection and removal
- Binary label encoding
- Stratified dataset splitting
- Model-specific tokenization
- Sequence truncation and dynamic padding

Turkish characters such as `ş`, `ğ`, `ü`, `ö`, `ç`, and `ı` are preserved.

Aggressive preprocessing methods such as stemming, stopword removal, and punctuation deletion are avoided because transformer models can benefit from the original sentence structure and contextual information.

---

## Experimental Setup

Each architecture is trained using three different hyperparameter configurations.

The experiments vary:

- Learning rate
- Batch size
- Effective batch size
- Number of epochs
- Maximum sequence length
- Weight decay
- Gradient accumulation

The strongest configuration for each architecture is selected according to its **validation macro F1-score**. The selected model is then evaluated once on the held-out test set.

A fixed random seed of `42` is used to improve reproducibility.

---

## Results

The final test-set results are shown below.

| Model | Architecture | Test Macro F1 | Test Accuracy |
|---|---|---:|---:|
| **BERTurk** | Encoder-only | **0.9235** | **0.9236** |
| Turkish GPT-2 | Decoder-only | 0.9129 | 0.9129 |
| mT5-small | Encoder-decoder | 0.8665 | 0.8665 |

### Best Configurations

| Model | Learning Rate | Batch Size | Epochs | Sequence Length | Weight Decay |
|---|---:|---:|---:|---:|---:|
| BERTurk | `2e-5` | `16` | `3` | `128` | `0.01` |
| Turkish GPT-2 | `1e-5` | `16` effective | `3` | `128` | `0.01` |
| mT5-small | `2e-4` | `8` | `4` | `384` source length | `0.0` |

BERTurk achieves the strongest overall performance. Its Turkish-specific pre-training and bidirectional encoder architecture make it particularly effective for this classification task.

Turkish GPT-2 also produces competitive results despite being pre-trained with an autoregressive language-modeling objective.

mT5-small performs below the classification-oriented models. Representing binary classification as text generation introduces additional complexity, including occasional outputs that do not exactly match the expected sentiment labels.

---

## Repository Structure

```text
turkish-movie-review-sentiment-analysis/
├── data/
│   ├── tr_polarity.neg
│   └── tr_polarity.pos
├── notebooks/
│   └── turkish_sentiment_analysis.ipynb
├── results/                     # Generated during notebook execution
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook may initially be placed in the repository root. The paths in the notebook should be updated if it is moved into the `notebooks/` directory.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/erenyildizSU/turkish-movie-review-sentiment-analysis.git
cd turkish-movie-review-sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

- Python
- PyTorch
- Transformers
- Datasets
- Accelerate
- SentencePiece
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Jupyter

---

## Usage

Start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open:

```text
notebooks/turkish_sentiment_analysis.ipynb
```

Run the notebook cells in order.

For Google Colab, upload the notebook and dataset files, then update the dataset paths if necessary. Enabling a GPU runtime is strongly recommended:

```text
Runtime → Change runtime type → T4 GPU
```

Training time and memory usage depend on the selected model, sequence length, batch size, and available hardware.

---

## Generated Outputs

During execution, the notebook may generate:

- Validation results for each hyperparameter experiment
- Saved model checkpoints
- Classification reports
- Confusion matrices
- Training and evaluation plots
- Final model-comparison tables

Large checkpoints and temporary training folders should not be committed to GitHub. They should be excluded through `.gitignore`.

---

## Key Findings

- Turkish-specific pre-training provides a clear advantage for this dataset.
- Encoder-only architectures are highly effective for binary sentiment classification.
- Decoder-only models can be successfully adapted with a classification head.
- Text generation is less efficient than direct class prediction for a two-label task.
- Macro F1-score provides a balanced evaluation of both sentiment classes.
- Validation-based model selection prevents the test set from influencing hyperparameter decisions.

---

## Limitations

- The dataset is limited to the movie-review domain.
- Only three hyperparameter configurations are evaluated per model.
- Computational constraints limit batch sizes and model sizes.
- The results may not directly generalize to social media, product reviews, or news comments.
- mT5 predictions require output normalization because generated text may not always exactly match the target labels.

---

## Future Work

Possible extensions include:

- Evaluating larger Turkish transformer models
- Performing broader hyperparameter optimization
- Applying cross-validation
- Testing domain-adaptive pre-training
- Using data-augmentation techniques
- Evaluating the models on additional Turkish sentiment datasets
- Building an ensemble of the strongest models
- Deploying the selected model through a web interface or REST API

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

---

## Author

**Hüseyin Eren Yıldız**

Computer Science and Engineering  
Sabancı University

---

## Acknowledgements

This project uses publicly available pre-trained models from the Hugging Face ecosystem and the Turkish Movie Sentiment Dataset introduced by Demirtas and Pechenizkiy.

Please cite the original dataset publication when using or extending this work.
