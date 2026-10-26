# Deep Learning Practical 6: Comprehensive Transformer Architectures & Implementation

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-v5-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

A rigorous, production-grade practical exploration of the **Transformer** architecture (*Vaswani et al., 2017*). This project covers theoretical from-scratch implementations in NumPy, mathematical validations against state-of-the-art Hugging Face models, parameter breakdowns, attention visualization, decoding strategies, KV-cache latency benchmarks, factual hallucination audits, zero-shot transfer evaluations on 200 IMDB reviews, and a multi-task production NLP mini-application.

---

## 📑 Table of Contents
1. [Repository Structure](#-repository-structure)
2. [Hugging Face Models Used](#-hugging-face-models-used)
3. [The 7 Core Concepts Explained](#-the-7-core-concepts-explained)
4. [Generated Visualizations & Captions](#-generated-visualizations--captions)
5. [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
6. [Measured Disadvantages in Practice](#-measured-disadvantages-in-practice)
7. [Mini Application: `text_toolkit`](#-mini-application-text_toolkit)
8. [Video Demonstration](#-video-demonstration)
9. [Installation & Execution](#-installation--execution)

---

## 📂 Repository Structure

```text
├── DL_PR6.ipynb               # Primary notebook (104 cells, 0 errors, full outputs)
├── pr6.ipynb                  # Mirror notebook for PR6 submission
├── DL_PR6.html                # Exported full HTML report with embedded visuals
├── pr6.html                   # Mirror HTML report
├── plots/                     # 13 High-resolution figures generated during execution
│   ├── task2_2_attention_matrix_scratch.png
│   ├── task3_2_real_attention_heads_grid.png
│   ├── task3_3_attention_entropy_comparison.png
│   ├── task3_4_sinusoidal_positional_encoding.png
│   ├── task3_6_distilbert_position_similarity.png
│   ├── task4_3_parameter_distribution_breakdown.png
│   ├── task4_5_distilbert_bidirectional_attention.png
│   ├── task4_6_causal_mask_scratch.png
│   ├── task4_7_gpt2_causal_attention.png
│   ├── task5_4_t5_cross_attention.png
│   ├── task6_1_confusion_matrix.png
│   ├── task6_2_truncation_accuracy_vs_length.png
│   └── task6_3_forward_pass_time_vs_length.png
├── generate_and_execute_lab.py # Programmatic builder & validation runner
├── test_all_tasks.py          # Quick pre-flight validation script
├── requirements.txt           # Library versions used
├── .gitignore                 # Cache and build ignore rules
└── README.md                  # Comprehensive technical documentation
```

---

## 🤖 Hugging Face Models Used

All models are instantiated using exact Hugging Face identifiers:
- **`distilbert-base-uncased-finetuned-sst-2-english`** (Encoder-Only): 6 layers, 12 heads, 768 hidden dimension, 66.9M parameters. Fine-tuned on the Stanford Sentiment Treebank for sentiment classification.
- **`distilbert-base-uncased`** (Encoder-Only): Pretrained masked language model used for bidirectional token filling (`fill-mask`).
- **`gpt2`** (Decoder-Only): 12 layers, 12 heads, 768 hidden dimension, 124M parameters. Autoregressive language model used for causal attention verification, manual decoding loops, sampling strategy comparisons, KV-cache benchmarking, and hallucination audits.
- **`t5-small`** (Encoder-Decoder): 60M parameters. Text-to-text conditional transformer used for English-to-French translation, abstractive summarization, and cross-attention alignment analysis.

---

## 🧠 The 7 Core Concepts Explained

### 1. Tokenization & Subword Representation
Traditional NLP represented vocabularies at either the full-word or character level. Full-word vocabularies suffer from the Out-Of-Vocabulary (OOV) dilemma where unseen words collapse into `<UNK>`, while character tokenization yields overly long sequences that strain attention. Subword tokenization (WordPiece in BERT/DistilBERT, BPE in GPT-2, SentencePiece in T5) strikes the optimal compromise: common words remain intact, while rare or complex terms are broken into morphological subwords (e.g., `pseudopseudohypoparathyroidism` into subwords with `##` prefixes). Special tokens like `[CLS]` provide aggregate sentence-level classification states, while `[SEP]` marks sequence boundaries.

### 2. Scaled Dot-Product Attention & Self-Attention
Self-attention maps an input sequence $X$ to Query ($Q$), Key ($K$), and Value ($V$) representations via learnable linear projections ($W_q, W_k, W_v$). The Query vector acts as an information retrieval request, Keys act as index identifiers, and Values contain the substantive content:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
If $q_i, k_i \sim \mathcal{N}(0, 1)$, the variance of their dot product is $d_k$. Without scaling by $\frac{1}{\sqrt{d_k}}$, large values of $d_k$ (e.g., 64) inflate dot products into extreme magnitudes, pushing Softmax into saturated flat tails where probabilities collapse to $1.0$ and gradients vanish. Scaling ensures unit variance and stable backpropagation. Unlike sequential RNNs, self-attention executes across the entire sequence simultaneously in $O(1)$ sequential operations.

### 3. Multi-Head Attention Mechanism
A single attention pattern averages all pairwise token interactions into one weight distribution per token, failing to capture simultaneous linguistic facets. Multi-Head Attention linearly splits the $d_{model}=768$ space into $h=12$ distinct subspaces of dimension $d_k=64$:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
Each head specializes in distinct syntactic, coreferent, or positional patterns (e.g., linking the pronoun *"it"* back to its antecedent *"animal"*). Because each head operates on dimension $d_k = d_{model}/h$, the total computational complexity remains $O(N \cdot d_{model}^2)$, identical to a single wide head.

### 4. Positional Encodings (Sinusoidal vs. Learned vs. Relative)
Because self-attention is strictly permutation-equivariant ($\text{Attention}(P X) = P\,\text{Attention}(X)$), it is order-blind. Vaswani et al. introduced fixed sinusoidal encodings using geometric frequencies:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
These allow the model to learn relative position offsets via linear transformations. DistilBERT and BERT replace fixed sinusoids with learned continuous embeddings (introducing a hard ceiling of 512 tokens). T5 discards absolute positional tables entirely, adding learned relative position biases directly to attention logits.

### 5. Layer Normalization vs. Batch Normalization
BatchNorm normalizes features across the batch and sequence dimension, making it unstable at small batch sizes and susceptible to statistical distortion from synthetic `[PAD]` tokens. Layer Normalization normalizes across the **feature dimension** for each token independently:
$$y = \gamma \odot \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$$
LayerNorm is independent of batch size ($B=1$ behaves identically to $B=128$), guarantees consistent computation during training and inference, and accommodates variable-length sequences without padding leakage.

### 6. Encoder-Only vs. Decoder-Only vs. Encoder-Decoder Architectures
- **Encoder-Only (DistilBERT)**: Unconstrained bidirectional self-attention where every token attends to all tokens in both directions. Optimal for classification, extraction, and understanding.
- **Decoder-Only (GPT-2)**: Applies a causal lower-triangular mask ($-\infty$ strictly above the diagonal) so tokens can only attend to past and present tokens. Tailored for autoregressive text generation.
- **Encoder-Decoder (T5)**: Merges a bidirectional encoder with a causal decoder via cross-attention, where Decoder Queries dynamically attend to Encoder Keys and Values to perform sequence-to-sequence transformation (translation and summarization).

### 7. Decoding Strategies & KV-Caching
- **Decoding Strategies**: Greedy search picks the argmax token at each step (deterministic but repetitive); Beam search maintains top-$k$ sequence candidates; Temperature sampling softens logits ($z/T$) to balance creativity vs. coherence; Top-$k$ and Nucleus (Top-$p$) sampling prune unlikely tail tokens.
- **KV-Caching**: During autoregressive decoding, past tokens are frozen and their Keys and Values do not change. By caching past $K$ and $V$ tensors rather than recalculating them at every step, per-step complexity drops from $O(N^2)$ to $O(N)$, unlocking a $>2\times$ generation speedup.

---

## 📊 Generated Visualizations & Captions

| Figure | Filename | Key Technical Insight |
| :--- | :--- | :--- |
| **Self-Attention from Scratch** | `plots/task2_2_attention_matrix_scratch.png` | Validates custom stable Softmax and confirms all attention rows sum strictly to $1.0$. |
| **Real Attention Heads Grid** | `plots/task3_2_real_attention_heads_grid.png` | $2\times 3$ grid of DistilBERT heads showing delimiter tracking (`[CLS]`), local diagonal syntax, and coreference resolution. |
| **Head Attention Entropy** | `plots/task3_3_attention_entropy_comparison.png` | Compares Layer 0 (broad, high entropy) vs. Layer 5 (focused, lower entropy) across all 12 heads. |
| **Sinusoidal Positional Encoding** | `plots/task3_4_sinusoidal_positional_encoding.png` | Visualizes the geometric frequency spectrum across 50 token positions and 128 embedding dimensions. |
| **Learned Positional Embeddings** | `plots/task3_6_distilbert_position_similarity.png` | Cosine similarity heatmap of DistilBERT's first 64 learned positions, showing continuous spatial decay away from the diagonal. |
| **Parameter Breakdown** | `plots/task4_3_parameter_distribution_breakdown.png` | Demonstrates that Feed-Forward blocks hold >42% of total parameters, acting as key-value associative memory. |
| **Bidirectional Attention** | `plots/task4_5_distilbert_bidirectional_attention.png` | DistilBERT attention showing prominent non-zero weights above the diagonal (bidirectional context). |
| **Causal Attention Scratch** | `plots/task4_6_causal_mask_scratch.png` | Lower-triangular causal attention matrix created by setting upper-triangular logits to $-\infty$. |
| **GPT-2 Causal Head Attention** | `plots/task4_7_gpt2_causal_attention.png` | Confirms that GPT-2 production attention has zero weights strictly above the diagonal ($\max < 10^{-6}$). |
| **T5 Cross-Attention Heatmap** | `plots/task5_4_t5_cross_attention.png` | Aligns French target tokens with English source tokens (*"maison"* attends directly to *"house"*). |
| **Confusion Matrix (200 IMDB)** | `plots/task6_1_confusion_matrix.png` | Zero-shot transfer confusion matrix for DistilBERT on 200 IMDB test reviews. |
| **Truncation vs. Accuracy** | `plots/task6_2_truncation_accuracy_vs_length.png` | Quantifies accuracy drops from 512 down to 64 tokens alongside execution time. |
| **Forward Pass Latency** | `plots/task6_3_forward_pass_time_vs_length.png` | Benchmarks single forward pass latency from 32 to 512 tokens. |

---

## 📈 Evaluation & Benchmark Results

### 1. Pretrained Sentiment Evaluation (200 IMDB Reviews)
DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) zero-shot transfer performance:
- **Accuracy**: **89.00%**
- **Precision**: **89.80%**
- **Recall**: **88.00%**
- **F1 Score**: **88.89%**
- *Note*: Evaluated out-of-the-box without any fine-tuning on the IMDB dataset.

### 2. Truncation Length Study (Task 6.2)
| Context Window (`max_length`) | Classification Accuracy | Total Inference Time (s) |
| :---: | :---: | :---: |
| **64 tokens** | 82.50% | 1.84 s |
| **128 tokens** | 86.50% | 2.61 s |
| **256 tokens** | 88.00% | 3.82 s |
| **512 tokens** | **89.00%** | 5.63 s |
- **Reviews Exceeding 512 Tokens**: **34 / 200 (17.0%)** of reviews were longer than 512 tokens, suffering unavoidable truncation.

### 3. KV-Cache Acceleration Benchmark (Task 5.7)
Generating 100 new tokens with GPT-2 on CPU:
- **With KV-Cache (`use_cache=True`)**: **0.86 seconds**
- **Without KV-Cache (`use_cache=False`)**: **1.94 seconds**
- **Empirical Speed-Up**: **2.25x faster**

---

## ⚠️ Measured Disadvantages in Practice

1. **Context Window Ceiling & Information Truncation**:
   In Task 6.2, truncating reviews from 512 to 64 tokens resulted in a **6.5% drop in accuracy**. Movie reviews frequently place their decisive verdict in the final paragraph. Because standard DistilBERT is bounded by a 512-position lookup table, 17% of reviews could not be fully read.
2. **Factual Hallucination in Autoregressive Models (Task 6.4)**:
   When prompted with factual queries, GPT-2 generated:
   - *"The capital of Australia is"* $\to$ Generated *"Sydney"* (Wrong - common statistical co-occurrence hallucination).
   - *"The author of Pride and Prejudice is"* $\to$ Failed to identify Jane Austen accurately.
   - Proves that autoregressive decoders are statistical token continuators, not verified knowledge graphs.

---

## 🛠️ Mini Application: `text_toolkit`

The production-ready utility `text_toolkit(text)` orchestrates DistilBERT and T5 into a unified multi-task pipeline:
- **Sentiment Classification**: DistilBERT evaluates sentiment and assigns a confidence score.
- **Abstractive Summarization**: T5 extracts the core narrative into a concise 60-word summary.
- **Multilingual Translation**: T5 translates the summary into French.
- **Defensive Boundary Handling**: Gracefully catches empty/whitespace strings and automatically truncates oversized passages (>450 words) with user notifications.

---

## 🎥 Video Demonstration

The walkthrough video covers the 7 core concepts with screen and face recordings, followed by a live interactive demo of `text_toolkit`:
- **Video Filename**: `DL_PR6_YourName_GRID.mp4`
- **Format**: MP4, 1080p, 5–10 minutes
- **Video Link**: [Watch the PR6 Project Walkthrough Video](https://drive.google.com/file/d/your-video-link-placeholder/view) *(Paste final hosting link here)*

---

## 🚀 Installation & Execution

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/Transformers_Lab_PR6.git
cd Transformers_Lab_PR6
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook DL_PR6.ipynb
```
Select **Kernel > Restart & Run All** to execute all 104 cells cleanly with zero errors.
