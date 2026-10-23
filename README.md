# TinyTransformer

A compact PyTorch implementation of recurrent models, attention, and Transformers.

This repo is built as a study in sequence modeling. The goal is not to be a polished framework or a production training stack. It is to make the design explicit: how the recurrent units work, why attention was introduced, how positional encoding restores order, and how the Transformer family builds on those ideas.

We'll progress from this simple recurrent model:
![rnn](https://miro.medium.com/v2/resize:fit:1000/1*d_POV7c8fzHbKuTgJzCxtA.gif)

to the Transformer architecture:
![transformer](https://cdn.prod.website-files.com/65c0cd3f2acb33427e35b1f6/69b54b23ecf21150cb53d6f0_67996a0a63943f59ad038d15_AD_4nXfPWwNnlZ1mPxjz3qy0BAcHl3wfWtJDcBIXVj-IteOtu9UxB8uIBRsfcqA2sskc-pzw5Vihbnzeji0NTLHjbt8QTiGMAFIWBDtNHMPwv7LnFHWmViWy-rG3rkXedrhPYnT_pUZe.gif)

## Why this project exists

Sequence models evolved because language is sequential, but not all dependencies are local. Early models processed tokens one step at a time, which made them naturally sequential but weak at long-range dependencies. The field then moved to gated recurrence, then to attention, and finally to parallel Transformer architectures.

The important questions are:

- how information moves through time
- how a model decides what to remember
- how a token interacts with other tokens in the same sequence
- how order is represented without recurrence
- how generation differs from encoding

This project keeps those questions visible in code and theory.

## The progression

<img width="1920" height="766" alt="image" src="https://github.com/user-attachments/assets/b0d1a8ed-e4f1-4fc7-ad09-1eb09866513f" />

### 1. RNN

The simplest recurrent unit keeps a hidden state that is updated at each time step. This makes it easy to reason about, but long-range dependencies are hard because the signal has to pass through many transformations.

The core issue is the path length of dependency: the signal from an earlier token is repeatedly transformed and weakened as it propagates through time.

### 2. LSTM

LSTM adds gates to regulate memory flow:

- input gate decides what new information to store
- forget gate decides what to discard
- output gate decides what to expose

This makes it much easier to preserve useful information over long spans.

### 3. GRU

GRU simplifies the LSTM idea by combining some of the gating logic. It keeps the benefits of gated memory while reducing complexity.

### 4. Attention

Attention solves a different problem: instead of forcing all information to flow through a single hidden state, the model can look directly at relevant positions in the sequence.

A token computes a similarity score against others in the sequence, then uses a softmax-weighted combination of the corresponding values. This gives each position access to a dynamic context window.

### 5. Transformer

The Transformer removes recurrence entirely. Instead of processing sequence elements step by step, it uses:

- embeddings
- positional encodings
- self-attention
- feedforward layers
- residual connections
- layer normalization

Self-attention allows each token to attend to every other token in parallel. This is the key reason Transformers scale much better than recurrent models.

### 6. GPT, BERT, T5

These models reuse the same Transformer backbone, but differ in structure and objective:

- GPT: decoder-only, autoregressive generation
- BERT: encoder-only, masked token prediction
- T5: encoder-decoder, text-to-text mapping

The core mechanism is the same; the difference is in how the model is trained and how it is exposed to inputs and outputs.

## Included implementations

### Recurrent models
- `modules/rnn.py`
- `modules/lstm.py`
- `modules/gru.py`

### Attention
- `modules/attention.py`

### Transformer blocks
- `modules/transformer_encoder.py`
- `modules/transformer_decoder.py`
- `modules/transformer_block.py`
- `modules/transformer_layers.py`
- `modules/transformer.py`

### Language-model variants
- `modules/gpt.py`
- `modules/bert.py`
- `modules/t5.py`

## Project structure

- `modules/` — model implementations
- `theory/` — notes explaining each architecture and its intuition
- `demo.ipynb` — notebook walkthrough
- `README.md` — project overview

## Reading path

The theory files are meant to be read in sequence:

### 1. [Recurrent Neural Networks (RNN)](theory/01_rnn.md)
The foundations of sequential processing and the vanishing gradient problem.

### 2. [LSTMs](theory/02_lstm.md)
Addressing the gradient problem with gates and additive cell updates.

### 3. [GRUs](theory/03_gru.md)
A simplified gating architecture that merges hidden and cell states.

### 4. [Attention Mechanisms](theory/04_attention.md)
Transitioning from fixed-length context to dynamic relevance with Scaled Dot-Product and Multi-head attention.

### 5. [Transformer Encoder](theory/05_encoder.md)
### 6. [Transformer Decoder](theory/06_decoder.md)
### 7. [Full Transformer Architecture](theory/07_full_transformer.md)
Building the Encoder-Decoder architecture, Positional Encodings, and Add & Norm layers.

### 8. [GPT](theory/08_gpt.md)
Decoder-only architecture for autoregressive text generation.

### 9. [BERT](theory/09_bert.md)
Encoder-only architecture utilizing Masked Language Modeling (MLM).

This order matters because each model becomes easier to understand once the limitations of the previous one are clear.

## The main idea

The classic problem in sequence modeling is dependency: how does the model relate one token to another when they are far apart?

RNNs do this via hidden state. LSTMs and GRUs improve this with gating. Attention gives the model direct access to relevant positions. Transformers then replace recurrence with structured parallel attention, which is why they became the dominant architecture.

The repo is built around that story.

## Minimal usage

```python
from modules.transformer import Transformer

model = Transformer(
    source_vocab_size=5000,
    target_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)
