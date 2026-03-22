# Transformer from Scratch

A minimal, education-focused implementation of the Transformer architecture and its predecessors (RNN, LSTM, GRU) in PyTorch.

## Overview

This project is a first-principles derivation of modern sequence models. It focuses on the mathematical "why" behind each component, with a direct mapping between theoretical derivations and the source code.

## Architecture

- **[Transformer Encoder](modules/transformer_encoder.py)**: Parallel sequence processing with Sinusoidal Positional Encoding and Multi-Head Attention.
- **[Attention](modules/attention.py)**: Scaled Dot-Product and Multi-Head Attention implementations.
- **[Recurrent Units](modules/)**: Implementations of RNN, LSTM, and GRU for historical context.
- **[GPT](modules/gpt.py)**: Decoder-only autoregressive generation.
- **[BERT](modules/bert.py)**: Encoder-only masked language modeling.
- **[T5](modules/t5.py)**: Unified text-to-text Encoder-Decoder framework.

## Documentation

Comprehensive technical notes are available in the `theory/` directory:

1. [RNN](theory/01_rnn.md)
2. [LSTM](theory/02_lstm.md)
3. [GRU](theory/03_gru.md)
4. [Attention](theory/04_attention.md)
5. [Transformer Encoder](theory/05_encoder.md)
6. [Transformer Decoder](theory/06_decoder.md)
7. [Full Transformer](theory/07_full_transformer.md)
8. [GPT](theory/08_gpt.md)
9. [BERT](theory/09_bert.md)
10. [T5](theory/10_t5.md)


---
*Built for deep understanding of the Transfomer Architecture.*
