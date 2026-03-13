# Transformer from Scratch

A minimal, education-focused implementation of the Transformer architecture and its predecessors (RNN, LSTM, GRU) in PyTorch.

## Overview

This project is a first-principles derivation of modern sequence models. It focuses on the mathematical "why" behind each component, with a direct mapping between theoretical derivations and the source code.

## Architecture

- **[Transformer](modules/transformer.py)**: Parallel sequence processing with Sinusoidal Positional Encoding and Multi-Head Attention.
- **[Attention](modules/attention.py)**: Scaled Dot-Product and Multi-Head Attention implementations.
- **[Recurrent Units](modules/)**: Implementations of RNN, LSTM, and GRU for historical context.

## Documentation

Comprehensive technical notes are available in the `theory/` directory:

1. [RNN: The Foundations](theory/01_rnn.md)
2. [LSTM: Gated Updates](theory/02_lstm.md)
3. [GRU: Simplified Gating](theory/03_gru.md)
4. [Attention: Dynamic Relevance](theory/04_attention.md)
5. [Transformer: Parallel Order](theory/05_transformer.md)


---
*Built for deep understanding of the Transfomer Architecture.*
