# Generative Pre-trained Transformer (GPT)

The original Transformer was built for purely bidirectional translation tasks (English to French), necessitating an Encoder to understand the English, and a Decoder to generate the French. 

**GPT throws half of the architecture away.** It is a **Decoder-Only** model focusing entirely on autoregressive text generation.

## 1. Nomenclature

What does "GPT" actually stand for mechanically?

*   **Generative**: The model physically generates sequence data. It operates "autoregressively"—meaning it outputs a single word, appends that word back to its own input prompt, and loops again to guess the next word.
*   **Pre-trained**: Before it can write poetry or code, the raw model undergoes "Pre-training". It is fed terabytes of unlabelled internet text and given one simple unsupervised task: *Guess the next word*. Over months of GPU time, attempting to guess the next word forces the model's weights to internally encode human grammar, logic, and factual world knowledge.
*   **Transformer**: It runs exclusively on the Decoder stack of the Attention Is All You Need architecture.

## 2. Decoder-Only Architecture

In the original full Transformer, the Decoder has a *Cross-Attention* block bridging it to the Encoder. Because GPT physically lacks an Encoder, **there is no cross-attention bridge**.

A single `GPTBlock` is stripped down to just two operations:
1.  **Masked Self-Attention**: The token looks at its own sequence history to understand grammar. A lower-triangular causal mask forces it to never look at future tokens (otherwise, during training, it would just cheat and copy the answer).
2.  **Feed-Forward Network (FFN)**: Standard logic gates. GPT famously swapped the original `ReLU` activation for `GELU` (Gaussian Error Linear Unit), which performs smoother, probablistic gating on negative numbers rather than just a hard cutoff at zero.

## 3. Architecture Flow

1.  **Input**: Feed a prompt "The robot is".
2.  **Embedding**: Map to dense continuous vectors.
3.  **Masked Stack**: Pass sequentially through $N$ stacked GPT Blocks.
4.  **Prediction**: Output logits mapping to the vocabulary scale. The highest probability is the predicted next word.
5.  **Autoregressive Loop**: Append the newly generated word to the prompt, and run the entire sequence back through the exact same flow.

## 4. Scaling and Unsupervised Learning

By dropping the Encoder constraint, GPT became a generalized "continuation engine." Instead of needing manually labelled, explicitly paired datasets (like French-to-English mappings), it scales infinitely. It learns syntax, logic, and factual reasoning purely dynamically by figuring out what words statistically logically come next in any given human context.
