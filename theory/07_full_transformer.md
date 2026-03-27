# Transformer Architecture Flow

This document describes the high-level data flow through the Transformer model, mapping the implemented Python modules directly to the components shown in Figure 1 of "Attention Is All You Need". Detailed theoretical breakdowns of the individual blocks are available in `05_encoder.md` and `06_decoder.md`.

<img width="966" height="557" alt="image" src="https://github.com/user-attachments/assets/d8540194-dffe-4799-b076-a69600e7dbab" />

## Diagram Legend (Arrows and Symbols)

Before tracing the boxes, it is essential to define the mechanical routing symbols natively used in the original paper's figure:

*   **Solid Black Arrows ($\rightarrow$)**: Represent the flow of continuous numerical tensors, shape `(batch, sequence_length, d_model)`.
*   **Splitting Arrows (One diverges into three)**: When an arrow splits into three lines entering a "Multi-Head Attention" box, it represents the tensor identically duplicating itself to act simultaneously as the Query ($Q$), Key ($K$), and Value ($V$).
*   **The $\oplus$ Symbol (Addition)**: Represents element-wise tensor addition used in two explicit locations:
    1.  **Bottom**: Merging the computationally distinct Positional Encodings into the Input/Output Embeddings.
    2.  **Add & Norm Wrap**: The residual connection visually bypassing a block, added directly back to the block's mathematical output before normalization.
*   **$N \times$ (Grey Box Wrapper)**: Indicates that the entire operation set inside the grey bounding box is stacked and repeated sequentially $N$ times.

## 1. Input Processing

**Components**: `Inputs` $\rightarrow$ `Input Embedding` $\oplus$ `Positional Encoding`

*   **Source Input**: The sequence tokens of the source language.
*   **Target Input**: The sequence tokens of the target language, manually prefixed with a `<SOS>` token to train next-token prediction (represented by the text "shifted right").
*   **Embedding**: Both specific sequences are mapped to continuous vectors of dimension $d_{model}$.
*   **Addition ($\oplus$)**: Since the parallel model has no recurrent memory, distinct absolute positioning equations are generated and injected via element-wise addition ($\oplus$).

## 2. Encoder Stack ($N \times$)

**Components**: `Multi-Head Attention` $\rightarrow$ `Add & Norm` $\rightarrow$ `Feed Forward` $\rightarrow$ `Add & Norm`

*   The Encoder processes the embedded source sequence.
*   **Attention Split**: The input tensor strictly splits three ways, serving as $Q, K,$ and $V$ for self-attention natively.
*   Each of the $N$ identical stacked layers mathematically refines these context representations.
*   **Output**: A continuous contextual representation (memory) of the source sequence. This final output is visually routed across the large architectural gap to the Decoder stack.

## 3. Decoder Stack ($N \times$)

**Components**: `Masked Multi-Head Attention` $\rightarrow$ `Multi-Head Attention` $\rightarrow$ `Feed Forward` (with interleaved `Add & Norm` blocks)

*   The Decoder sequentially processes the embedded, shifted-right target sequence.
*   **Masked Attention Split**: The initial Decoder tensor splits three ways ($Q, K, V$). A lower-triangular causal mask computationally prevents indices from attending to subsequent sequence indices to preserve the autoregressive generation loop during simultaneous training.
*   **Cross-Attention Bridge (The Two Origin Arrows)**:
    *   **Arrow from Decoder**: The vertical output of the prior Masked Attention moves upward and provides exclusively the Query ($Q$).
    *   **Arrows from Encoder**: The lateral output of the Encoder stack crosses the architecture gap, natively splitting into two branches, serving as Keys ($K$) and Values ($V$). This constitutes the sole physical connection between the two halves of the entire model.
*   **Output**: The final refined target sequence representation, shape `(batch, target_seq_len, d_model)`.

## 4. Output Generation

**Components**: `Linear` $\rightarrow$ `Softmax` $\rightarrow$ `Output Probabilities`

*   **Linear Projection**: A linear layer maps the final stage's $d_{model}$-dimensional vector to logits mapping precisely to the target vocabulary magnitude.
*   **Softmax**: Converts the raw linear logits into a strict normalized probability distribution.
*   **Prediction (Top Arrow)**: The vocabulary index associated with the highest probability value corresponds to the predicted next token.
