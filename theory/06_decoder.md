# Transformer Decoder

**Encoder**: Processes the source language.

**Decoder**: Learns to generate the target language.

---

## 1. Why do we feed "Outputs" into the Decoder?

The architecture diagram shows "Outputs (shifted right)" going into the bottom of the Decoder. Why are we feeding the model the *output*?

### Phase A: Real-World Use (Inference)
When you actually use a Translator app, **it does not know the answer**. It works in a slow loop:
1. You give it the source sentence.
2. You give the Decoder a `<SOS>` (Start of Sentence) token.
3. It processes and predicts the first word: `"Yo"`.
4. You take `"Yo"`, add it to the Decoder input `["<SOS>", "Yo"]`, and run the model again.
5. It predicts `"soy"`.
6. You loop this process one by one until it outputs `<EOS>` (End of Sentence).

### Phase B: Training (The Need for Speed)
Generating words one by one in a loop is incredibly slow. If we trained the model this way on millions of sentences, it would take years. 
During training, **we already hold the correct answer key** in our dataset. 

Instead of looping, we take the entire correct answer `["<SOS>", "Yo", "soy", "feliz"]` and feed it into the Decoder **all at once**. 
By doing this, we can calculate the loss and train the model for every single word prediction simultaneously:
1. At position `["<SOS>"]`, did it predict `"Yo"`?
2. At position `"Yo"`, did it predict `"soy"`?
3. At position `"soy"`, did it predict `"feliz"`?

---

## 2. Causal Masking (The "Blindfold")

### 2.1 The Training Bug
Feeding the entire answer key at once creates a massive mathematical bug. The Self-Attention mechanism calculates relationships between *every* word in the array. 

If we feed `["<SOS>", "Yo", "soy", "feliz"]` into the Decoder, the mathematical calculation for `"Yo"` will literally look ahead at `"soy"` and `"feliz"`. The model will just copy the answer from the future instead of learning to predict it.

### 2.2 The Matrix Solution
To fix this bug, we apply a **Lower-Triangular Mask** to the attention scores ($QK^T / \sqrt{d_k}$) exactly one step **before** the Softmax. 
We force every future word's score to $-\infty$.

$$
\text{Scores} = \begin{bmatrix}
Q_{\text{Yo}} \cdot K_{\text{Yo}} & Q_{\text{Yo}} \cdot K_{\text{soy}} & Q_{\text{Yo}} \cdot K_{\text{feliz}} \\
Q_{\text{soy}} \cdot K_{\text{Yo}} & Q_{\text{soy}} \cdot K_{\text{soy}} & Q_{\text{soy}} \cdot K_{\text{feliz}} \\
Q_{\text{feliz}} \cdot K_{\text{Yo}} & Q_{\text{feliz}} \cdot K_{\text{soy}} & Q_{\text{feliz}} \cdot K_{\text{feliz}}
\end{bmatrix} \xrightarrow{\text{Mask}}
\begin{bmatrix}
Q_{\text{Yo}} \cdot K_{\text{Yo}} & \mathbf{-\infty} & \mathbf{-\infty} \\
Q_{\text{soy}} \cdot K_{\text{Yo}} & Q_{\text{soy}} \cdot K_{\text{soy}} & \mathbf{-\infty} \\
Q_{\text{feliz}} \cdot K_{\text{Yo}} & Q_{\text{feliz}} \cdot K_{\text{soy}} & Q_{\text{feliz}} \cdot K_{\text{feliz}}
\end{bmatrix}
$$

Because $e^{-\infty} = 0$, applying Softmax ($e^x / \sum{e^x}$) zeroes out the future:

$$
\text{Attention Weights} = \begin{bmatrix}
1.0 & \mathbf{0} & \mathbf{0} \\
0.3 & 0.7 & \mathbf{0} \\
0.1 & 0.5 & 0.4
\end{bmatrix}
$$

**Result**: 
- Row 1 ("Yo") gets **0%** information from the future.
- Row 2 ("soy") looks at itself and "Yo".
- We calculate sequential steps massively in parallel.

---

### 2.3 The $Q, K, V$ in Masked Self-Attention
Just like the Encoder, the Decoder generates $Q, K,$ and $V$ from the **exact same source**—its own target sequence (`["<SOS>", "Yo", "soy", "feliz"]`).
* **Query ($Q$)**: "I am the word 'soy', what should I pay attention to?"
* **Key ($K$) & Value ($V$)**: "We are the words in the sequence."

The Mask simply steps in to stop $Q_{\text{soy}}$ from looking at $K_{\text{feliz}}$. It forces $Q_{\text{soy}}$ to only pull Values from `<SOS>`, `Yo`, and `soy`.

---

## 3. Cross-Attention (The Translation Bridge)

Masked Self-Attention looks at the *Decoder's own past history*. 
**Cross-Attention** looks at the *Encoder's fully processed source sentence*.

This is the only place the two halves of the model physically connect.

### 3.1 The $Q, K, V$ Split
Unlike Self-Attention where $Q, K,$ and $V$ all come from the exact same sentence, Cross-Attention severs them:

*   **Keys ($K$) & Values ($V$)**: Come from the Encoder (Source: `"I am happy"`).
*   **Queries ($Q$)**: Come from the Decoder (Target State: `"Yo"`).

### 3.2 Concrete Math Example (Zooming in on one word)
Remember, the model processes the entire array `["<SOS>", "Yo", "soy", "feliz"]` in parallel at the exact same time. But to understand the math, let's zoom in entirely on the exact moment the Decoder is processing the single word `"Yo"` (Position 1). 

It needs to know which English words to focus on to help predict the next Spanish word.

**1. Calculate Relevance (Dot Product)**
The Decoder takes its Query vector for `"Yo"` and mathematically tests it against all three Keys from the Encoder:
$$
\text{Scores} = \begin{bmatrix}
Q_{\text{Yo}} \cdot K_{\text{I}} & Q_{\text{Yo}} \cdot K_{\text{am}} & Q_{\text{Yo}} \cdot K_{\text{happy}}
\end{bmatrix}
$$

Because the model has been trained, the vectors naturally align where translations map. The dot product for `Yo` $\leftrightarrow$ `I` will be massive.
$$
\text{Scores} = \begin{bmatrix} 9.2 & 0.1 & -0.5 \end{bmatrix}
$$

**2. Convert to Percentages (Softmax)**
We push the scores through Softmax ($e^x / \sum{e^x}$) to turn them into strict percentages that sum to 1.0:
$$
\text{Weights} = \begin{bmatrix} 0.99 & 0.01 & 0.00 \end{bmatrix}
$$

**3. Pull the Meaning (Weighted Sum)**
Finally, the Decoder builds its new "Translation Context Vector" by multiplying these percentages against the Encoder's Values ($V$):
$$
\text{Context Vector} = (0.99 \times V_{\text{I}}) + (0.01 \times V_{\text{am}}) + (0.00 \times V_{\text{happy}})
$$

**The Result**: The Decoder's stream for `"Yo"` has successfully reached across the bridge, mathematically grabbed $99\%$ of the English concept for `"I"`, and pulled it into the Spanish generation stream. 

---

## 4. The Final Prediction (Linear & Softmax)

So, we just built a rich context vector for the `"Yo"` position that contains both the Spanish history (`"Yo"`) AND the English context (`"I"`). How does this actually become the predicted word `"soy"`?

### 4.1 Feed-Forward & Residuals
First, this context vector passes through an independent basic neural network (Linear $\rightarrow$ ReLU $\rightarrow$ Linear) to finalize its internal logic. Residual Add & Norm connections loop around every layer to preserve gradients.

### 4.2 The Final Linear Dictionary
The vector leaves the end of the Decoder blocks at size $d_{model}$ (e.g., 512 numbers). 
We pass it through a massive final **Linear layer** that acts as a vocabulary dictionary. It expands those 512 numbers into $50,000$ numbers (one slot for every possible word in the Spanish language).

### 4.3 Softmax Probabilities
We apply **Softmax** to those 50,000 raw numbers. 
The neuron specifically assigned to the word `"soy"` lights up with a $98\%$ probability. The neuron for `"perro"` (dog) gets $0.001\%$. 

**Because `"soy"` has the highest probability, it is the official prediction.**

### 4.4 Summary of the Parallel Timeline
Even though we process the whole array at once, every single position is predicting the *next* step independently:

*   **Position 0 input**: `<SOS>` $\xrightarrow{\text{focuses on 'I am'}}$ outputs prediction `Yo`
*   **Position 1 input**: `Yo` $\xrightarrow{\text{focuses on 'am'}}$ outputs prediction `soy`
*   **Position 2 input**: `soy` $\xrightarrow{\text{focuses on 'happy'}}$ outputs prediction `feliz`

The model's final generated output at index $t$ is ALWAYS the prediction for what the word at index $t+1$ should logically be.

---

## 5. Architectural Nuances

### 5.1 The $\sqrt{d_{model}}$ Embedding Scale
Before adding positional encodings to word embeddings, we multiply the embeddings by $\sqrt{d_{model}}$ (e.g., $\sqrt{512} \approx 22.6$).

**Why do we do this?**
* **Embeddings** start as very small numbers (low variance weights).
* **Positional Encodings** are strong, rigid signals from sine and cosine functions (between $-1$ and $1$).
* If we added them directly, the strong positional signals would completely drown out the fragile meaning of the word.
* Scaling the embeddings up makes them "loud" enough to survive the addition without losing their meaning.

> [!NOTE]
> This is an entirely separate mathematical operation from the $1/\sqrt{d_k}$ scaling used inside self-attention (which prevents vanishing Softmax gradients).

### 5.2 Decoder Depth (`num_layers`)
The Decoder normally consists of a stack of multiple identical blocks (the original paper used `num_layers = 6`). 

These blocks do **not** run in parallel like multi-head attention. They run **sequentially**:
1. **Layer 1**: Receives target embeddings + encoder context. Finds basic relationships.
2. **Layer 2**: Takes Layer 1's exact output as input. Refines context further.
3. **Layer 6**: Produces a massive, highly-contextual understanding of the sentence, ready for the final prediction.

**The Assembly Line**: 
Input $\rightarrow$ Block 1 $\rightarrow$ Block 2 $\rightarrow \dots \rightarrow$ Block 6 $\rightarrow$ Final Output.

### 5.3 Stacking with `nn.ModuleList` (PyTorch Specifics)
In PyTorch, you cannot store this sequence of blocks in a standard Python list like `[Block1, Block2]`. 

**The consequence of a standard list**:
* PyTorch's computational graph becomes blind to the blocks.
* It fails to track their weights.
* It skips updating them entirely during backpropagation.

**The solution**:
You must wrap the stack in `nn.ModuleList`. This explicitly registers every block as an official sub-module of the Decoder, guaranteeing gradients flow cleanly through every layer.
