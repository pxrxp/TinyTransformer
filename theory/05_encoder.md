# Transformer Encoder

**Encoder**: Processes the source language.

**Decoder**: Learns to generate the target language.


The Transformer processes all tokens in parallel. Unlike RNNs, it has no built-in sense of order or time. Every "fix" in the architecture exists to solve a problem that this parallelism creates.

---

## 1. Positional Encoding: Reconstructing Order

Standard attention is permutation-invariant: $\text{softmax}(QK^T)V$ yields the same result for "Dog bites human" and "Human bites dog." We must "brand" each word vector $x \in \mathbb{R}^{d_{model}}$ with its position $pos$ before it enters the attention block.

### 1.1 First Principles: The Binary Counter Intuition

If we want to represent ordered numbers uniquely, we look at a binary counter ($0-7$):

| Position ($pos$) | Bit 2 ($2^2$) | Bit 1 ($2^1$) | Bit 0 ($2^0$) |
| :--- | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |
| **Frequency** | **Low** | **Mid** | **High** |

**The Pattern**:
- **Bit 0** (LSB): Flips every step. It captures high-frequency, local adjacency.
- **Bit 2** (MSB): Flips only every 4 steps. It captures low-frequency, global position.

### 1.2 Moving to Sinusoids (The Smooth Binary Clock)

Neural networks struggle with binary because it is "jagged"—a bit flips from $0 \to 1$ instantly, creating non-differentiable jumps. We solve this by replacing the discrete square-wave of binary digits with smooth, continuous **sinusoids**.

**The Formula:**
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**Mapping the Intuition:**
- **High Frequency ($i \to 0$)**: The wave oscillates rapidly, mirroring the **LSB** of our binary counter. It tells the model "who is my immediate neighbor."
- **Low Frequency ($i \to d/2$)**: The wave moves slowly, mirroring the **MSB**. It tells the model "where am I in the document."
- **Why the $10000$ base?**: It ensures the wavelength of the slowest dimension is $2\pi \cdot 10000$, ensuring global uniqueness over long documents.

**The "Solo Sine" Failure (Periodicity)**
Why not just use one sine wave? Because $\sin(1) = \sin(2\pi + 1)$. A single wave repeats and the model loses its place. Multi-frequency encoding mimics binary: even if the "fast" bits repeat, the "slow" bits keep the overall state unique.

### 1.3 The Solution: The Sin/Cos Pair and Rotation

A sinusoid alone encodes absolute points. By using a **Sine + Cosine pair** for each frequency, we create a rotational signal. Because of the identity $\sin(A+B) = \sin A \cos B + \cos A \sin B$, the encoding for a relative offset $pos+k$ is a linear rotation of the encoding at $pos$:

$$
\begin{pmatrix} PE_{pos+k, 2i} \\ PE_{pos+k, 2i+1} \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} PE_{pos, 2i} \\ PE_{pos, 2i+1} \end{pmatrix}
$$

**Why this matters**: This allows the attention mechanism to learn that two words are exactly "$k$ steps apart" purely through linear transformations (dot products), regardless of their absolute position in a document.

---

## 2. Layer Normalization

### 2.1 The Problem: Activations Exploding or Vanishing

Every linear layer computes $h = Wx$ where $h, x \in \mathbb{R}^{d_{model}}$ and $W \in \mathbb{R}^{d_{model} \times d_{model}}$. If the average weight magnitude is $1.1$, after $L$ layers:

$$\|h_L\| \approx 1.1^L \cdot \|x\|$$

| Layers | Factor | Activation scale |
| :--- | :---: | :---: |
| 1 | $1.1^1$ | $1.1 \times \|x\|$ |
| 10 | $1.1^{10}$ | $2.6 \times \|x\|$ |
| 50 | $1.1^{50}$ | $117 \times \|x\|$ |
| 96 | $1.1^{96}$ | $\approx 10000 \times \|x\|$ |

If weights are $0.9$ instead: $0.9^{96} \approx 0.00008$. The signal vanishes. Both cases make gradients useless during backpropagation.

### 2.2 What is Normalization?

**Goal**: After any transformation, make the output have mean $0$ and variance $1$ so it always lands in the same numerical range.

**Step 1**: Compute the mean of the vector's components, then subtract it (re-centering):
$$\mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad \hat{x}_j = x_j - \mu$$
Now $\hat{x}$ has mean $0$.

**Step 2**: Compute the standard deviation and divide by it (re-scaling):
$$\sigma = \sqrt{\frac{1}{d}\sum_{j=1}^{d} \hat{x}_j^2}, \qquad \hat{x}_j \leftarrow \frac{\hat{x}_j}{\sigma}$$
Now $\hat{x}$ has mean $0$, variance $1$.

**Step 3**: The model might need a different scale for the task (e.g. ReLU dies at 0). So we let it rescale and shift back with learned parameters $\gamma, \beta$:
$$x_{\text{out}} = \gamma \hat{x} + \beta$$

The only open question is: **which components of $x$ do we average over to get $\mu$ and $\sigma$?**

### 2.3 Why BatchNorm Fails for Sequences

**Batch Normalization**: For position $t$, compute $\mu$ and $\sigma$ across all $B$ sentences in the batch:
$$\mu_t = \frac{1}{B}\sum_{b=1}^{B} x_{b,t}$$

In NLP, sentences have different lengths, so shorter ones are padded with zeros. Consider a batch of 2 sentences, max length 4:

| Position | Sentence A: "The cat sat" | Sentence B: "The cat sat on the mat" |
| :--- | :---: | :---: |
| 1 | "The" $\to [0.2, -0.5, \dots]$ | "The" $\to [0.2, -0.5, \dots]$ |
| 2 | "cat" $\to [0.8, 0.1, \dots]$ | "cat" $\to [0.8, 0.1, \dots]$ |
| 3 | "sat" $\to [-0.3, 0.9, \dots]$ | "sat" $\to [-0.3, 0.9, \dots]$ |
| 4 | **$[0, 0, \dots]$ (pad)** | "on" $\to [0.5, -0.2, \dots]$ |

At position 4, BatchNorm computes:
$$\mu_4 = \frac{\mathbf{0} + [0.5, -0.2, \dots]}{2} = [0.25, -0.1, \dots]$$

The batch mean is pulled toward zero by the padding. The normalization of "on" is now corrupted by a zero vector from Sentence A. This is **Cross-Sentence Pollution**.

### 2.4 The Solution: Manual Layer Normalization

We implement this by calculating the statistics directly over the token dimension:

1. **Mean**: $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
2. **Variance**: $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$
3. **Normalize**: $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. **Scale & Shift**: $y = \gamma \hat{x} + \beta$ (using learnable parameters $\gamma, \beta$)

By writing this explicitly in code, we avoid the black-box of standard libraries and see exactly how each token's features are re-centered and re-scaled. Each token normalizes itself using only its own $d_{model}$ numbers. What Sentence A's padding looks like is completely irrelevant to how "on" is normalized.

---

## 3. Residual Connections

### 3.1 The Problem: Gradients Vanish Through Depth

Let $x_l \in \mathbb{R}^{d_{model}}$ be the output vector of layer $l$, and $L$ the total number of layers. During backprop the gradient of the loss $\mathcal{L}$ w.r.t. layer $l$'s input uses the chain rule:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l}^{L-1} f_k'(x_k)$$

Each $f_k'(x_k)$ is a Jacobian matrix $\in \mathbb{R}^{d_{model} \times d_{model}}$. If each has spectral norm $0.9$ and there are $L-l = 48$ layers between $l$ and the output:
$$0.9^{48} \approx 0.008$$
The gradient reaching layer $l$ is $\approx 1\%$ of what it was at the output. That layer barely trains.

### 3.2 The Solution: Skip Connections

Instead of $x_{l+1} = f(x_l)$, every sub-layer does:
$$x_{l+1} = x_l + f(x_l)$$

Now the gradient for any layer $l$ includes a direct additive path:
$$\frac{\partial x_{l+1}}{\partial x_l} = \mathbf{I} + \frac{\partial f}{\partial x_l}, \quad \mathbf{I} \in \mathbb{R}^{d_{model} \times d_{model}}$$

Even if $f'(x_l) \approx 0$ (saturated or uninitialized), the $\mathbf{I}$ term keeps the gradient alive. Deep layers receive signal from the very beginning.

- **Information Zeroing**: Because we add, a layer can erase any signal it no longer needs by learning to output its negative: $x + (-x) = 0$.

---

## 4. Feed-Forward Networks

### 4.1 The Problem: Attention is Linear

Attention computes a weighted sum of value vectors:
$$\text{Attn}(Q,K,V) = AV, \quad A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}$$

where $A \in \mathbb{R}^{L \times L}, Q, K \in \mathbb{R}^{L \times d_k}$, $V \in \mathbb{R}^{L \times d_v}$, and $L$ is the sequence length. $A$ is a matrix of scalars (the attention weights), so $AV$ is a matrix multiplication — a linear map. Stacking two attention layers:
$$A_2(A_1 V) = (A_2 A_1) V$$
which is again a single matrix multiplication. $L$ attention layers collapse to **one** linear map. The model cannot represent any conditional logic.

### 4.2 Why Linear Maps Can't Do Conditional Logic

A linear function satisfies $f(ax + by) = af(x) + bf(y)$. The output is always a smooth, weighted blend of inputs — it cannot produce a threshold or an if-then split.

Concrete example: suppose we want to output $1$ if $x > 0$ and $0$ otherwise.
- For a linear $f$: $f(3) = 3w + b$ and $f(-2) = -2w + b$. No single choice of $w, b$ can make both equal to the correct output $1$ and $0$ respectively.
- For ReLU: $\max(0, 3) = 3$ (active) and $\max(0, -2) = 0$ (dead). The neuron has a threshold at $0$ — exactly the kind of conditional behavior linear maps cannot express.

### 4.3 The Solution: ReLU + Expansion

After every attention block, each token $x \in \mathbb{R}^{d_{model}}$ is independently passed through:
$$\text{FFN}(x) = \underbrace{\max(0,\, xW_1 + b_1)}_{\text{expand to }4d_{model}\text{ + non-linearity}}\,W_2 + b_2$$
$$W_1 \in \mathbb{R}^{d_{model} \times 4d_{model}}, \quad W_2 \in \mathbb{R}^{4d_{model} \times d_{model}}$$

- **ReLU ($\max(0, \cdot)$)**: Each of the $4d_{model}$ neurons is independently either active or dead. The combination of many such on/off decisions is what lets the FFN implement piecewise conditional logic.
- **$4d_{model}$ expansion**: More neurons = more independent thresholds the model can learn. Compressing back to $d_{model}$ forces the model to distill those decisions into the main stream.
- Attention **gathers** information from other tokens. The FFN **processes** it, independently, token by token.

---

## 5. Dropout: Stochastic Regularization

Deep Transformers are prone to **co-adaptation**—where neurons become overly dependent on each other, leading to overfitting on specific patterns in the training set.

### 5.1 Bernoulli Masking

Dropout is implemented by multiplying the input tensor $x$ element-wise by a mask $m$ sampled from a **Bernoulli distribution**:

$$P(m_i=k) = \begin{cases} 1-p & \text{if } k=1 \\ p & \text{if } k=0 \end{cases}$$

Where $p$ is the dropout probability. In code, this is executed via `torch.rand_like(x) > p`.

### 5.2 Expected Value Scaling

During training, applying the mask $m$ reduces the expected magnitude of the tensor. For an indicator variable $m_i \sim \text{Bernoulli}(1-p)$, the expected value is:

$$E[m_i] = 1 \cdot (1-p) + 0 \cdot p = 1-p$$

To ensure the output $y$ has the same expected value as the input $x$ (preserving activation magnitude), we apply **Inverted Dropout** by scaling the output by $\frac{1}{1-p}$:

$$E\left[\frac{x_i \cdot m_i}{1-p}\right] = \frac{x_i}{1-p} E[m_i] = \frac{x_i}{1-p} (1-p) = x_i$$

**Consequence**: The expected sum and variance of the activations remain consistent between training (where elements are dropped) and evaluation (where $p=0$ and no elements are dropped).
