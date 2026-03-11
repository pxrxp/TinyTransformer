# Attention Mechanisms

Attention allows a model to dynamically focus on relevant sections of a sequence by computing alignment scores between all pairs of positions.

<p align="center">
<img width="300" src="https://github.com/user-attachments/assets/1b6e67e9-ec25-4f1d-963c-0ee6060f13ad" />
</p>

> [!NOTE]
> In this document, vectors are denoted with an arrow (e.g., $\vec{v}$) and are assumed to be row-vectors.

## 1. The Query, Key, and Value Model

Rather than processing a sequence step-by-step, Attention treats the sequence like a searchable database. Every word in the sequence is projected into three distinct vectors:

- **Query ($\vec{q}$)**: The search term. "What am I looking for?"
- **Key ($\vec{k}$)**: The index. "What information do I contain?"
- **Value ($\vec{v}$)**: The content. "What is the actual data to extract?"

### 1.1 Dimensionality ($d_{model}$ and $d_k$)

- **$d_{model}$**: The total embedding size (e.g., 512). The width of the main "highway."
- **$d_k$**: The size of the search space. Generally $d_k = d_{model} / h$ where $h$ is the number of heads.

### 1.2 Concrete Example: "The cat sat"

Imagine we are calculating the output for the word **"cat"** at position $i=2$.

1. **Generation**: We calculate $\vec{q}_{cat}$, $\vec{k}_{the}, \vec{k}_{cat}, \vec{k}_{sat}$, and $\vec{v}_{the}, \vec{v}_{cat}, \vec{v}_{sat}$.
2. **Similarity (Dot Product)**: We check how similar the Query for "cat" is to every Key in the sequence:
   - $Score(cat, the) = \vec{q}_{cat} \cdot \vec{k}_{the}$
   - $Score(cat, cat) = \vec{q}_{cat} \cdot \vec{k}_{cat}$
   - $Score(cat, sat) = \vec{q}_{cat} \cdot \vec{k}_{sat}$
3. **Softmax**: We turn these raw scores into probabilities (weights) that sum to 1.
   - e.g., $Weights = [0.1, 0.7, 0.2]$ (meaning "cat" attends mostly to itself).
4. **Weighted Sum**: The final output for "cat" is the weighted average of the Values:
   $$ \text{Result}_{cat} = 0.1 \cdot \vec{v}_{the} + 0.7 \cdot \vec{v}_{cat} + 0.2 \cdot \vec{v}_{sat} $$

This "Result" vector now contains information about "cat", but also context from "the" and "sat".

---

## 2. Mathematical Structure: $Q K^T$

To perform this for all words in parallel, we pack the vectors into matrices $Q, K, V$.

### 2.1 The Dot Product Matrix
The operation $Q K^T$ computes every possible pairwise similarity in one step:

$$
Q K^T = \begin{pmatrix} \text{---} & \vec{q}_1 & \text{---} \\ \text{---} & \vec{q}_2 & \text{---} \\ & \vdots & \\ \text{---} & \vec{q}_L & \text{---} \end{pmatrix} \begin{pmatrix} \mid & \mid & & \mid \\ \vec{k}_1^T & \vec{k}_2^T & \dots & \vec{k}_L^T \\ \mid & \mid & & \mid \end{pmatrix} = \begin{pmatrix} \vec{q}_1 \cdot \vec{k}_1 & \vec{q}_1 \cdot \vec{k}_2 & \dots \\ \vdots & \vec{q}_2 \cdot \vec{k}_2 & \dots \\ \vec{q}_L \cdot \vec{k}_1 & \dots & \vec{q}_L \cdot \vec{k}_L \end{pmatrix}_{L \times L}
$$

Entry $(i, j)$ in the $L \times L$ matrix is the similarity of word $i$ to word $j$.

---

## 3. The Scaling Factor and Variance

### 3.1 Why Scale? (Softmax Saturation)
The softmax function $\sigma(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$ is extremely sensitive to the magnitude of scores. Large variances cause "peaky" distributions:

| Score $x$ | $e^x$ | Softmax Prob |
| :--- | :--- | :--- |
| 2.0 | 7.4 | 0.0003 |
| **10.0** | **22026.5** | **0.9997** |
| 1.5 | 4.5 | 0.0000 |

**The Problem**: If a probability reaches 1.0 or 0.0, the gradient $p_i(1 - p_i)$ becomes 0. The model stops learning (Vanishing Gradient). Dividing by $\sqrt{d_k}$ keeps the variance at 1, keeping scores in a range where softmax is still "steep."

### 3.2 Rigorous Derivation of Variance $d_k$

#### 3.2.1 Prerequisite: Numerical Range
We use Normalization layers and specific Weight Initialization to ensure inputs ($\vec{q}, \vec{k}$) have **Mean 0** and **Variance 1**. Without this, the scaling factor $\sqrt{d_k}$ would be mathematically meaningless.

#### 3.2.2 The Proof
**Definition**: Variance measures the spread from the mean.
$$ \text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2 $$

Assume $q_i, k_i$ are independent with mean 0 ($E[X]=0$) and variance 1 ($\text{Var}(X)=1$). This implies $E[X^2] = 1$.

For a single dot product $z = \vec{q} \cdot \vec{k} = \sum_{i=1}^{d_k} q_i k_i$:
1. $E[z] = \sum E[q_i] E[k_i] = 0$.
2. $\text{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = (1 \cdot 1) - 0 = 1$.
3. Since variances of independent terms add:
$$ \text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k $$

We divide by $\sqrt{d_k}$ so that $\text{Var}\left(\dfrac{z}{\sqrt{d_k}}\right) = \dfrac{\text{Var}(z)}{(\sqrt{d_k})^2} = \dfrac{d_k}{d_k} = 1$.

---

## 4. Multi-Head Attention (MHA)

MHA uses parallel "heads" to look at different sequence relationships simultaneously.

### 4.1 The Mechanism
1. **Split**: The $d_{model}$ is divided into $h$ heads of size $d_k$ (e.g., $512 \to 8 \times 64$).
2. **Parallel Work**: Each head uses its own learned weights to find relationships. Head 1 might focus on grammar, while Head 2 focuses on semantic meaning.
3. **Concat**: The results are glued back together to get a single vector of size 512.
4. **Project**: A final matrix $W_O$ allows the heads to share information.

---

## 5. Attention Masking

Masking "hides" certain words by forcing their attention weight to zero.

### 5.1 The Softmax Trick ($-\infty \to 0$)
We set raw scores to $-\infty$ before Softmax. Since $e^{-\infty} = 0$, those positions contribute nothing:

$$
\text{softmax}(\begin{pmatrix} 10.5 & -\infty & 2.1 \end{pmatrix}) = \begin{pmatrix} \frac{e^{10.5}}{e^{10.5} + 0 + e^{2.1}} & 0 & \frac{e^{2.1}}{e^{10.5} + 0 + e^{2.1}} \end{pmatrix}
$$

### 5.2 Causal (Look-ahead) Masking
Used to prevent words from "seeing the future." We overwrite the upper triangle of the similarity matrix with $-\infty$:

$$
\text{Scores} = \begin{pmatrix} 2.1 & 1.1 & 0.5 \\ 0.8 & 3.0 & 1.2 \\ 1.0 & 0.5 & 2.5 \end{pmatrix} \xrightarrow{\text{Mask}} \begin{pmatrix} 2.1 & -\infty & -\infty \\ 0.8 & 3.0 & -\infty \\ 1.0 & 0.5 & 2.5 \end{pmatrix}
$$

$$ \text{softmax}(\begin{pmatrix} 2.1 & -\infty & -\infty \end{pmatrix}) = \begin{pmatrix} 1.0 & 0.0 & 0.0 \end{pmatrix} $$
Word 1 can now only attend to itself, preserving the causal order.
