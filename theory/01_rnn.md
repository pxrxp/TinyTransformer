# Recurrent Neural Networks (RNN)

Recurrent Neural Networks are designed to process sequential data by maintaining a hidden state $h_t$ that captures information from previous time steps.

## The Basic RNN Cell

The hidden state at time $t$ is computed using the previous hidden state $h_{t-1}$ and current input $x_t$:

$$h_t = \sigma(W_{ih}x_t + W_{hh}h_{t-1} + b)$$

Where:
- $h_t$ is a vector of size $d_h$.
- $W_{hh}$ is a $d_h \times d_h$ weight matrix.
- $\sigma$ is an element-wise activation function (like $\tanh$).

## Why learn this?

Understanding the calculus behind RNNs is the key to understanding **why Transformers replaced them**. 

The "Attention is All You Need" paper was a direct response to the fundamental mathematical flaws of RNNs. By seeing exactly where the gradient "dies" in an RNN, you'll understand why LSTMs added "cell states" and why Transformers eventually abandoned recurrence entirely for parallel attention.

## The Gradient Problem

To train the RNN, we need to know how the loss $L$ changes with respect to our weights. By the chain rule, this depends on how $h_t$ changes with respect to $h_{t-1}$, $h_{t-2}$, and so on.

### 1. The Jacobian Matrix

When we derive a vector $h_t$ with respect to another vector $h_{t-1}$, we get a matrix of every possible partial derivative:

$$J = \frac{\partial h_t}{\partial h_{t-1}} = \begin{bmatrix} 
\frac{\partial h_{t,1}}{\partial h_{t-1,1}} & \cdots & \frac{\partial h_{t,1}}{\partial h_{t-1,n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial h_{t,n}}{\partial h_{t-1,1}} & \cdots & \frac{\partial h_{t,n}}{\partial h_{t-1,n}}
\end{bmatrix}$$

### 2. Deriving the Jacobian

Let's look at the equation $h_t = \sigma(z_t)$ where $z_t = W_{ih}x_t + W_{hh}h_{t-1} + b$.

By the chain rule, $\frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_{t-1}}$.

#### Why the Diagonal?
The activation $\sigma$ is **element-wise**. This means the $i$-th element of $h_t$ only depends on the $i$-th element of $z_t$:
- $\frac{\partial h_{t,1}}{\partial z_{t,1}} = \sigma'(z_{t,1})$
- $\frac{\partial h_{t,1}}{\partial z_{t,2}} = 0$ (because change in $z_2$ doesn't affect $h_1$)

When you put these into a matrix, all the off-diagonal terms are zero. Only the diagonal contains the values $\sigma'(z_{t,i})$. We write this as $\text{diag}(\sigma'(z_t))$.

#### The Linear Part
The derivative of the linear term $W_{hh}h_{t-1}$ with respect to $h_{t-1}$ is simply $W_{hh}^T$.

#### The Result
Combining these, we get:
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\sigma'(z_t)) W_{hh}^T$$

### 3. The Multiplicative Chain

For a sequence of $T$ steps, the gradient of the loss $L$ with respect to the initial state $h_0$ involves multiplying these Jacobians together:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^T \left( \text{diag}(\sigma'(z_t)) W_{hh}^T \right)$$

If we repeatedly multiply by the same matrix $W_{hh}^T$:
- If the weights in $W_{hh}$ are small, the values in the product will shrink toward zero exponentially (**Vanishing Gradient**).
- If the weights are large, the values will grow exponentially (**Exploding Gradient**).

## The Solution: Additive Updates

The core issue is that the gradient is forced through a **multiplicative chain**. If any link in the chain is small, the whole gradient vanishes.

LSTMs solve this by introducing a **cell state** $C_t$ that is updated using **addition**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

### The Hadamard Product ($\odot$)

The symbol $\odot$ denotes the **Hadamard product**, which is just **element-wise multiplication**.
Example: $\begin{bmatrix} a \\ b \end{bmatrix} \odot \begin{bmatrix} c \\ d \end{bmatrix} = \begin{bmatrix} ac \\ bd \end{bmatrix}$.

In LSTMs, this lets the "forget gate" $f_t$ decide which parts of the memory to keep. Because the update to $C_t$ is additive, the gradient can flow through the sum without being repeatedly scaled by the weight matrix $W_{hh}$. This allows the gradient to persist over long distances, a mechanism often called the **constant error carousel**.
