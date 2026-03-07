# Long Short-Term Memory (LSTM)

The LSTM improves upon the basic RNN by separating hidden state into two components to solve the vanishing gradient problem.

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/9e090cb2-d65c-448f-a0c2-db10e72274e7" />
</p>

## 1. Naming: Why LSTM?

- **Short-Term Memory**: Represents the hidden state $h_t$. In a vanilla RNN, this is the only state, and it is overwritten or "squashed" at every step, making it short-lived.
- **Long-Term Memory**: Represents the cell state $C_t$. Because it is updated using addition (the constant error carousel), it can carry information across many time steps.

Combining these two state types gives the **Long Short-Term Memory**.

## 2. Linear Algebra Context

Let $d_{in}$ be the input dimensionality and $d_h$ be the hidden dimensionality.

- $x_t \in \mathbb{R}^{d_{in}}$
- $h_{t-1} \in \mathbb{R}^{d_h}$

The model processes these by concatenating them side-by-side:
$$v_t = [h_{t-1} \mid x_t] \in \mathbb{R}^{d_h + d_{in}}$$

## 3. The Four Gate Signals ($i, f, \tilde{C}, o$)

The LSTM computes four "control signals" of size $d_h$. They match the dimension of the hidden state because they perform element-wise operations on the memory vectors.

Each signal follows the general formula:
$$Signal = \text{Activation}(x_t W_i + h_{t-1} W_h + b)$$

### Explicit Gate Formulas
1. **Input Gate ($i_t$)**: Controls what new info enters the memory.

$$i_t = \sigma(x_t W_{ii} + h_{t-1} W_{hi} + b_i)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/0acfd7ea-e15a-439f-9518-71c695dd2b94" />
</p>

2. **Forget Gate ($f_t$)**: Controls what to delete from old memory.

$$f_t = \sigma(x_t W_{if} + h_{t-1} W_{hf} + b_f)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/a2b76fcb-c4e3-43d8-805f-f477a8286677" />
</p>

3. **Candidate Gate ($\tilde{C}_t$)**: The new potential memory content.

$$\tilde{C}_t = \tanh(x_t W_{ig} + h_{t-1} W_{hg} + b_g)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/21316c52-680f-4eab-b7cb-51173a85e289" />
</p>

4. **Output Gate ($o_t$)**: Controls what to reveal as the hidden state.

$$o_t = \sigma(x_t W_{io} + h_{t-1} W_{ho} + b_o)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/cd06ed46-f5bf-4549-b634-f17d8e43ab97" />
</p>

Weight dimensions: $W_{i} \in \mathbb{R}^{d_{in} \times d_h}$, $W_{h} \in \mathbb{R}^{d_h \times d_h}$, $b \in \mathbb{R}^{d_h}$.

## 4. Weight Stacking (Why $4 \times d_h$)

To optimize execution, we combine the eight separate weight matrices into two global matrices:

$$W_{ih} = [W_{ii}, W_{if}, W_{ig}, W_{io}] \in \mathbb{R}^{d_{in} \times 4d_h}$$
$$W_{hh} = [W_{hi}, W_{hf}, W_{hg}, W_{ho}] \in \mathbb{R}^{d_h \times 4d_h}$$

The full computation becomes:
$$Z = x_t W_{ih} + h_{t-1} W_{hh} + b$$
Where $Z \in \mathbb{R}^{4d_h}$. The code then uses `chunk(4, dim=1)` to split $Z$ back into four $d_h$-sized vectors before applying the activations.

## 5. State Update Equations

The cell state $C_t$ is updated via the additive update rule:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/0503dcd3-4f1e-44db-b0ef-52e5d4d0ef8b" />
</p>

The hidden state $h_t$ is produced by filtering the squashed cell state:

$$h_t = o_t \odot \tanh(C_t)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/15350043-6d2b-43ff-8559-2157308e387b" />
</p>