# Gated Recurrent Unit (GRU)

The GRU is a simplified variation of the LSTM that merges the cell state and hidden state into a single vector.

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/68a3ea9e-021e-4a80-8a38-83c86830e9ce" />
</p>

## 1. State Simplification

Unlike the LSTM, which maintains a separate long-term cell state ($C_t$), the GRU only has a single **hidden state** $h_t \in \mathbb{R}^{d_h}$.

The model uses gating to achieve the same additive update property, but with fewer operations.

## 2. Gate Signals and Dimensions

The GRU computes two gates and one candidate hidden state, each of size $d_h$:
- Input vector $x_t \in \mathbb{R}^{d_{in}}$
- Previous hidden state $h_{t-1} \in \mathbb{R}^{d_h}$

To produce signals of size $d_h$, the weights must have the following dimensions:
- $W_{i} \in \mathbb{R}^{d_{in} \times d_h}$
- $W_{h} \in \mathbb{R}^{d_h \times d_h}$
- $b \in \mathbb{R}^{d_h}$

## 3. Mathematical Formulas

### Gate Calculations
1. **Update Gate ($z_t$)**: Controls how much of the previous state is kept vs. how much new info is added.

$$z_t = \sigma(x_t W_{iz} + h_{t-1} W_{hz} + b_z)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/badb659d-2965-4014-833d-126e563c4e9a" />
</p>

2. **Reset Gate ($r_t$)**: Controls how much of the previous state to "forget" when calculating the new candidate.

$$r_t = \sigma(x_t W_{ir} + h_{t-1} W_{hr} + b_r)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/dde817a6-e275-46d9-9f80-95ed340d15e6" />
</p>

### State Updates
3. **Candidate Hidden State ($\tilde{h}_t$)**:

$$\tilde{h}_t = \tanh(x_t W_{ih} + (r_t \odot h_{t-1}) W_{hh} + b_h)$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/242a1647-31b6-486b-9823-1f525db0b2f9" />
</p>

4. **Final Hidden State ($h_t$)**:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/da9eddc5-8c27-43ab-981b-1637e5bc3a0d" />
</p>

The formula for $h_t$ is a linear interpolation. If $z_t = 0$, the state is preserved; if $z_t = 1$, it is replaced by the new candidate.

## 4. Implementation Optimization

Similar to the LSTM, we can compute the $z_t$ and $r_t$ gates using a single concatenated multiplication. We define a weight matrix of size $(d_{in} + d_h) \times 2d_h$.

However, the candidate state $\tilde{h}_t$ depends on the result of the reset gate ($r_t \odot h_{t-1}$), so the hidden-to-hidden transformation $W_{hh}$ for the candidate must be calculated separately after the reset operation.