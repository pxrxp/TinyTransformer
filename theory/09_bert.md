# Bidirectional Encoder Representations from Transformers (BERT)

While GPT threw away the Encoder to focus on sequence generation, **BERT threw away the Decoder.**

BERT is an **Encoder-Only** model. It does not generate continuous text. Instead, it acts as the ultimate "Reading Comprehension" engine.

## 1. Nomenclature

*   **Bidirectional**: While GPT is strictly forward-facing (autoregressive), BERT looks left AND right simultaneously. When processing a word, it uses the entire surrounding sentence to deduce its meaning.
*   **Encoder**: It physically relies exclusively on the Encoder stack of the Transformer architecture.
*   **Representations**: It does not output human-readable text. It outputs massive arrays of dense mathematical vectors ("embeddings"). These representations perfectly capture the exact contextual meaning of the input text, which you can then feed into simple classification heads.
*   **Transformers**: It uses the standard Multi-Head Attention blocks.

## 2. Bidirectional Context

In autoregressive decoders like GPT, predicting the word "bank" using the prompt "I walked to the bank" is strictly forward-facing. The word "bank" has no idea if the next word is going to be "to deposit cash" or "to go fishing".

BERT entirely removes the Lower-Triangular Causal Mask. Every token is mathematically allowed to self-attend to every other token ($Q K^T$) forward and backward simultaneously. 
`"I walked to the bank"` $\leftarrow$ `bank` $\rightarrow$ `"to go fishing"`. 
The embedding for `bank` becomes heavily contaminated with the concept of water, definitively contextualizing it.

## 3. Training Objectives

You cannot train a bidirectional model on Next-Token Prediction like you do with GPT. If you remove the Causal Mask, the word at position $t$ will literally just look at the explicit answer written at position $t+1$, copy it, and learn absolutely zero grammar.

BERT researchers inverted the training paradigm using two novel approaches simultaneously:

### 3.1 Masked Language Modeling (MLM)
Instead of predicting the *next* word, BERT plays Fill-In-The-Blank. 
1. Feed the system a completely visible, whole sentence.
2. Randomly replace 15% of the words with a literal `<MASK>` token.
   - `"The man went to the <MASK> to buy milk."`
3. Because the model is bidirectional, the `<MASK>` token looks at "went to the" AND "to buy milk".
4. We force the model's MLM Linear Head to guess what the `<MASK>` token originally was using that surrounding context.

### 3.2 Next Sentence Prediction (NSP)
To force the model to understand macro-level logical flow across paragraphs:
1. Feed the system two distinct sentences separated by a `<SEP>` token.
2. 50% of the time, Sentence B logically belongs right after Sentence A. 50% of the time, it is totally random.
3. A special classification token `<CLS>` is always planted at index 0. Because everything attends to everything bidirectionally, the `<CLS>` embedding acts as a contextual sponge, soaking up the global relationship between the two sentences.
4. Pass the final `<CLS>` output vector into a binary classifier to predict: `Is_Next` or `Not_Next`.
