# Build GPT2 From Scratch

This repository contains a from-scratch implementation of GPT, built step by step to deeply understand how modern language models like GPT-2 work under the hood.
The project is intentionally structured so that each component is implemented progressively, making it easy to follow the evolution of the model and training pipeline.

# Goal
  - Rebuild GPT-style transformer from scratch
  - Understand every piece of the system:
  - Tokenization & embeddings
  - Self-attention (multi-head)
  - Transformer blocks
  - Optimization & training dynamics
The focus is not just using models, but understanding how they actually work

# Important Note

  - This is a base language model, not a chat model.
  - Trained to predict next tokens
  - Learns patterns from text
  - Generates raw completions
It is not fine-tuned for conversation (ChatGPT-style)

# Example Output
After training, prompting with:

    - Hello, I'm a language model,
produces outputs like:

    - Hello, I'm a language model, and my goal is to understand patterns in text data.
    - Hello, I'm a language model, designed to predict the next word in a sequence.
    - Hello, I'm a language model, and I generate text based on learned distribution.
