# Transformer Attention Implementations

This repository contains two implementations related to Transformer models:

1. Scaled Dot-Product Attention (NumPy)
2. Simple Transformer Encoder Block (PyTorch)

---

## 1. Scaled Dot-Product Attention (NumPy)

File: scaled_attention.py

This script:
- Computes attention scores using QK^T
- Scales scores by sqrt(d_k)
- Applies softmax
- Computes the context vector using attention weights and V

Run:

python3 scaled_attention.py

---

## 2. Simple Transformer Encoder Block (PyTorch)

File: simple_transformer_encoder.py

This encoder block includes:
- Multi-head self-attention
- Feed-forward network (two linear layers with ReLU)
- Residual connections
- Layer normalization

Tested on:
Batch size: 32  
Sequence length: 10  
Embedding dimension: 64

Expected output shapes:
Input: [32, 10, 64]  
Output: [32, 10, 64]  
Attention weights: [32, 10, 10]

Run:

python3 simple_transformer_encoder.py

---

## Folder Structure

scaled_attention.py  
simple_transformer_encoder.py  
README.md

---

## Purpose

These scripts help understand:
- How attention is calculated
- How a transformer encoder processes sequences
- The role of normalization and residual connections
