Input hidden states X
        │
        ├─────────────────────────────── residual path
        │
        ▼
Input RMSNorm (across 5120 features per token)
        │
        ▼
Q, K, V projections
        │
        ├── Q RMSNorm (across 128 features per token/head)
        ├── K RMSNorm (across 128 features per token/head)
        └── V is not normalized
        │
        ▼
Apply RoPE to Q and K
        │
        ▼
Attention: softmax(QKᵀ / √128)V
        │
        ▼
Concatenate 40 heads
        │
        ▼
o_proj: 5120 → 5120
        │
        ▼
Add attention residual
X₁ = X + attention_output
        │
        ├─────────────────────────────── residual path
        │
        ▼
Post-attention RMSNorm
        │
        ├──→ up_proj:   5120 → 17408 ───────────┐
        │                                        │
        └──→ gate_proj: 5120 → 17408 → SiLU ────┤
                                                 ▼
                                    Elementwise multiplication
                                    SiLU(gate) ⊙ up
                                                 │
                                                 ▼
                                    down_proj: 17408 → 5120
                                                 │
                                                 ▼
                                    Add MLP residual
                                    X₂ = X₁ + MLP_output
                                                 │
                                                 ▼
                                    Input to the next layer