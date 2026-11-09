# Changelog

## [v0.2.0] - 2026-11-09

### Added
- **Vision-Language Model (`VisionLanguageModel`)**: Dual-encoder architecture aligning multi-scale spatial & HOG visual features with hashed character n-gram and word-level text representations in a shared latent space.
- **Zero-Shot Classification & Cross-Modal Search**: Classify images via natural language prompts (`zero_shot_classify`) and retrieve images from text queries (`search_images`, `search_texts`).
- **Transformer & Modern Neural Primitives**: Added `MultiHeadAttention`, `LayerNorm`, `GELU`, and `Dropout` with analytical forward and backward passes in pure NumPy.
- **Contrastive Learning (`InfoNCELoss`)**: Symmetric cross-entropy contrastive loss with multi-positive label support and exact L2-normalization gradients.
- **Optimizers**: Added `AdamW` with decoupled weight decay and `clip_grad_norm`.
- **CLI Commands**: Added `minimind vlm-train` and `minimind zero-shot`.

### Improved
- Streamlined codebase across all modules (`network`, `knowledge`, `parsers`, `data`, `viz`) with zero-pickle `.npz` model serialization.
- Expanded test suite to 33 automated tests including finite-difference gradient checks for all neural and transformer layers.
