# Embedding Quality Evaluation Report

## Overview
This report summarizes the evaluation of the "Embedding Gemma" model after applying Composite TSSN incision. The goal was to verify that the sparse model retains the semantic embedding capabilities of the dense baseline.

## Methodology
- **Model**: Embedding Gemma (Gemma 3 architecture, Encoder-only).
- **Baseline**: Dense PyTorch model (`embeddinggemma_local`).
- **Sparse Model**: OpenVINO IR with TSSN layers injected (`model_ir_pruned/gemma_embedding_pruned.xml`).
- **Incision Target**: `__module.layers.12.mlp.down_proj/aten::linear/MatMul` (Mid-layer FFN Down Projection).
- **Sparsity**: 70% (0.7).
- **Metric**: Cosine Similarity of Mean-Pooled Sentence Embeddings.

## Results

| Sentence | Cosine Similarity |
| :--- | :--- |
| "The future of artificial intelligence is sparse." | 0.9882 |
| "OpenVINO optimizes deep learning models for inference." | 0.9741 |
| "The quick brown fox jumps over the lazy dog." | 0.9904 |
| "Semantic search requires high quality vector embeddings." | 0.9941 |
| "Photosynthesis is the process by which plants use sunlight to synthesize foods." | 0.9773 |

**Average Semantic Preservation: 0.9848**

## Conclusion
The sparse TSSN model demonstrates **excellent semantic preservation** (98.5% similarity to dense baseline). The incision process successfully reduced the model complexity (via sparsity) without degrading its ability to generate high-quality vector embeddings.

## Fixes Implemented
1.  **Model Conversion**: Fixed the OpenVINO conversion to export the `Gemma3Model` (base) instead of `Gemma3ForCausalLM`, ensuring the output is a 768-dimensional vector space rather than a 262k-dimensional logit space.
2.  **Incision Tool**: Updated `apply_incision_tool.cpp` to accept command-line arguments for target layer and sparsity, allowing for precise targeting.
