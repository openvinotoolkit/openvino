# LFM2.5 speech references

`lfm2_speech.npz` contains a two-layer random F32 depthformer with unequal Q/KV
heads, eight codebooks, full per-book logits, summed language feedback and averaged
codec features. `lfm2_detokenizer.npz` contains a mixed convolution/attention LFM2
detokenizer with a 1282-channel spectral head and a 30-token sliding window.
The test executes its GenAI-adapted input-embeddings path at 18/37/18 frames.

Regenerate using `gen_lfm2_speech_accuracy.py --liquid-audio <checkout>`, with
`PYTHONPATH=<llama.cpp>/gguf-py`, NumPy, PyTorch and Transformers installed. The
LiquidAI checkout must be revision `19e65845923a7f136442c95137884ec61eb386aa`.
The generator runs the publisher transformer and Transformers Lfm2Model; OpenVINO
does not generate expected outputs. Python 3.12 type syntax is adapted for Python 3.10.
These are represented F32 fixtures, separate from quantized checkpoint comparisons.
