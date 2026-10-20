# Nemotron quantization references

`iq2_xxs_{qbytes,deq}.npy` and `nvfp4_{qbytes,deq}.npy` contain deterministic block bytes and the exact F32 results from real ggml CPU at llama.cpp `23f7aabc16c72f6f707dba227251c35d7bd1ed96`. Regenerate with `nemotron_quant_oracle.cpp` linked against that checkout's libggml-base, then convert the emitted `.bin` files to NumPy arrays (uint8 bytes, float32 reference). Each file covers four 256-value rows. Raw weight and native parser tests compare exactly, including rank-three expert layouts and F32 scales.
