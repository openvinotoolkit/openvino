# Roadmap to "iGemma" (Infected Gemma)

## Phase 1: The "Patient Zero" (Proof of Concept)
**Goal**: Create a single functioning Gemma model with *one* infected layer that runs without crashing.
1.  **Incision**: Use `inject_tssn_layer.py` to replace a specific FFN layer in Gemma with `CompositeTSSN`.
2.  **Verification**: Load the model with the extension and run a simple inference.
3.  **Status**: *Ready to start.* We have the tools (`inject_tssn_layer.py`, `benchmark_gemma_performance.py`).

## Phase 2: The "Fever" (Evolutionary Rehabilitation)
**Goal**: Restore the intelligence of the lobotomized model.
1.  **Problem**: The "infected" layer is initialized with simple magnitude pruning + sign quantization. It will degrade perplexity significantly.
2.  **Solution**: Run `evolutionary_cycle.py` (Metabolic War) on the infected layer(s).
    -   **Input**: Calibration data (e.g., Wikitext).
    -   **Process**: Evolve the sparse topology (synapses) and sensitivity values to minimize reconstruction error (MSE) against the original dense layer output.
3.  **Status**: *Needs adaptation.* `evolutionary_cycle.py` currently runs on synthetic data. We need to feed it real activations from the Gemma model.

## Phase 3: The "Outbreak" (Full Infection)
**Goal**: Infect all FFN layers (or a strategic subset) to maximize efficiency.
1.  **Scaling**: Apply the incision and evolution process to all 18-24 layers of Gemma.
2.  **Pipeline**: Automate this so it's not manual labor. `python infect_all.py --model gemma-2b`.

## Phase 4: "iGemma" Chat (The Interface)
**Goal**: A usable chat interface.
1.  **Backend**: A Python script using `ov.Core` + `CompositeTSSN` extension to generate tokens.
2.  **Frontend**: A simple CLI or Streamlit app.
3.  **Omni-Capabilities**:
    -   **Vision**: Infect **PaliGemma** (Google's VLM). The FFNs in the language decoder are the same.
    -   **Audio**: Infect a speech model (e.g., Whisper encoder or a multimodal LLM).

## Immediate Next Steps
1.  **Run `inject_tssn_layer.py`** on the downloaded Gemma IR to create `gemma_ir_tssn`.
2.  **Test Inference**: Run `benchmark_gemma_performance.py` on the infected model.
3.  **Build the "Rehab" Loop**: Create a script that:
    -   Runs the original model on text to get "Gold" activations for Layer X.
    -   Runs the infected Layer X.
    -   Evolves Layer X to match Gold.
