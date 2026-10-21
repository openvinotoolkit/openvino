# WWB model validation

Use WhoWhatBench (WWB) for generated-text quality when enabling a real GGUF decoder or investigating quantized output drift. Keep operation/layout, floating graph, state/cache and API checks: WWB complements them and does not validate media/projector or speech-output contracts. Quantized token/logit differences alone do not block support when graph/integration checks pass and WWB shows no material additional loss versus llama.cpp.

## Reference and comparison

Generate reference answers once per model from original prequantization FP16/F32/BF16 weights. A publisher floating GGUF or a pinned HF export is suitable; converting original BF16 to FP16 is allowed. An FP16/F32 expansion of Q4/Q8 values is only an arithmetic diagnostic and cannot serve as this quality reference. Record ancestry, immutable revisions, file SHA256s and tensor-type inventories, and verify model architecture, tensor names/shapes and tokenizer tables agree.

Score native llama.cpp using the requested quantized GGUF and GenAI using that exact same file against the same saved reference CSV. Prefer WWB's standard semantic similarity (`sentence-transformers/all-mpnet-base-v2`); pin the encoder revision and WWB source. Its mean cosine score measures agreement with reference answers, not percent correctness. Built-in KL may reject GenAI; preflight actual version/backend support before proposing that metric.

Choose a representative prompt set covering the model's intended language/tasks and enough generated tokens to expose drift. Record prompt provenance/count and maximum tokens; use a small one-prompt generation/scoring preflight before a long batch. Do not infer broad quality from a short replay. Greedy runs must disable repetition, frequency and presence penalties in both engines. Match chat rendering, special-token policy, EOG IDs, context, threads, attention and KV precision, and record backend arithmetic differences. On CPU use F32 inference hint, dynamic quantization disabled, and reference KV precision. For Q4_K comparisons set `OV_GGUF_Q4_K_ZP_F16=1` before the process; keep it unset for the unfiltered unit suite.

## Inputs and execution

Audit identical prompt token IDs and stopping behavior before comparing decoders. Render templates once or prove both runtimes render the same bytes. With differing tokenization/EOG configuration, retain the strict failure and generate a clearly labeled matched-input comparison using encoded reference tokens/configuration. Also measure the supported default string-input path for affected prompts; a forced-input result alone cannot establish tokenizer/API parity. Never silently change the user's default path.

Use WWB's model adapters where compatible. A native helper built against the pinned llama.cpp header/library is suitable when a Python wrapper lacks architecture support. Generate independently per prompt, with a fresh state and the same token limit, and save CSVs with `prompts,answers`. Enforce equal row counts, prompt strings and order before scoring: WWB may zip and truncate unequal inputs.

Example with compatible WWB CLI and precomputed answers:

```bash
wwb --model-type text --gt-data reference.csv --target-data llama_quant.csv \
    --metrics similarity --data-encoder sentence-transformers/all-mpnet-base-v2 \
    --output llama_score
wwb --model-type text --gt-data reference.csv --target-data genai_quant.csv \
    --metrics similarity --data-encoder sentence-transformers/all-mpnet-base-v2 \
    --output genai_score
```

If eager imports require unrelated modalities, use an isolated scoring environment and call the actual pinned WWB `TextSimilarity.evaluate` on audited CSVs. Keep metric implementation unchanged, record that entrypoint, and verify identical-answer scoring near one before real measurements. Pin/cache the encoder for reproducibility; avoid upgrading shared runtime dependencies.

Keep inputs stable and delegate prescribed download/generation/scoring batches according to local delegation instructions. Workers receive a reusable runner, pinned paths/hashes/settings, output paths and coverage/exit criteria. They return compact counts, means/minima, grouped failures and artifact paths; the main agent reviews low-score answers and accepts scope. Use fresh output paths after failures or plan-only attempts. Reuse saved reference and unchanged rows only with matching model, prompt, settings and runtime evidence; retain exact regeneration/reuse counts. Selective default-input reuse also needs proof that string and encoded overloads reach the same graph and changed stop IDs could not affect recorded generation.

## Support decision and report

Compare each backend's mean/per-prompt score against the same floating reference and report the GenAI-minus-llama.cpp gap. Review lowest-score answers and largest gaps for incoherence, repetition, task failure or truncation. Do not invent a universal threshold or require identical free-running tokens; make material degradation and required sample coverage explicit for the task. Similar scores plus coherent reviewed outputs establish support within the measured checkpoint/device/modality/configuration. Keep unresolved crashes, bad floating graph behavior, unsupported API contracts and material extra quality loss as issues.

Preserve failed numerical/tokenizer/EOG results as diagnostics; do not relabel them as parity passes. A text WWB pass does not establish image/audio/TTS, long-context or GPU quality. Publish prompt count/token limit, original reference lineage, settings, both scores, gap and limitations in the support documentation/PR; retain all answers, token histories, per-prompt scores, commands and runtime/model hashes on disk.
