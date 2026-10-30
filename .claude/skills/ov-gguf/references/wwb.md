# Generated-text quality with WWB

Use WhoWhatBench (WWB) when generated-text quality or quantization loss is part of the task.
Keep numerical parity, operation/layout, state/cache and API acceptance separate. Text quality
scores do not validate projector, media or speech-output contracts and do not waive an existing
numerical gate.

## Reference and comparison

For quantization-quality evaluation, use the same saved reference answers from original
prequantization floating weights for both backends. A pinned publisher floating GGUF or an HF
export can supply that reference. An expansion of quantized values isolates arithmetic and
cannot measure loss against original weights. Record reference ancestry, immutable revisions,
file hashes, conversion options and tensor-type inventories; verify architecture, tensor shapes
and tokenizer tables agree.

Generate native llama.cpp and GenAI answers using the exact same requested quantized file and
score them against that saved reference. Pin WWB, the metric and any scoring encoder revision.
Check the installed CLI/backend capabilities before a batch. Semantic similarity measures
agreement with reference answers, not percentage correctness; choose metrics and material-loss
criteria appropriate to the task before inspecting results.

Use representative prompts for the requested languages/tasks and enough generation to expose
drift. Record prompt provenance/count and token limits. Preflight one prompt through generation
and scoring; verify identical-answer scoring reaches the metric's expected maximum. A short
teacher-forced replay does not establish free-running quality.

## Inputs and execution

Audit prompt token IDs and stopping behavior first. Match rendered template bytes, special-token
policy, EOG IDs, context, sampling/penalties, attention and KV precision, and record arithmetic
settings independently. For strict CPU comparisons use the established F32 inference and
disabled dynamic-quantization settings. Set checkpoint-specific zero-point modes before launch;
keep them unset for the unfiltered unit suite.

If default string-input paths differ, retain that failure and label any comparison using matched
encoded inputs. Also test the supported default path for affected prompts; forced inputs alone
do not establish tokenizer or API parity. Start each independent prompt with fresh state.

Save reference and backend answers as CSVs with `prompts,answers`; verify equal row counts,
prompt strings and row order before scoring. Use a compatible pinned WWB adapter or a native
helper built against the pinned backend headers. If scoring dependencies differ from inference,
use an isolated environment and record the actual metric entrypoint without changing it.

Keep fixtures and runtime stable. Reuse saved answers only when model, prompt, generation and
runtime evidence match; record reused and regenerated counts. An input-path or stop-policy
change also requires evidence that saved answers remain valid. Keep separate attempt outputs.

## Report

Report both backend scores against the same reference, their gap, and per-prompt results.
Inspect low-score answers and large gaps for incoherence, repetition, task failure or truncation.
State the tested checkpoint, device, quantization, configuration, prompt count and token limit.
Retain answers, token histories, scoring inputs, commands and runtime/model hashes.

Report numerical parity, quality and integration outcomes independently. Preserve failures and
missing coverage; similar text scores do not establish image/audio, long-context or another
device's behavior. Apply the task's agreed acceptance criteria rather than inventing a universal
threshold or requiring identical free-running tokens.
