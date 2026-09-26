# Host-routed MoE runtime semantics

These guarantees apply to the existing host-routed executor after an expert
submodel has been prepared. They do not expand the graph-matching contract,
change expert unrolling, or alter LLM configuration defaults.

- Mixing scores are coefficients, not a second routing decision. Every finite
  nonzero score is retained, including negative and very small values. NaN and
  infinity are rejected. An epsilon cutoff can change the result when an expert
  output is large.
- Single-token execution uses the existing fixed-K batched executable. If fewer
  than K coefficients are nonzero, unused slots reuse an active expert's weights
  with a zero score. The request cache is keyed by that padded weight selection;
  scores and input/output tensors are rebound on every invocation.
- An all-zero selection produces zero output without expert inference. Prefill
  leaves its output accumulator cleared.
- If prefill validation or inference fails, both request slots for every chunk
  size are drained before reuse. Cleanup does not replace the original exception
  or scatter partial results.
- Expert weight slices require a valid leading expert dimension and in-range
  expert index. Four-bit expert boundaries must be byte-aligned.

## Regression coverage

`moe_executor_test.cpp` tests the real executor using prepared, evaluatable expert
submodels and distinct expert weights. It covers float32/float16 scores, decode
and chunked prefill, enabled/disabled decode request caching, changing bindings
and selections, zero/negative/tiny coefficients, deferred failures and reuse,
and packed-weight slice validation.

These are native runtime tests, not physical-NPU accuracy or performance results.
Generic topology matching and LLM policy integration are tested separately.