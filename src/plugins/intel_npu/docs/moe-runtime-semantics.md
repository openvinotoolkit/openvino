# Host-routed MoE runtime semantics

## Design overview

A mixture-of-experts (MoE) layer combines expert outputs using mixing scores
produced by the model's router. There are E available experts and K selected
slots per token. In NPUW's `HOST_ROUTED` path, the host consumes those scores and
dispatches the required expert work to compiled submodels:

1. The model computes routing scores and hidden-state inputs. The router itself
   need not execute on the host.
2. The host identifies nonzero expert contributions and binds the corresponding
   weights, scores and inputs to prepared expert requests.
3. The compiled expert submodels execute on the configured device, such as the NPU.
4. Their score-weighted outputs are combined into the MoE layer's result.

Host routing describes the dispatch mechanism, not the device executing the
expert arithmetic. In particular, **batched expert execution on the NPU is still
host-routed**. In the separate `DEVICE_ROUTED` path, expert selection is performed
inside the compiled graph rather than by this host executor.

### Single-token decode

The host uses `EXPERT_BATCH`: one prepared fixed-K executable contains K expert
slots. It binds the selected experts' weights and mixing scores, then executes
all slots in one inference request. Requests can be cached by their bound expert
selection; changing scores and input/output tensors must still be rebound.
This batching is part of `HOST_ROUTED`, not `DEVICE_ROUTED`.

### Multi-token prefill

The host uses `EXPERT_ITERATIVE`: tokens are grouped by expert and split into
chunks supported by the prepared executables. Double-buffered asynchronous
requests overlap host preparation with device execution. Score-weighted chunk
outputs are placed in an accumulator and combined by the downstream reduction.

The guarantees below apply after expert submodels have been prepared. They do
not expand graph matching, change expert unrolling, or alter LLM configuration
defaults.

## Runtime guarantees

- Mixing scores are coefficients, not a second routing decision. Every finite
  nonzero score is retained, including negative and very small values. An epsilon
  cutoff can change the result when an expert output is large.
- **Any NaN or infinity is an error**, including an entirely NaN routing tensor.
  Both modes validate scores before treating them as zero or nonzero. Invalid
  scores are not converted to zeros, so backend routing failures remain visible.
- If fewer than K coefficients are nonzero during decode, unused slots reuse an
  active expert's weights with a zero score. The request cache is keyed by that
  padded weight selection; scores and input/output tensors are rebound each time.
- An entirely finite, exactly-zero mixture produces zero expert contribution
  without expert inference. Prefill leaves its accumulator cleared. Both modes
  emit an NPUW warning for the affected subgraph when warning logging is enabled,
  so an unexpected all-zero router result can be investigated. This is a
  coefficient-handling guarantee, not a claim that all-zero routing is typical
  of trained models. It does not apply to nonfinite scores.
- If prefill validation or inference fails, both request slots for every chunk
  size are drained before reuse. Cleanup does not replace the original exception
  or scatter partial results.
- Expert weight slices require a valid leading expert dimension and in-range
  expert index. Four-bit expert boundaries must be byte-aligned.

## LLM stage configuration

LLM compilation detects compatible batched-MoE graphs structurally and retains
router/expert name hints for legacy exports. Host-routed stages preserve the
`expert` partition tag even for small models, without replacing existing tags.
The existing shape-compute folding runs after stage shapes are prepared.

Without an explicit `NPUW_LLM_GENERATE_MOE_HINT`, automatic device routing requires
NPU architecture `5010`, compiler version 7.29 or later, single-token generation,
and a supported batched-MoE boundary in every prepared generate variant. Every
candidate boundary must pass the shared matcher and device-eligibility checks;
an unsupported expert body prevents automatic device routing. A name hint alone
is insufficient. Other cases retain the host-routed default.

Explicit routing hints are preserved. Device routing remains unsupported for
prefill, and explicit device decode must satisfy the core lowering requirements.
`DENSE` remains restricted to CPU execution.