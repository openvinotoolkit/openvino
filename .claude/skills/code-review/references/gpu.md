# Intel GPU Plugin Review

Apply in addition to [Plugins](plugins.md). Severity, evidence threshold, and
output format follow the main [code review skill](../SKILL.md). For design
background, consult `src/plugins/intel_gpu/docs/` (memory formats, kernel
synchronization, dual-runtime dispatch, graph optimization passes, dynamic
shapes).

## Scope the change

- Classify the affected layers: graph transformation, program building,
  primitive/implementation selection, kernel selector, OpenCL kernel, oneDNN
  integration, OpenCL or Level Zero runtime, memory/USM, model cache, dynamic
  shapes, layout/padding, precision/quantization, LLM/SDPA/KV cache, device
  capability detection.
- State the execution scope the diff establishes (GPU generation or
  capability, XMX vs non-XMX, iGPU/dGPU, OpenCL/Level Zero, static/dynamic
  shapes, precisions, layouts). Do not claim broader impact than the evidence
  shows; mark the rest unknown.

## Trace multi-variant paths end to end

For kernel, dispatch, precision, or selection changes, follow the condition
through every stage:

host-side selection/validation -> primitive/descriptor state -> JIT constants
-> kernel argument construction -> every compiled kernel variant (e.g.
prefill/decode, paged/non-paged, tail/non-tail) -> fallback or rejection path
-> tests.

- If only some variants were inspected, say so explicitly (for example,
  "decode path inspected, prefill not inspected"). Do not infer correctness of
  an uninspected variant from a similar one.
- For combinations the change does not support, identify the guard that
  rejects them; a missing guard is a finding, not a test gap.

## GPU-specific correctness checks

- OpenCL and Level Zero paths stay semantically consistent when both are
  supported.
- Device-capability guards match the feature used (subgroup sizes, XMX,
  extensions); unsupported devices fall back or fail explicitly.
- Layout, pitch, offset, and padding handling in index and dispatch math,
  including tail work-items.
- Packed or quantized data, at every load/decode site:
  - signed vs unsigned storage and storage vs logical value;
  - zero-point convention and where it is applied;
  - scale/group indexing and broadcast;
  - byte packing, nibble order, and address/pitch conversion;
  - vector or tile tails, including whether padded lanes can be stored.
- In-place execution, memory aliasing/reuse, and event or barrier
  dependencies. Report a race only with the shared state, concurrent access,
  and missing synchronization identified.

## Performance

Look for an explicit mechanism, not a generic "may be slower":

- added kernel launches, reorders, copies, or full-tensor memory passes;
- added host/device transfers or blocking waits in the inference path;
- allocations in hot paths or larger/longer-lived temporaries;
- lost fusion or a selection change that moves shapes to a generic or
  reference kernel (silent fallback);
- reduced parallelism, occupancy, or subgroup efficiency;
- higher compilation, model-load, first-inference, or cache-import cost;
- improvement on one GPU generation at the expense of another.

Keep measurement scopes separate (kernel time, end-to-end latency/throughput,
compile/load time, first inference, cache import/export, memory). A kernel
speedup is not an end-to-end improvement.

Distinguish newly enabled paths (absolute numbers expected) from pre-existing
paths changed by the PR (comparison against the base commit on the affected
GPU generation required). Prioritize current hardware (Xe2 and newer), but do
not drop checks for still-supported non-XMX paths, which use the plugin's
own OpenCL kernels instead of oneDNN.

## Design and maintenance

- Logic sits in the right layer; generation- or runtime-specific code is not
  mixed into generic paths.
- Workarounds explain the hardware/runtime constraint and have a removal
  condition or tracking reference.
- New files and directories are placed in a logically meaningful location.
- Flag new or changed third-party components explicitly for a human reviewer.
- Escalate architectural intent and product trade-offs to GPU owners as
  questions rather than findings.
