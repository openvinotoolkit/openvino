# Transformers primitive conversion tests

This suite verifies that tensor primitives implemented by Transformers can be
converted to OpenVINO, compiled, and executed with outputs matching PyTorch.
It calls the installed library's implementations directly, using deterministic
inputs and small local configurations. It downloads no models or processors.
It is independent of the PyTorch operator layer-test suite and does not assert
operator kinds in captured graphs.

The matrix includes activations, projection/chunking/grid utilities, attention,
causal/bidirectional/sliding/chunked/packed masks, rotary positions, dynamic and
static caches, vision/audio building blocks, time-series processing, and
generation score processors. Attention covers prefill, decode, cross attention,
and grouped heads; caches cover updates and beam reordering. Every case runs
through tracing and export and checks output count, shapes, dtypes, and numerical
agreement (`rtol=atol=1e-4`). CPU execution requests f32 inference.
Traced graphs are reshaped to the example inputs before compilation; dynamic
shape reuse is outside this matrix.

Known failures remain executable strict xfails: static sliding-cache prefill
output lengths and PyTorch tracing or export restrictions in cache reordering,
dynamic/long RoPE, MinP, and Eta sampling.
An `OpConversionFailure` always fails the test, including an expected-failure
case. No case currently requires a new OpenVINO opset operation or is skipped
for missing conversion support.

## Running the suite

```bash
python3 -m pip install -r tests/requirements_pytorch -r tests/layer_tests/transformers_tests/requirements.txt
PYTHONPATH=tests/layer_tests TEST_DEVICE=CPU OMP_NUM_THREADS=1 \
  python3 -m pytest tests/layer_tests/transformers_tests -m precommit -n 4 --dist load
```

Use `-k cache`, `-k export`, or another primitive/mode name to select cases.
Inputs are returned alongside primitive outputs so export keeps otherwise-unused
inputs. Cache objects are recreated for each invocation to keep conversion and
reference execution independent. xdist distributes cases across worker processes.
The 132 primitives produce 264 trace/export cases. A local CPU run with Python
3.10, PyTorch 2.12.1+cpu, and Transformers 5.18.0 took about eight seconds with
four workers (252 passed, 12 expected failures), excluding dependency installation
and build time. CI runs the suite as a dedicated precommit step.

## Establishing library coverage

A fixed list of examples cannot establish exhaustive coverage of Transformers.
This suite currently covers representative primitive families, not every tensor
primitive in every architecture. The following mechanisms expose missing coverage:

* Activation and RoPE cases are discovered from Transformers' registries, so new
  registered variants become cases automatically. A variant requiring a new
  configuration fails rather than being silently omitted.
* Each case has a stable family/name ID and uses the actual library implementation.
* The audit below records ATen overloads exercised by each case and scans all
  installed Transformers Python sources for explicit PyTorch call sites. It flags
  call sites whose source lines were not executed during the primitive matrix.
* Review the uncovered source implementations at every library upgrade. Identify
  distinct tensor primitives and branches, add direct input recipes, and validate
  both conversion paths. Do not declare full coverage while uncovered inference
  primitives lack a case or an explicit exclusion.

```bash
python3 tests/layer_tests/transformers_tests/scripts/op_coverage.py \
  --output /tmp/transformers-op-coverage.json
```

The JSON includes library versions, per-case eager results, ATen overloads, and
source call sites with execution flags. It fails if any primitive cannot execute
in PyTorch. Source inventory includes training, distributed execution, device
kernels, and architecture-specific code. Tensor methods and indirect calls are
not statically inventoried, and executing a source line does not establish branch
coverage. Initialization is excluded from ATen recording. This audit is a gap
finder, not proof of full coverage or conversion support.

Remaining coverage includes MoE routing, recurrent/state-space kernels,
multimodal packing, additional generation processors, dtype variants, fully
masked attention rows, and reuse with dynamic shapes. Keep those gaps visible
when reporting suite coverage.
