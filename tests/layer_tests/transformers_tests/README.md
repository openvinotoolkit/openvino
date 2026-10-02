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
The matrix includes 132 shared recipes and 65 model-local recipes, each running
through trace/export. CI runs the suite as a dedicated precommit step.
The precommit coverage-inventory test also runs the eager audit once, independently
of conversion mode and device. Timings exclude dependency installation and builds.
The expanded precommit matrix takes about 26 seconds locally with four workers
(383 passed, 12 expected failures; Python 3.10, PyTorch 2.12.1+cpu).

## Establishing library coverage

A fixed list of examples cannot establish exhaustive coverage of Transformers.
This suite currently covers representative primitive families, not every tensor
primitive in every architecture. The following mechanisms expose missing coverage:

* Activation and RoPE cases are discovered from Transformers' registries, so new
  registered variants become cases automatically. A variant requiring a new
  configuration fails rather than being silently omitted.
* Each case has a stable family/name ID and uses the actual library implementation.
* Model-local recipes import the named function/class from its actual modeling
  module. They cover five rotation helpers, head repetition, 25 RoPE application
  variants, 21 eager attention implementations, and 13 normalization classes.
  Cases exercise interleaved/partial/trailing RoPE, vision prefix preservation,
  audio rotation, attention sinks, soft-capping, position bias, and channel-first
  normalization. Nonuniform norm weights prevent identity initialization from
  hiding differences.
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

`test_coverage.py` invokes this script in precommit and compares its results with
the committed `coverage_inventory.json`. It fails when the suite requirements,
Transformers version, source-call inventory, primitive IDs, or per-primitive ATen
overloads change. It runs on every precommit invocation so the gate also works in
installed test packages without Git history or changed-file metadata. Requirements
updates therefore cannot silently retain an outdated coverage inventory.

For a dependency update, inspect the full audit JSON and compare it with the old
version's report. Add recipes for new inference primitives, or document explicit
exclusions and reasons here, before updating the inventory:

```bash
python3 tests/layer_tests/transformers_tests/scripts/op_coverage.py \
  --output /tmp/transformers-op-coverage.json \
  --update-baseline tests/layer_tests/transformers_tests/coverage_inventory.json
```

Commit the reviewed inventory with the requirements and recipes. Regenerating the
inventory alone acknowledges changes; it does not establish coverage of new
primitives. Activation/RoPE registries discover new variants automatically, while
other families still require source review and explicit recipes. The inventory
excludes executed source-line counts because optional dependency paths can differ
across platforms; the full audit report retains those counts for review.

The audit inventories model-local definitions named `rotate_half`, `repeat_kv`,
`apply_rotary_pos_emb`, `eager_attention_forward`, and classes ending in `RMSNorm`
or `LayerNorm`. For Transformers 5.18.0 it finds 1,071 definitions in these families;
65 have direct recipes across 55 model modules. Each record contains its source
file, symbol, implementation fingerprint, and `direct_recipe` flag. Untested
definitions are not silently treated as covered because another model has similar
code. Fingerprints help identify duplicates for review; they do not prove semantic
equivalence, particularly when helper functions, configuration, or inheritance
differ. Direct recipes exercise 57 of 85 recorded implementation fingerprints;
this does not count the remaining copies as directly tested. Changes to this
inventory also fail the precommit baseline check.

This naming-based inventory does not discover every primitive: model-local MLPs,
MoE routers, state-space blocks, patch/position embedding classes, and arbitrary
tensor helper methods still need additional discovery rules and explicit recipes.
The expanded eager audit executes 294 of 44,453 explicit PyTorch call sites (0.66%)
in 66 files and observes 115 ATen overloads. These are call-site observations,
not whole-library line/branch coverage or exhaustive operator coverage.

Remaining coverage includes MoE routing, recurrent/state-space kernels,
multimodal packing, additional generation processors, dtype variants, fully
masked attention rows, and reuse with dynamic shapes. Keep those gaps visible
when reporting suite coverage.
