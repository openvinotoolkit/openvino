Transformers primitive conversion tests use small local configurations and call
library implementations directly. They download no checkpoints or processors.
The matrix covers activations, projections, chunking, attention and masks,
rotary positions, caches, vision/audio building blocks, time-series utilities,
and generation score processors. This is representative library coverage;
MoE/state-space kernels, additional processors, dtypes, and dynamic-shape reuse
remain gaps.

Install the separate dependencies and run either precommit path with xdist:

```bash
python3 -m pip install -r tests/requirements_pytorch -r tests/layer_tests/pytorch_tests/requirements_transformers
PYTHONPATH=tests/layer_tests TEST_DEVICE=CPU TEST_PRECISION=FP32 OMP_NUM_THREADS=1 \
  python3 -m pytest tests/layer_tests/pytorch_tests/test_transformers.py -m precommit -n 4 --dist load
PYTHONPATH=tests/layer_tests TEST_DEVICE=CPU TEST_PRECISION=FP32 OMP_NUM_THREADS=1 PYTORCH_TRACING_MODE=EXPORT \
  python3 -m pytest tests/layer_tests/pytorch_tests/test_transformers.py -m precommit_torch_export -n 4 --dist load
```

Both paths use `PytorchLayerTest` to check captured operators, convert, compile,
and compare tensor shapes, dtypes, and numerical results. Inputs are returned
alongside outputs to preserve otherwise-unused exported inputs. Cache objects
are recreated per call. Known conversion/tracing/accuracy failures have strict
xfail marks with concrete reasons; unexpected passes fail so obsolete marks
can be removed. Transformers is optional for the general layer suite, and the
PyTorch layer CI job installs its requirements explicitly. Windows retains the
existing serial execution restriction; Linux/macOS use xdist's load scheduling.

Run the library/operator audit separately:

```bash
python3 tests/layer_tests/pytorch_tests/scripts/transformers_op_coverage.py \
  --output /tmp/transformers-op-coverage.json
```

It records eager ATen overloads per case and explicit PyTorch call sites across
the installed Transformers sources, with execution flags. Source inventory also
includes training, distributed, GPU, and architecture-specific code. A line
execution flag does not establish branch coverage; tensor methods and indirect
calls are not statically inventoried. Setup/initialization are excluded from
operator recording. The audit fails if a primitive cannot run in PyTorch and
does not establish OpenVINO conversion support. Review its gaps on library
upgrades and add direct tests for distinct primitives and branches.

CPU validation with Python 3.10, PyTorch 2.12.1+cpu, Transformers 5.18.0, and
OpenVINO 2026.5.0-78260-469d2b98bd8 used four xdist workers: trace completed in
4.93 seconds (111 passed, 21 strict xfails), and export in 25.66 seconds
(103 passed, 29 strict xfails), with no unexpected failures or skips. The audit
ran all 132 cases successfully and observed 112 ATen overloads. These timings
exclude dependency installation. CI uses the repository's pinned PyTorch version;
Windows and macOS were not validated locally.
