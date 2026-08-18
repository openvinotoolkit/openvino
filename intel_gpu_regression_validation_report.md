# Intel GPU Regression Validation Report

## Revisions and Scope

Validation was performed on one Intel system containing both integrated and
discrete GPUs. The comparison revisions are:

- Fresh upstream master (M): `7a0a0b6b4655125b2d02ebc3c897b0ddfadbd310`
- Candidate before fixes (C0): `3a2dfe0cd02d5bd8820140e39dc23eece517964e`
- Validated candidate after fixes (C1): `b094deb12936614a87f7f5dea413c5ebc9aaee57`
- Latest remote branch head: `7125f598962192bdc40dadf79155e40006d6f9eb`
- Latest-head integration with fixes (C2): `e59b4712f65b7dde43aecb12fe55bd1f3696bcd1`
- Final candidate with Vulkan recovery (C3): `a723a82cf828e6302209a13a380e2a1ec7e6dda8`

C1 contains the regression fixes and fresh-master merge. C2 adds the latest two
branch commits; its tree differs from C1 in 11 host-size/test portability files.
C3 fixes a transient Vulkan command-slot lifecycle defect found by extended
dynamic stress and adds its regression test.
The detailed machine inventory is in
[`intel_gpu_regression_machine_inventory.md`](intel_gpu_regression_machine_inventory.md).

## Machine and Device Mapping

- Host: `DUT2198BMGFRD`, Ubuntu 24.04.3, kernel 7.0.0-29, Core i5-13400
- iGPU: Intel UHD Graphics 730, `0000:00:02.0`, i915, renderD128
- dGPU: Intel Arc B580, `0000:03:00.0`, xe, renderD129
- OpenCL: Intel Compute Runtime 26.27.39122.12, IGC 2.38.3
- Vulkan: Mesa 25.2.8, loader/layers 1.3.275

| Configuration | iGPU | dGPU | Available IDs |
|---|---|---|---|
| M/C1 OCL-only | `GPU.0` | `GPU.1` | `0 1` |
| C1 Vulkan-only | `GPU.0` | `GPU.1` | `0 1` |
| C1 mixed default OCL | `GPU.0` | `GPU.1` | `0 1 vulkan_0 vulkan_1` |
| C1 mixed explicit Vulkan | `GPU.vulkan_0` | `GPU.vulkan_1` | same |

Both GPUs were also compiled and inferred from the same `ov::Core`; alternating
requests and simultaneous async inference passed with distinct stable checksums.

## Build Matrix

| Revision | OCL-only | Vulkan-only | OCL+Vulkan |
|---|---|---|---|
| M | Plugin, functional tests, benchmark and sample PASS | N/A | N/A |
| C0 | Production targets PASS; test build exposes candidate defects | Production targets PASS; full test target FAIL | Production build runs into the known pre-fix ABI corruption |
| C1 | Plugin, unit, functional, benchmark and sample PASS | Same full target set PASS | Same full target set PASS (4243/4243) |
| C2 | Plugin, unit, functional, benchmark and sample targets PASS | Production target 1865/1865 | C1 evidence retained; C2 changes are host-size/test portability only |
| C3 | OCL binary unchanged from C2; production harness PASS | Incremental plugin and unit target PASS | Current mixed production plugin built 1219/1219; OCL and Vulkan smoke PASS on both GPUs |

All builds use GCC 13.3, Ninja, Release, the same system dependencies, and
configuration-specific runtime lists. Test builds enable GPU debug capabilities;
performance builds must not.

## Correctness and Compatibility Results

| Test set | iGPU | dGPU | Result |
|---|---:|---:|---|
| C1 OCL architecture/correctness loop | 580/580 | 580/580 | PASS |
| C2 OCL architecture/correctness loop | 580/580 | 580/580 | PASS |
| C1 expanded OCL functional set | 1046 pass, 239 skip, 1 baseline fail | 1044 pass, 238 skip, 4 baseline fails | PASS vs M |
| C2 expanded OCL functional set | 1046 pass, 239 skip, 1 baseline fail | 1044 pass, 238 skip, 4 baseline fails | PASS vs M/C1 |
| C2 changed-operation portability set | 434 pass | 430 pass, 3 skip, 1 baseline fail | PASS vs M/C1 |
| C1 Vulkan architecture loop under validation | 270/270 | 270/270 | PASS |
| C0/C1 Vulkan harness | PASS | PASS | Exact checksum parity |
| C1 OCL cross-device alternating + simultaneous async | PASS | PASS | Same process |
| C1 mixed OCL and explicit Vulkan harness | PASS | PASS | Both runtimes, same plugin |
| C2 OCL harness and cross-device run | PASS | PASS | Exact C1/M checksums |
| C3 transient-slot module test, 5 repeats | 5/5 | 5/5 | 500 reset transitions per GPU |
| C3 high-level 100-shape stress processes | 10/10 | 10/10 | C2 dGPU failed 3/5 before fix |
| C3 final Vulkan performance/correctness runs | 5/5 | 5/5 | Exact checksums and cache PASS |
| C3 mixed production OCL harness | PASS | PASS | Static/dynamic/import/concurrency, exact checksums |
| C3 mixed production Vulkan harness | PASS | PASS | Export/import, cache, concurrency and 20 dynamic transitions |

The iGPU functional failure is `VersionTests.pluginCurrentVersionIsCorrect`.
The dGPU has that failure plus three compiled-blob/cache-hint failures. Each
failure reproduces on M and is therefore not candidate-specific.

The C2 dGPU portability sequence also exposes
`convolution_gpu_bfyx_f16.dynamic_fused_input_scalar_and_non_scalar_fp32` with a
NaN on the reference side. The identical ordered filter fails on M and C1;
running the test alone passes 20/20 on M, C1 and C2. It is an order-dependent
baseline test defect, not a candidate regression.

Static inference, repeated dynamic shapes, model export/import, model cache in both
cross-device orders, and four-request concurrency pass on both OCL devices. A
fresh C3 mixed-production plugin additionally passes OCL static/dynamic/import/
concurrency smoke on both devices. The Vulkan harness passes static/dynamic
inference, cache, export/import and four concurrent requests with exact
C0/C1/C3 checksums; its mixed C3 run also passes 20 alternating dynamic
transitions per GPU. The new transient-slot test
also passes once per GPU with Validation Layers enabled, with zero validation
errors. Before the C3 fix, C0 and C2 intermittently failed dynamic Vulkan
inference with `Transient command resources were not released`; the root cause
and recovery evidence are recorded in `intel_gpu_fixed_regressions.md`.

## Numerical Evidence

| Backend | Device | Static checksum | Dynamic checksums |
|---|---|---:|---:|
| OCL C1/C2/C3 | UHD 730 | 1993.155740798 | 145.552655995, 72.916523933 |
| OCL C1/C2/C3 | Arc B580 | 1983.154060781 | 149.185859382, 71.118761301 |
| Vulkan C0/C1/C3 | UHD 730 | 3329.866262197 | 843.844614804, 412.789550364 |
| Vulkan C0/C1/C3 | Arc B580 | 3329.866262197 | 843.844614804, 412.789550364 |

## Acceptance Matrix (Checkpoint)

| Gate | iGPU | dGPU |
|---|---|---|
| Build | PASS | PASS |
| Device enumeration | PASS | PASS |
| Compile model | PASS | PASS |
| Static inference | PASS | PASS |
| Dynamic inference | PASS | PASS |
| Cache | PASS | PASS |
| Export/import | PASS | PASS |
| Multi-stream | PASS | PASS |
| OCL regression suite | PASS vs M/C1/C2 | PASS vs M/C1/C2 |
| Vulkan regression suite | PASS C3 | PASS C3 |
| Validation Layers | PASS | PASS |
| Performance vs fresh master | PASS | PASS |
| Vulkan performance vs C0 | PASS | PASS |

Raw logs, GoogleTest XML and cache/harness artifacts are mirrored under
[`intel_gpu_regression_artifacts/2026-08-17`](intel_gpu_regression_artifacts/2026-08-17/README.md).

## Final Result

The existing OpenCL path preserves M correctness, enumeration, caching,
serialization, dynamic execution, concurrency, latency and throughput on both
GPUs. C3 also preserves C0 Vulkan performance while fixing the newly exposed
dynamic transient-resource crash without a wait or feature reduction. No
candidate-specific failure remains in the completed gates.

## Hardware Caveat

The dGPU kernel log repeatedly reports `PCODE Mailbox failed: -6 Illegal
Command`; GuC 70.44.1 is below the driver-recommended 70.54.0. Results remain
repeatable, but this external firmware condition must be considered when judging
performance noise.
