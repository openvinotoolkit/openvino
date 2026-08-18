# Intel GPU Performance Comparison

## Revisions and Method

- M: fresh upstream `7a0a0b6b4655125b2d02ebc3c897b0ddfadbd310`.
- C0: integrated candidate before fixes, `3a2dfe0cd02d5bd8820140e39dc23eece517964e`.
- C2: latest-head integration plus the five OCL compatibility fixes,
  `e59b4712f65b7dde43aecb12fe55bd1f3696bcd1`.
- C3: final Vulkan recovery, `a723a82cf828e6302209a13a380e2a1ec7e6dda8`.

All acceptance runs used Release builds with debug capabilities and Vulkan
Validation Layers disabled. Five A/B pairs were run in alternating order on an
otherwise idle DUT. Latency used 100 warm inferences and throughput used 400
inferences. The dGPU OCL throughput result uses seven 400-inference pairs.
“Noise” is the full baseline run range divided by its median; it is not an
arbitrary acceptance threshold.

## M vs C3 — Existing OpenCL Path

The C3 production OCL binary is identical to C2: the additional fix is compiled
only into the Vulkan runtime.

| Backend | Device | Workload | M | C3 | Delta | Noise | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| OCL | UHD 730 | Warm median (ms) | 5.278 | 5.300 | +0.42% | 0.66% | PASS |
| OCL | UHD 730 | Throughput (fps) | 196.328 | 196.414 | +0.04% | 0.49% | PASS |
| OCL | Arc B580 | Warm median (ms) | 0.538 | 0.539 | +0.19% | 0.56% | PASS |
| OCL | Arc B580 | Throughput (fps) | 2402.355 | 2426.149 | +0.99% | 2.46% | PASS |

| Device | Metric | M | C3 | Delta |
|---|---|---:|---:|---:|
| UHD 730 | Compile (ms) | 523.418 | 526.805 | +0.65% |
| UHD 730 | Warm p95 (ms) | 5.424 | 5.428 | +0.07% |
| UHD 730 | Dynamic A / B (ms) | 304.812 / 8.186 | 303.285 / 8.166 | -0.50% / -0.24% |
| UHD 730 | Cache first / second compile (ms) | 477.786 / 3.805 | 476.029 / 3.693 | -0.37% / -2.94% |
| Arc B580 | Compile (ms) | 380.708 | 384.192 | +0.92% |
| Arc B580 | Warm p95 (ms) | 0.620 | 0.611 | -1.45% |
| Arc B580 | Dynamic A / B (ms) | 210.138 / 224.734 | 209.313 / 223.531 | -0.39% / -0.54% |
| Arc B580 | Cache first / second compile (ms) | 188.128 / 2.567 | 189.210 / 2.548 | +0.58% / -0.74% |

No OCL slowdown exceeds baseline variability or repeats across related metrics.

## C0 vs C3 — Vulkan Preservation

| Backend | Device | Workload | C0 | C3 | Delta | Noise | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| Vulkan | UHD 730 | Warm median (ms) | 6.525 | 6.520 | -0.08% | 0.63% | PASS |
| Vulkan | UHD 730 | Throughput (fps) | 207.730 | 206.487 | -0.60% | 6.82% | PASS |
| Vulkan | UHD 730 | Dynamic A median (ms) | 13.068 | 12.890 | -1.36% | 0.89% | PASS |
| Vulkan | UHD 730 | Dynamic B median (ms) | 5.656 | 5.636 | -0.35% | 2.17% | PASS |
| Vulkan | Arc B580 | Warm median (ms) | 4.893 | 4.894 | +0.02% | 2.35% | PASS |
| Vulkan | Arc B580 | Throughput (fps) | 330.507 | 331.580 | +0.32% | 3.18% | PASS |
| Vulkan | Arc B580 | Dynamic A median (ms) | 10.960 | 10.902 | -0.53% | 1.52% | PASS |
| Vulkan | Arc B580 | Dynamic B median (ms) | 10.693 | 10.578 | -1.08% | 2.62% | PASS |

The final iGPU values use five C0 and five C3 samples. Static and throughput
dGPU values use 12 C0 and five C3 samples. Only two C0 dGPU runs produced a
complete dynamic distribution because the newly found transient-slot defect
aborted the other ten; C3 completed 5/5 final performance runs and 10/10
additional 100-shape stress processes.

Secondary medians also remain stable: iGPU compile 31.094→31.485 ms, p95
8.932→8.520 ms, warm-cache compile 3.754→3.812 ms; dGPU compile
25.117→25.421 ms, p95 5.606→5.635 ms, warm-cache compile 2.488→2.433 ms.
The fix adds no wait, allocation, serialization, or normal-path submission; it
only resets an already completed transient command buffer that would otherwise
trip the correctness assertion.

## Caveat and Evidence

The Arc B580 kernel repeatedly reports `PCODE Mailbox failed: -6 Illegal
Command`, and its GuC firmware is older than the driver recommendation. This is
treated as an external noise source, not hidden by a wider threshold. Raw logs
and harness snapshots are preserved in
[`intel_gpu_regression_artifacts/2026-08-17`](intel_gpu_regression_artifacts/2026-08-17/README.md).
