Linux Kernel THP Alignment Behavior and Performance Impact on OpenVINO
======================================================================

Overview
--------

Some Linux kernel versions include a change to Transparent Huge Page (THP) alignment behavior for anonymous memory allocations (commit `efa7df3e3bb5`). This behavior may introduce artificial memory gaps and prevent THP coalescence, which can significantly reduce inference performance for large and dynamic AI workloads, even when OpenVINO is configured correctly.

This document explains the affected workloads, root cause, symptoms, how to validate the issue, and how to mitigate performance regressions.

Affected Workloads
------------------

This behavior is most visible when running:

* Hugging Face Transformer models (e.g., BERT, T5, YOLOv5, etc.)
* Tokenization and dynamic batching in Transformer models lead to non-deterministic memory allocation patterns. Intermediate buffers are allocated per token length, batch size, and attention window. These buffers are often between 512 KB and 1.8 MB and may be created and freed in bursts, particularly when using ONNX Runtime or Hugging Face Optimum with OpenVINO.
* Dynamic batching or dynamic input sequence lengths
* ONNX Runtime or Optimum integration with OpenVINO
* Linux systems with THP enabled and strict PMD alignment enforced

These workloads typically allocate many buffers between 512KB and 1.8MB, created and freed frequently during inference.

Root Cause (Short Summary)
--------------------------

The kernel enforces PMD (2MB) alignment for anonymous mappings ≥ 2MB.

This can create small gaps between adjacent virtual memory regions, which:

* Prevent VMA merging
* Although the VMA flags may remain identical, gaps introduced by strict PMD alignment prevent adjacent VMAs from merging. VMA merging occurs during ``__mmap_region()``, but the problematic alignment originates earlier in ``thp_get_unmapped_area_vmflags()``, breaking a necessary precondition for huge page coalescence.
* Prevent THP from collapsing small pages into huge pages
* Increase TLB misses and page faults

The result is higher latency and lower throughput during inference.

Kernel Fix
----------

A proposed kernel fix changes the alignment logic:

* Only align mappings when the length is exactly divisible by the PMD size
* Avoids injecting gaps that break THP eligibility

This restores:

* Contiguous memory regions
* THP coalescence
* Lower latency → higher throughput

Linux kernel mailing list discussion: see commit `efa7df3e3bb5` and corresponding THP alignment patch.

Observed Impact (Example Results)
--------------------------------

Testing:

    * Throughput: + ~12.8% sustained across 10k requests
    * p95 Latency: – ~9.4% (e.g., 38.7ms → 35.0ms)
    * TLB Misses: – ~15% (perf-stat)

Environment:

    * Intel Xeon Platinum (Intel Developer Cloud)
    * Linux kernel 6.9.0-rc baseline → patched version
    * Hugging Face Transformers using FP16 tensors 

Actual results vary by hardware, model size, batch size, and token length.

How to Check if You Are Affected
--------------------------------

You may be impacted if you observe:

* Lower-than-expected inference speed with OpenVINO
* THP enabled but low huge-page usage in:
  * ``/sys/kernel/mm/transparent_hugepage/``
  * ``/proc/meminfo``
* Performance improves with different kernel version

Tools for diagnosis:

* ``perf stat`` — TLB misses, page faults
* ``cat /proc/meminfo`` — THP stats
* Kernel release notes for alignment behavior

Recommended Actions for OpenVINO Users
--------------------------------------

* Prefer kernels with the updated THP alignment fix applied
* Confirm THP status and kernel version during benchmarking
* If running on managed cloud platforms, consider this behavior when diagnosing performance issues

Even when OpenVINO is configured correctly, Linux kernel memory policy can strongly influence real-world inference performance.

References
----------

* Linux kernel mailing list discussions on anonymous mapping alignment
* Performance data from AI/ML inference workloads using Hugging Face Transformers with OpenVINO