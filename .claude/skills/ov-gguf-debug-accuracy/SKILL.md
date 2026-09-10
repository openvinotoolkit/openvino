---
name: ov-gguf-debug-accuracy
description: >
  Debug wrong or degraded output from a GGUF model running through the OpenVINO GGUF frontend or
  the ggml-openvino backend. Use when a GGUF model converts but produces garbage, repeated or
  drifting tokens, a cosine-similarity cliff between layers, output that diverges from llama.cpp
  CPU, or a broadcast/shape crash that appears only at decode or only for one architecture. Do
  NOT use for ops that fail to convert at all ("Translation for operation type ... is not
  implemented"), or for build and CMake failures.
---

1. Read [src/frontends/gguf/docs/debugging_accuracy.md](../../../src/frontends/gguf/docs/debugging_accuracy.md) — the coarse-to-fine bisection strategy, the ggml-CPU oracle technique, the catalogue of bug archetypes with their generalizable lessons, the debug env vars, and the checklist.
2. Apply its one governing rule before anything else: every accuracy claim is a comparison against the **real llama.cpp CPU implementation**, never a hand-derived reference. If you are about to write out an op's math to form an expectation, generate it from ggml instead.
3. Work the bisection steps in order and do not open a debugger until they have cornered the bug. Check the bug archetypes first when a cosine cliff or dynamic-shape crash points near one.
4. After any fix to a shared path — especially a VIEW or `op_case` predicate — re-run the other supported architectures and confirm their classification and output are unchanged.
