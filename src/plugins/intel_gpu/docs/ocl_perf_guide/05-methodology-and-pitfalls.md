# 05. Methodology, performance-claim discipline, timeline, pitfall catalogue, xe_hpg port

This chapter records the **working methods** that settled while building the `sdpa_ocl` kernel family (`ocl_v2/sdpa_ocl*.cl`, host `ocl_v2/sdpa/`, design doc `docs/sdpa_ocl.md`).
For the kernel techniques themselves see the other chapters: DPAS/tiling (→ 01-dpas-and-tiling.md), memory IO/prefetch/barrier (→ 02-memory-io-prefetch-barriers.md),
numerics/softmax/quantization (→ 03-numerics-softmax-quantization.md), spill/ISA/profiling (→ 04-spill-isa-profiling.md).

Notation: **MEASURED** = measured on hardware by the user or an agent, with a source note. **PREDICTED/ASSUMED** = a prediction from reading code or from a model.
Hardware: **B580** = Arc B580 (Xe2, early development machine), **B70** = Arc Pro B70 (Xe2, development machine since 2026-07-05, `GPU.1`/`--device_suffix=1`),
**iGPU** = TGLLP UHD (no XMX, `GPU.0`), **DG2** = Arc A770 (Xe-HPG, IP 12.55.8, separate PC), **ARL-H** = gfx 12.74 (not verified on real hardware).
The source notation `memory:<name>` means `~/.claude/projects/-home-shingyuk-work-openvino-eddy/memory/<name>.md` (local memory on the development PC; not in the repository).

---

## 5.1 Division of work and working style (repeated in the repository AGENTS.md + user feedback)

| Rule | Content | Basis |
|---|---|---|
| The user executes | The user runs builds, gtests, ab (full corpus), dumps, benchmarks, and even `-fsyntax-only`, `built`, `l0`. The agent provides **exact command blocks** and **predicted values** | memory:build-is-user-only, ab-runs-are-user-only (delegation revoked 2026-09-27) |
| The user commits | The agent only proposes a one-line message. Reasons/rationale go in code comments and `docs/sdpa_ocl.md`, not in git history | memory:short-commit-messages |
| Plan approval before code changes | Detailed plan -> approval -> implementation. Out-of-scope findings are not fixed but recorded in `docs/sdpa_ocl.md` "Known issues" | AGENTS.md "Do not expand task scope", refactor 2026-09 |
| Split long work into sessions | At the end of a session, update memory + leave a **resume prompt** (a Korean-language prompt at the end of each `sdpa-ocl-ki-t*` note) | PROMPTS.md |
| Waiting/polling | Poll background jobs in `sleep 240` steps; do not use 600 | memory:wait-with-sleep-240 |
| Numeric-claim discipline | No "faster" without a measurement; state the methodology (hardware/pinned clock/repetitions/metric); distinguish measured vs assumed | AGENTS.md |
| find->verify workflow | **Dedupe** the per-lens findings **first**; 2 refuters only for blocker/major, 1 for minor, or the agent checks directly with grep. Got the remark that 80 verify agents for 40 findings was "too much" | memory:workflow-verify-dedupe-first (2026-09-27) |
| No editing a running script | Check first with `ps -eo pid,etime,args \| grep <script>`. bash keeps reading from the old byte offset, so if the length changes it executes the tail of the file as commands. If unavoidable, keep the byte length identical | memory:dont-edit-running-scripts (t2_census.sh incident) |

---

## 5.2 A/B harness — tools that prove "behaviour preserved" without a GPU

### 5.2.1 Tools and locations (all untracked, under `test/`; excluded locally by the repository `.gitignore`)

| Tool | Role |
|---|---|
| `test/sdpa_ocl_ab.py` | Offline A/B. Subcommands `snapshot, embed, l0, corpus, ab/l1/l1p/l2, selftest, xab, pset, built, hpg, modes` (`sdpa_ocl_ab.py:1147-1187`) |
| `test/sdpa_ocl_gtests.sh` | gtest runner **run by the user**. `run <tag> [group..]`, `dump <tag>`, `diff <a> <b>`, `list`. Group table `GROUPS_DEF` (`sdpa_ocl_gtests.sh:34`) |
| `test/sdpa_ocl_known_issues/` | Per-issue repro kit: `repro.sh list\|<item>\|all`, `t2_census.sh` (dispatch census), `*_syntax.sh`, python reproducers. Arms differ in only one environment variable / one device, print observed vs predicted, and mark mismatches with `!!` |
| `test/microbench/` | (a) ocloc static ISA comparison `compare_isa.sh *.cl`, (b) real-GPU layout probe `run_probe.sh`, (c) `probe_dpas_api.sh` (which DPAS/2D-block builtins exist). **Device fixed to `bmg`** |
| `test/dump_isa.sh` etc. | Real kernel ISA dump (→ 04-spill-isa-profiling.md) |
| `test/sdpa_hpg_classify.sh`, `test/sdpa_perf_hpg.py` | DG2 failure classification / short plain-SDPA performance table (for the record, not a gate) |

### 5.2.2 Proof levels (what "identical" means at each level)

| Level | Compared | If identical, it means | Limits |
|---|---|---|---|
| **L0** | Embed text of the kernel source (after stripping comments/indentation) | The bytes that go into the `.inc`/.so are identical | Only comments/indentation are L0. Line breaks in code or whitespace around `* / ( ) ? :` are L1 |
| **L1** | `clang -E` **token list** of each config | Tokens identical after preprocessing (whitespace-insensitive) | clang runs without the OpenCL default headers, so driver-defined macros are not expanded. Not an oracle for what IGC sees |
| **L1'** | IGC front-end `.ll` | FE IR identical (added parentheses, etc.) | Because of the token cache, L1' of token-identical text is a copy of base (tautology). The verdict comes from L1/L2 |
| **L2** | ISA metrics of the `ocloc -device bmg` output (`instCount, numGRF, opcode histogram, zeinfo, call/ret`) | **ISA grade A = identical** | B = heavy ops identical + no scratch increase + inst ±0.5% (needs a performance measurement), C = everything else |
| **xab** | Pairs of two corpora (e.g. r0 vs r1), ISA compared after removing retired jit names | Same kernel even after the host jit cleanup | `XAB_DROPPED` must be extended at each step |
| **pset** | **multiset** of the host jit `(name, value)` of two corpora | A host-only change does not alter the jit set | Reordering is allowed (the prelude is only `#define`s; 0 duplicates is re-checked every time) |
| **built** | Is the base text present in the built `.inc` (chunks joined) + the linked `.so` + `ov_gpu_unit_tests` | The kernel I verified went into the **actual binary** | False negative if run during linking |
| **hpg** | Offline compile of the corpus with `ocloc -device dg2/arl-h`, TSV | Static check on xe_hpg of compile/DPAS count/spill/SLM | Not execution (→ 5.7) |

### 5.2.3 Commands (the form verified in memory:sdpa-ocl-ab-harness)

```bash
python3 test/sdpa_ocl_ab.py snapshot s1f                                  # freeze the working-tree .cl (no --force: breaks the proof chain)
python3 test/sdpa_ocl_ab.py l0 --base s1e [--work work]                   # embed text identical?
python3 test/sdpa_ocl_ab.py corpus test/sdpa_ocl_corpus/r0 --base 235639f1a2   # index the dumps; base mismatch 0 required
python3 test/sdpa_ocl_ab.py ab --corpus <dir> --base X --work Y --levels l1,l1p,l2 --l2-from-l1 -j 28
python3 test/sdpa_ocl_ab.py xab <corpusA> <corpusB>
python3 test/sdpa_ocl_ab.py pset <corpusA> <corpusB> [--l1 -j 28] [--selftest]
python3 test/sdpa_ocl_ab.py selftest --corpus <dir> --base <sha>          # always rerun after fixing the tool (negative control)
python3 test/sdpa_ocl_ab.py built --base s1g [--release]                  # only after the build has finished
bash test/sdpa_ocl_gtests.sh run r0 [U1 U2 ...]; bash test/sdpa_ocl_gtests.sh dump r0
bash test/sdpa_ocl_gtests.sh diff test/sdpa_ocl_corpus/r1/logs test/sdpa_ocl_corpus/r3/logs
```

Scale (MEASURED, memory:sdpa-ocl-ab-harness, 32 cores `-j 28`): corpus 2332 configs + 6110 synthetic jobs. L1+L2 cold ~30 min, one side cached ~10 min, L1 only 2-3 min.
ocloc compile time per 1 config: 0.2-2 s.

### 5.2.4 Why this harness is "exact" (the points where misuse yields a false proof)

- **Build Log removal**: the runtime appends `/* Build Log: <dump path> ... */` to the end of the dumped source **after** compiling. If it is not removed, 6177 dumps become 6177 "configs" (`BUILD_LOG_RE`, `sdpa_ocl_ab.py:235`).
- **base is stored as a SHA** (`base_id()`, `:121`). If the index records `HEAD`, the kernel body silently changes after the next commit. Committing moves HEAD, so the R0 reference is always an explicit SHA such as `235639f1a2`.
- **dump == embed(base) minus the `#include batch_headers` line, byte-identical**: `corpus` proves which build a dump came from (the 9/11 dumps match 81a3ffa944 and not the next commit).
- **L1 is a token list, not a string** (fixed 2026-09-24): the whitespace-stripped comparison judged `const int` == `constint` and `a - -b` == `a--b` to be equal. The selftest `merge` case must produce an L1 DIFF.
- **Token-key cache** (`cache/tok/`): text with identical tokens reuses the compile. A new snapshot compiles only the configs whose tokens changed (B3: 1462 of 8448).
- **`--l2-from-l1`**: if L1 is identical, the work-side compile is skipped. Valid because ocloc is deterministic in its input tokens, but wrong for an edit that adds an `#if` on a macro that only the driver predefines.
- **`grep spill` is meaningless**: every .asm has `-abortOnSpill` in `//.full_options`. The real signals are zeinfo's `*scratch*/*spill*/*private*` and `numGRF`.
- **`sync` is a heavy op**: SWSB `sync.nop/allrd/allwr` ±1..5 alone produces a C (74 of 86 in T4). It can be cleared if the send/dpas/math/load/store/CALL counts did not move and the source has no barrier. `disable_mid_thread_preemption true->None` is IGC's **~600-inst threshold** (true for instCount<=598, None for >=601, exhaustive over 109,802 compiles) and not a functional change.
- **Synthetic modes**: `modes` (q2d0, kv2d0, cur0, mm0, bidir0, gate0, dpf0 ...) flip a single macro that has no host-dependent jit. Toggles where the host also changes other jits (`K_PA_I8_2D=0` turns on u4 1D) appear only in real env dumps (U4_* groups).
- **Runtime build options** (used for L2): sdpa_ocl uses `-Dcl_intel_dot_accumulate -Dcl_intel_global_float_atomic -Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate -cl-mad-enable -cl-std=CL3.0`, decode uses `-cl-mad-enable -cl-std=CL3.0`. The driver internal options are `-internal_options '-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4'`.
  `-device bmg` is the B70/B580 target (`-device xe2` is interpreted as the lnl-m iGPU). `-cl-mad-enable` must be inside `-options`.
- **ocloc result == runtime IGC dump** (MEASURED, memory:sdpa-ocl-kq-tile-keys-32-slower): at k16, dpas 32=32, send.ugm 98=98, total inst 3039=3039, spill 0=0, identical hash. So ocloc figures are the ground truth for the runtime kernel. **Spill is the exception** (see 5.4).

### 5.2.5 Knob A/B "without a build" (editing the jit `#define` directly)

Tile macros and the like are not baked into the source but injected as jit `#define`s, so with `OV_GPU_DUMP_SOURCES_PATH=./ OV_GPU_MAX_KERNELS_PER_BATCH=1 TEST_USE_SDPA_OCL=1 SDPA_SKIP_REF_CHECK=1 <test> --device_suffix=1`
dump a single `.cl` -> edit `#define kq_sg_tile_keys 16` -> `ocloc compile -device bmg` -> extract the `.text.sdpa` offset with `readelf -S -W X_bmg.bin` -> `iga64 -p=xe2 -d kernel.gen`.
SLM/GRF/spill come from the `*.zeinfo` of `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./occ` + `//.thread_config numGRF=`. This method was used to compare structural mutations in ISA without rebuilding OpenVINO
(memory:sdpa-ocl-kq-tile-keys-32-slower, the "ocloc splice" in sdpa-ocl-block2d-gate-relaxation). `OV_GPU_MAX_KERNELS_PER_BATCH=1` puts 1 kernel per program, which makes the dump a single entry.

### 5.2.6 Proof recipe by kind of code change (settled during the known-issues work, memory:sdpa-ocl-known-issues-repro)

| Change | Proof |
|---|---|
| Behaviour-preserving refactor of kernel `.cl` | `l0`/`l1`/`l2` (whole corpus); helper extraction requires `always_inline` (5.6 #17) |
| Kernel bug fix | In `ab --levels l1,l2`, **only the target configs** may leave A. All others stay A |
| Host-only change | New `dump <tag>` -> `corpus` -> `pset r3 <tag>` (+`--l1`) + `gtests.sh diff r3/logs <tag>`; only the intended state transitions are allowed (T7: `USE_2D_BLOCK_IO_Q 1->0` exactly 105 configs) |
| Adding a test | First check on the code **before** the fix that the negative control FAILs as predicted (T8: of 8, f16 4 PASS / bf16 4 FAIL) |
| Behaviour change | A/B differing is expected. It is a subject of **measurement**, not of proof. Prove only that untouched configs are unchanged |

Reference corpus names: r0 (R0=`235639f1a2` binary), r1 (s1g), r3 (Phase 3), rT3/rT4/rT7/rT8 (issue stages), rS4/s5_* (xe_hpg).
"identical to r3" always means `gtests.sh diff test/sdpa_ocl_corpus/r3/logs <tag>`. r3 baseline: U2 FAIL 2, F2 FAIL 15 SKIP 7, NEG_bidir FAIL 44, NEG_gate FAIL 11, everything else PASS.
The `NEG_*` groups are negative controls that **must FAIL** (`SDPA_OCL_BIDIR=0`, `SDPA_OCL_BIDIR_GATE=0`): a PASS means the suite does not observe the feature.

---

## 5.3 Execution/environment pitfalls (in the order they cost us)

| # | Pitfall | Symptom | Remedy |
|---|---|---|---|
| E1 | **2 GPUs** (iGPU `GPU.0` + B70 `GPU.1`) | Without `--device_suffix=1` it runs on the iGPU, produces `sdpa_opt__*` and ignores `TEST_USE_SDPA_OCL`. ocl/micro dumps are byte-identical, `//.platform TGLLP` | `--device_suffix=1` on every gtest. B70=1, iGPU=0. (On the DG2 PC the numbering differs: OV `GPU.1` = OpenCL HWQ `0`) (memory:dual-gpu-device-suffix) |
| E2 | **Wrong gtest binary** | `0 tests from 0 test suites` (no error; looks like a stale build) | `src/.../tests/unit/**` -> `ov_gpu_unit_tests` (most PA/SDPA), `tests/functional/**` -> `ov_gpu_func_tests` (`ScaledAttn*`, F1/F2). Location `bin/intel64/{Debug,Release}/` (not `build/`) (memory:gpu-func-test-binary) |
| E3 | **`setupvars.sh` wipes the arguments with `set --`** | In a script, `for kv in "$@"; do export "$kv"; done` after `source setupvars.sh` exports nothing -> 3 "different" configs were all measured at the defaults (1 hour lost) | Capture `SDPA_ARGS=("$@")` **before** sourcing. Sign: different configs but identical SLM/GWS/LWS. Confirm that the toggle took effect with `SDPA_OCL_TRACE_CONFIG=1` or a source dump. For python items, do conda activate + setupvars **outside** the script |
| E4 | **`NAME=VALUE` in shell helpers** | `P=x cmd` passed via `"$@"` is an expanded word, so it becomes the command name -> "command not found" | `env NAME=VALUE ... cmd` |
| E5 | **Kernel embedding** | `.cl` is not loaded from disk at runtime but embedded at build time into `gpu_ocl_kernel_sources.inc`. Real path `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/` (no `src/`; the `.../intel_gpu/src/graph/...` copy is stale); long kernels are split into raw-string chunks, so grep misses tokens | `sdpa_ocl_ab.py built`. `ov_gpu_unit_tests` links the graph library **statically**, so it has its own copy of the kernels: if the unit-test relink is stale, it runs the old kernel (memory:sdpa-ocl-kernel-embedding) |
| E6 | **Debug vs Release confusion** | The user said "I built it" but only the Release plugin (tests OFF) had been rebuilt (T2 22:1x). The Release binary was a Phase 2 build, so a rebuild was needed before the Release performance measurement | Check with the binary mtime/`built`. Performance measurement on Release/RelWithDebInfo, accuracy on the Debug corpus |
| E7 | **In-process env caching** | The lane cannot be flipped within the same process. env is read 1 time, with raw `std::getenv`, at host compile time | Compare lanes in separate processes |
| E8 | **Driver build cache** | On a cache hit the build log is empty, so build-log/dump analysis is impossible | `NEO_CACHE_PERSISTENT=0` (DG2 S2 lesson). The model blob cache does not include lane/stage in its key and reuses an old micro blob (`compiled_model.cpp:282-303`) |
| E9 | **Tensor dump env** | `OV_` prefix, `OV_GPU_DUMP_TENSORS_PATH` needs a trailing `/`, `OV_GPU_DUMP_LAYER_NAMES` is a **full regex_match on upper-case names** (wrap in `.*`), `OV_VERBOSE` (not `OV_GPU_VERBOSE`)=4 prints `Enqueue stage <kernel>`, `OV_GPU_DUMP_ITERATIONS=0` is the warm-up infer (memory:gpu-tensor-dump-env). `stream.finish()` is called before the dump, so it is not racing |
| E10 | **`-fsyntax-only` blind spot** | Does not catch unused functions in an anonymous namespace (-Wunused-function belongs to the codegen stage) | Check the call sites by hand. Extract the flags from `CXX_(DEFINES\|INCLUDES\|FLAGS)` in `flags.make` with `sed`. `-Wall -Werror -Wmissing-declarations` |
| E11 | **Removing a variable in a mutation check** | The build fails on `-Werror` unused -> the results of the old binary are read | Keep the variable and add `&& false` (S5) |
| E12 | **Combining mutations** | MUT-2+3 in one build -> FAILs outside the union of the predictions | Build each mutation alone |
| E13 | **Clock during performance measurement** | If the B70 clock fluctuates, the sign flips | Pin 2800 MHz; separate processes in ABBA interleaving |
| E14 | **`SDPA_OCL_NO_GETENV` build** (ON by default since 2026-10-19) | every `SDPA_OCL_*` env knob is a silent no-op; a sweep measures the default N times | build a separate install with `-DSDPA_OCL_NO_GETENV=OFF` and `cmake --install <build> --prefix <dir>`, then switch it back ON; confirm a production install by `strings libopenvino_intel_gpu_plugin.so \| grep ^SDPA_OCL_` matching the baseline, and a new `.cl` by a unique string such as a changed `#error` text |
| E15 | **arm names that are prefixes of each other** in result globs | `tgt_r*.log` also matched `tgt_rebaseB_r*.log`, so one arm's statistics silently contained the other arm (sample count doubled) | match `<arm>_r[0-9]*`; check the per-arm sample count |
| E16 | **git-lfs post-commit hook during a rebase** | about 7 s per replayed commit; a 98-commit rebase took about 10 min and outlived a 300 s tool timeout | run it in the background; keep a backup branch; drop uncommitted work first (save it as a patch and check that the save path exists) |

---

## 5.4 Attribution and verification rules

1. **An A/B differs in exactly one thing.** A one-sided control (toggle on only, or off only) is a hypothesis, not a conclusion.
   - Incident (2026-08-14, head-486 `CL_OUT_OF_RESOURCES`): the full `*ScaledAttn*` sweep died. Ran only that test alone with `SDPA_OCL_DKS_ACTIVE=0` -> passed -> concluded "definitive, ours" and designed a fix. The user ran it alone without the toggle -> also passed. The variable was not the toggle but **inside/outside the sweep**, and the cause was suite-level GPU memory pressure. We very nearly "fixed" correct code.
   - If a crash happens only in long runs, **the execution context is the variable**: compare sweep vs sweep.
   - A provably no-op toggle (confirmed byte-identical with HEAD by ocloc splice) is a useful control.
   - **A heterogeneous failure list is itself evidence**: if graph-transformation tests are mixed in with kernel tests, at least one of them is unrelated to the kernel. Diff the **failure sets** between configurations before building a theory (`*paged*` 85 vs 70 -> 0 ocl-only failures).
2. **Check that the control is a real control**: in the compressed BY_CHANNEL tests, `TEST_USE_SDPA_OCL(_DECODE)=0` was **not** a control before the T1 fix (the fallback read the token-major page as d-major and FAILed on its own). The control is `OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR=0` (d-major page, U3 group) or "same test, before vs after the fix".
3. **Run the negative control before the fix.** T7: before the fix, the control revealed that my prediction "wrong unless the base is 64 B aligned" was **wrong** (B70 measurement: stride %16 is the rule; a base misalignment of 4/16/32/48 B is fine if the stride is a multiple of 16 B). T8: the functional test data was degenerate (5.6 #20), so it would have passed on the old code too -> found by the control.
4. **A metric that moves is not necessarily a metric that matters.** spill 2688->0 (`SDPA_OCL_256GRF=1`), yet the kernel got **47% slower**: the spill was a symptom and the LSC message count was the real cost (memory:sdpa-ocl-block2d-gate-relaxation). cliloader occupancy% is **inversely correlated** with speed (the slowest config had occ 70%) -> do not use it as a performance proxy.
5. **Oracle location**: in a chained network only **the first layer** has identical inputs (memory:layer0-identical-input-oracle). The per-layer average error is propagated error, and a correct kernel and a buggy kernel saturate to the same value (minicpm4: `L0 3e-5 -> L1 7e-3 -> L2 5e-2 -> plateau 0.08`). Compare the L0 error with the **output dtype ULP** (f16 ~4.9e-4 relative; max relL2 ~5e-4 = identical within the representation limit). Self-check: under causal, the tokens in `[0, tile)` depend only on themselves, so if that range is bit-identical the dumps are proven comparable (`pwk4 vs pwk8` block 0 = exactly 0.0000, averaged over 24 layers).
6. **Classify the toggle before believing the result**: for a *bit-preserving* toggle (same values and order: `KV_2D/Q_2D/A_2D/BLOCK_SKIP/DKS_ACTIVE/256GRF`), "no change" is a consequence of a correct kernel and therefore **no evidence at all**. For an *order-changing* toggle (accumulation-order changes such as `KQ_TILE_KEYS/PER_WG_KEYS`), moving an end-to-end metric on a chaos-sensitive model is just a re-roll of the dice. "Only the tile knobs move the score" is the signature of a correct kernel (7 runs wasted on minicpm4).
7. **An end-to-end text metric (WWB) is not a kernel accuracy gate.** On small models it is ±0.05 sensitive to last-bit changes and bimodal per prompt. Use a per-op comparison (gtest) against an exact reference. The user designated the micro value re-measured with `TEST_USE_SDPA_OCL=0` (0.819114) as the reference value.
8. **golden-split test trick** (bidir token_type_ids, memory:sdpa-ocl-bidir-token-type-ids): without new golden data, use "a run that splits the query range in two is identical to a single run" as an equivalence oracle. The reference was also extended to be bidir-aware (`PagedAttentionReference`).
9. **gtest tolerances are loose** -> the main evidence of equivalence is L1/L2/xab/pset. For gtests, diff the **set**, including skips.
10. **Check whether the harness data hides errors** (5.6 #19-21): N(0,0.1) data makes softmax nearly uniform, so Q/K misreads are invisible and only V misreads are large. Use `logit_scale_gain` (scale x128)/`runtime_scale_multiplier`. The output buffer is not reset, so use a **NaN-poison fixture** (fill the output with NaN and rerun, which catches unwritten elements from index 0).
11. **While the tier mask rejects, "PASS with HPG=1" is not sdpa_ocl evidence**: always judge by the `sdpa_ocl` line (>=1) in the dispatch census (`OV_VERBOSE=4`, `Enqueue stage`). Do not count a SKIP/opt fallback as an OCL success.
12. **Value/cost of independent review**: a read-only multi-perspective review (T7: 32 agents found the requantize bug, T8: 12 agents found the degenerate data, a second 3-lens review re-derived the arithmetic) is worth running **before** long builds/dumps. But dedupe first (5.1).

---

## 5.5 Performance-claim discipline and representative results

### 5.5.1 Measurement methodology

| Item | Method | Source |
|---|---|---|
| Kernel time | "Device Performance Timing" of `cliloader -d -dv` (CLIntercept), with `SPILL=` shown on the kernel name line. Run by the user | memory:sdpa-ocl-beats-micro-256grf |
| Bottleneck candidates | Check GPU active/stalled, SBID and barrier stalls with VTune GPU Hotspots, and compare before/after on the same GPU, workload and build. If VTune has no stall PCs, withhold source/ISA line attribution and decide by cliloader device time + correctness A/B | B70 GPU Hotspots records, details in chapter 04 §4.1.1 |
| Clock | B70 **pinned at 2800 MHz** (unpinned, micro max 16458->10000, large variance) | memory:sdpa-ocl-int8-perf |
| Repetitions | For kernels with few calls (prefill 2 per invocation) the mean is noise -> **MIN/median over N runs** (e.g. 8). `-n 1 -ic 4` is 62+2 prefill calls, ~2 min | 〃 |
| Real-model harness | Wrap `benchmark.py -d GPU.1 -m <model> -n 1 -ic 4 -pf <jsonl>` in `cliloader` and `grep -E "sdpa_(ocl\|micro)__prefill"`. e2e 1st token uses `-n 3 -ic 256` | 〃 |
| Interpreting results | SDPA prefill is ~10-15% of the 1st token latency (llama-3.1-8b: 69.8/475 ms) -> a 7% kernel improvement = ~0.9% e2e. **The kernel level is the optimization metric; e2e barely moves** | 〃 |
| Is a value a "total" or "per call" | In the notes, values in the hundreds of millions of ns are the **sum of device time** over the whole run; thousands to millions of ns are per call. Distinguished per table | each note |
| Accuracy | Record a ref-check PASS with every performance A/B. But when the performance-only mode `SDPA_SKIP_REF_CHECK=1` is used, accuracy is a separate gate | kq-tile-keys note |
| xe_hpg performance comparison (S7+ guidance) | On the same Release build/DG2/driver/inputs/cache state, **current OCL vs the current actual micro/opt** (check the actual stage, not the name). OCL/control are **separate processes** that differ only in `TEST_USE_SDPA_OCL`. Distinguish device kernel time / the whole primitive pipeline / E2E. Separate diagnostic runs (dump/verbose) from timing runs. If the variance is larger than the difference, withhold judgement. Static values (ocloc spill, tile, SLM) are not time or actual spill | memory:sdpa-ocl-xe-hpg-s7-performance §4 |

### 5.5.2 Effect of the hardware move (memory:hardware-migration-b580-to-b70)

On 2026-07-05 the development machine changed from B580 -> B70 (both Xe2). **Still valid (hardware-independent)**: ISA instruction mix, layout probe results, builtin availability, the `-device bmg` target, accuracy.
**Must be re-measured**: all absolute ns and ocl-vs-micro ratios. B580 figures (e.g. int8 prefill 1.69x -> 1.45x -> 1.23x) are **history**, and the B70 re-measurements differ (B70: head=64 q=7 pinned mean 1.055x/floor 1.117x; head=128 q=4096 is 1.40x, i.e. the small microbench hid the real gap).
Figures dated before 2026-07-05 are B580, later ones B70 (definite only where the note says so; "estimated" in the table below means estimated from the date).

### 5.5.3 Representative results table

| Result | Before -> after | Hardware/method | Status | Source |
|---|---|---|---|---|
| PA MIXED u4 (llama-3.2-1b) | OCL 145,603 ns vs sdpa_micro 147,825 ns (**-1.50%**); -8.23% vs the first exact baseline 158,661; V_PREFETCH on/off 145,603/151,866 (-4.12%) | User cliloader kernel mean (representative kernel 64 calls, SIMD16 REG128 SLM=17664 GWS[272x512] LWS[16x16]). Hardware not recorded in the note, B70 estimated from the date | MEASURED | memory:sdpa-ocl-mixed-exact-complete |
| plain f16 prefill, llama-3.1-8b q=4096 | micro 1,930,454 -> ocl default 2,181,376 (13% slower) -> **ocl BEST 1,796,200 (7.0% faster)**; e2e 1st token 466.98 -> 462.85 ms | B70, pinned 2800 MHz, per-call device time, 3 passes, variance <0.5%. BEST = 256GRF + tq32/pwk4/pwq2 | MEASURED | memory:sdpa-ocl-beats-micro-256grf |
| DG2 S7a PA PREFILL initial baseline, head 128 / 32:8 / seq 4096 | first device-USM ABBA OCL 110.42–113.27 ms vs micro 4.616 ms (about 24x); follow-up ABBA 126.74 vs 4.543 ms (27.90x) | A770/DG2, cliloader device kernel time; the values come from different runs, so they are not linked into one delta time series | MEASURED, initial prototype | chapter 06 §6.3 |
| DG2 raw K/V standalone PA PREFILL | OCL <= micro × 1.03; 32 block-16 head sizes, 2,642 PASS records / 2,434 unique conditions matching the current source policy, worst +2.6234% | A770/DG2, device USM, same-round paired ratio, queued32 and wait-each. The original K/V are used in one attention kernel, no external pre-pass | MEASURED, kernel-only; product integration and end-to-end not done | chapter 06 §6.3, test/sdpa_ocl_xe_hpg/s7a/perf/opt/CHECKPOINT_20261004_ASM_GATE3.md |
| head=128 int8 compressed prefill q=4096 | k16 default 5,102,600 -> **k16pwg4 3,526,979** (faster than micro 4.17M); sg_per_wg 8<16<32, wgTK 64<128<256, all "bigger is slower" | B70 2800 pinned, per-call min. Accuracy ref-checked only at head=64 (head=128 real data had not been verified) | MEASURED | memory:sdpa-ocl-kq-tile-keys-32-slower |
| causal/window key loop bound | causal_k: 3,421,875 -> 3,300,312 (1.04x, 1.78x fewer tile iterations); window W=256: 2,266,562 (1.46x), micro 2,037,708 | B70 2800 pinned, `*paged*96` q=1024 head128 f16. For causal_k the user confirmed a large improvement on the real llama-3.1-8b model. window has no effect on the real model because llama has no window (SLIDING_WINDOW_SIZE=0 removes the block) | MEASURED | memory:sdpa-ocl-causal-bound |
| gemma-4 head-72 vision prefill | 9,570,971 -> 1,030,869 ns (**9.28x**, 37.28% -> 1.11% of GPU time), from 8.11x slower than micro to 1.14x faster. block2d gate %64->%16 + base fixup (6.96x) + DKS_ACTIVE (1.33x) | Dated 2026-08-14 (B70 estimated), total device time | MEASURED | memory:sdpa-ocl-block2d-gate-relaxation |
| phi-4 vision head-72 padded view | 7.5x slower -> fixed in bd8d90097d | 〃 | MEASURED (note) | memory:sdpa-ocl-block2d-padded-view-gate |
| gpt-oss-20b u4 PA MIXED head 64 | 1,474,574,702 -> 376,221,254 ns (**3.92x**), 6.3% faster than micro 401,436,373. Cause = the 32 B u4 head-64 row misses the block2d 64 B rule -> scalar gather for both K and V (192 `load.ugm.d8u32` / 16 dpas per k0). Fix = whole-page read with `intel_sub_group_block_read_uc16` + retiling | Total device time (B70 estimated) | MEASURED | memory:sdpa-ocl-u4-head64-page-read |
| decode (PA GENERATE) f16 | **-7.79%** vs `pa_gqa_single_token` 1,188,447,305 ns (M=4 -4.65%, +head-dim split S*V -5.78%) | llama-3.1-8b, ctx ~4352, 32/8 heads, head 128; cliloader total device time. Hardware not recorded, B70 estimated from the date | MEASURED | memory:sdpa-ocl-decode-kernel |
| decode gemma-4 (head 512 + SWA256 mix) | pa_opt 718.1M -> sdpa_ocl_decode **525.9M (0.73x)**; **e2e 11.43 -> 11.40 ms (effectively unchanged)** | 1024 tok prompt, 2040 decode tokens. SG_PER_WG beats M, V_TILES>=16 discriminator | MEASURED | memory:sdpa-ocl-decode-tiling-sg-per-wg |
| `pa_kv_cache_update_ref` u4 token-major writer | total 303.5M (regressed) -> **237.1M (-21.9%)**, -14.1% vs upstream d-major 276.0M; prefill -52.2% (probe -52.1%) | llama-3.1-8b B70 head128, 8 kv heads. generate exactly tied with d-major (the probe prediction of -4.4% missed) | MEASURED | memory:pa-kvup-u4-token-major-writer |
| Limit of the writer above | the token-major page dirties 16 sectors per thread vs 2 for d-major, +12%; 4 fix candidates **rejected after measurement**, ceiling 986 ns/dispatch, code unchanged | gemma-4 26b, 2040 tokens, B70. e2e net effect: decode attention -0.185 ms/token vs kvup +0.019 ms/token | MEASURED | memory:pa-kvup-token-major-store-sector-floor |
| int8 plain prefill (B580, **history**) | ocl/micro 1.69x -> Stage-1 1.45x -> Stage-2 1.23x (median 15.1 -> 12.8 us). V transform_8b alone was the first win; then a K-load geometry redesign | B580 head=64 prefill, MIN over 8 runs | MEASURED, **stale on B70** | memory:sdpa-ocl-int8-perf |
| Refactor performance-neutral | R0 157,307 ns -> R2 157,580 ns (+0.17%, noise) | L4 measurement (6ac827ff6f build) | MEASURED | memory:sdpa-ocl-refactor-2026-09 |
| VTune-guided f16 V-read merge | `16r16x2c`: 524.34 -> 521.01 ms (-0.64%), identical MD5; SDPA span 2.9%/5.2% shorter, SBID stall 43.9–45.5% -> 41.5–43.5% | B70 GPU Hotspots + same-workload benchmark. Some collections had 0 stall PCs, so per-instruction attribution is limited | MEASURED | B70 profiling session, 2026-07-31; details and caveats in chapter 04 §4.1.1 |
| PA GENERATE head 64 (u4), `SG_PER_WG` 8 -> 16 | GENERATE device time 0.45-0.51x at 1-6 partitions, 0.83x at 17-18 (lfm2.5, llama-3.2-1b, minicpm4); lfm2.5 2nd token 2.132 -> 2.069 ms (32-token prompt) | B70 2800 MHz, cliloader total device time; llm_bench 12 iterations rotated, P-cores | MEASURED | memory:sdpa-ocl-decode-sg-m-rule |
| PA GENERATE head 128 (u4), new SG x M rule + M round-up | device time llama-3.1-8b 0.821/0.792 (32/1024 prompts / 4K), qwen3-8b 8K 0.767, phi-4-mini 0.818/0.823; 2nd token -0.8..-2.1% | B70, cliloader; llm_bench 3 rotated rounds x 3 iterations | MEASURED (uncommitted) | memory:sdpa-ocl-decode-sg-m-rule |
| lfm2.5-1.2b 2nd token, branch vs reference build | branch +3.5%/+2.7% -> -1.3%/-1.8% after the upstream RMS+RoPE fusion (rebase) and 16 subgroups at head 64 | B70 P-cores, 12 iterations rotated. The +20% reported on Windows was NOT reproduced (host-bound emulation on E-cores: +1%) | MEASURED | memory:lfm25-decode-gap-rms-rope-sg16 |

**Bounds of the DG2 performance results**: the standalone kernel gate of S7a PA PREFILL was measured. Integrating the raw-kernel policy into the product source and confirming it in OpenVINO dispatch/end-to-end has not been done. The DG2 S1 micro-lane Debug accuracy baseline is not a performance comparison. The S6/S7a product regression and ARL-H real-hardware verification also remain.

### 5.5.4 Experiment results table (both adopted and rejected; mean ns, memory:sdpa-ocl-mixed-exact-complete)

| Experiment | on | off | Conclusion |
|---|---:|---:|---|
| First correct GRAN=1 | 158,661 | - | Correctness baseline |
| legacy GRAN=0 | 152,432 | - | **Wrong output** (regression /0,/1) -> not taken as the baseline just because it is fast |
| paired Kc DWORD retention (long-lived private array) | 177,716 | - | About 12% worse, removed. **"No long-lived private arrays"** |
| SV_TRIM | 151,677 | 158,422 | Adopted |
| KQ_FAST | 172,310 | 151,746 | Worse, removed |
| KQ_TRIM | 157,932 | 151,529 | Worse, removed |
| V_PREFETCH | 145,603 | 151,866 | Adopted (-4.12%) |

---

## 5.6 Master pitfall/refutation catalogue

Format: number. Situation -> wrong hypothesis/mistake -> fact/cause -> rule.

### A. Attribution errors (wrong suspect)

1. **T5 head-512 compressed F2 failure** (memory:sdpa-ocl-ki-t5-plain-i8-head512): "prime suspect = the sdpa_ocl V `_8b_32r16x4c` read" (a layout assumption probed only at 64 B width/x=0 on B580). The real cause: `Validate()` in `dynamic_quantize_kernel_opt_kv_cache.cpp` rejects `input_dims.back().v > 256` -> the ref kernel cannot append -> cache corruption -> wrong output at the q=1 step. The fix is 256->512 (upstream PR #38466, not on this branch). **Rule: the evidence proved only "the q=1 step is wrong", and we jumped to "sdpa_ocl is wrong". The first bisection step `TEST_USE_SDPA_OCL=0` was not run; check the cache producer before suspecting the consumer kernel.**
2. **minicpm4-0.5b WWB -0.070** (memory:minicpm4-wwb-chaos-not-a-bug): a non-monotonic discriminator that was bad only at `pwk==4` (2 ok/4 bad/8 ok) was misread as a bug. In fact L0 <=1.4 ULP, amplified over 24 layers. Rules 5.4 #5-#7. **Do not re-investigate.**
3. **head-486 CL_OUT_OF_RESOURCES** -> one-sided toggle control (5.4 #1). In fact suite memory pressure.
4. **tile_keys=32 wrong output (C7)** (memory:sdpa-ocl-tk32-bug-hunt, **unresolved**): on `basic/31` (head 64 prefill q=1024) it FAILs **only when `tk=32 && pwk>=4 && kq_query_blocks>=2`**, wrong from the second query block on (index 35264 = token 275, WG-local q 19), tokens 0..274 correct. All memory-path toggles/256GRF/BLOCK_SKIP/tiling invariants/S_slm/Q_slm static checks were ruled out. **2026-09-25: the same repro no longer fails -- masked since PA prefill uses MICRO_MATH=1 (hypothesis; the plan for a `SDPA_OCL_MICRO_MATH=0` arm is at the end of the note)**. The note itself also draws the distinction: "do not confuse with minicpm4: here a gtest against an exact reference fails deterministically at the same index, so it is a real defect". **override-only; `kq_sg_tile_keys` other than 16/32 is an `#error`. Do not ship 32 as a default.** Side correction: the note's old S_slm model was wrong (a subgroup block read is **strided**: component i of lane L = `p[i*sub_group_size + L]`; a coverage/conflict analysis cannot tell the two bijections apart).
5. **SLM-occupancy hypothesis (why kq_sg_tile_keys 16->32 is slower)** (memory:sdpa-ocl-kq-tile-keys-32-slower): the ISA ruled out spill/total inst/sync/V-load latency/dispatch/dpas parallelism and pointed at "SLM 17.5->25.7KB, so occupancy 7->5 WGs", but **a device measurement refuted it**: in reality two independent axes, the workgroup size (sg_per_wg) and kq_wg_tile_keys, both "bigger is slower". The key axis is the reduction axis, so more subgroups add atomic max/barrier/SLM round-trips and no useful work. The WG count (2048) is the same for all configs, so it is not an under-fill story. **A static model must be validated by a hardware measurement.**
6. **fewer tiles != less time** (causal-bound): causal_k cut the tile iterations 1.78x but the time only 1.04x; window cut them 1.71x for 1.46x. At 1024 WGs/sg_per_wg=16 the kernel is dominated by reduction overhead, not work-bound (this was an open puzzle, later resolved as per-iteration overhead: ocl 2112 k0 iter/head vs micro 272 (7.76x)).
7. **The intuition "ocl is slow because of kernel efficiency"** (causal-bound): in fact ocl had no causal bound and did **1.60x more work**, and per unit of work it was already 1.41x faster than micro. Before comparing efficiency, match the **amount of work** (KQ macs/head, causal efficiency).
8. **The spill number is a symptom** (gate-relaxation): 256GRF took the spill 2688->0 but was 47% slower (LSC message count, halved resident threads). **REG256 is not free**: 25% slower with the default tiles (2.18M->2.73M); it pays off only when tuned **as a pair** with the large tiles that previously spilled.
9. **ocloc spill predictions are useless** (beats-micro-256grf): 4 candidates with spill=0 at `-device bmg` spilled 3.3k-19k B at runtime. Choose candidates **by runtime measurement only** (cliloader `SPILL=`). `T128k256` has the fewest k0 iterations but is 2.2x slower due to spill -> "the k0 iteration count is not the objective function".
10. **"Fewer reads make it faster"** (int8-perf): 3 attempts at cutting the number of V reads failed. The bottleneck the ISA showed was the **load geometry** (SIMD-1 scalar scale/zp + medium-size K-data messages), not the count. cp-pair read reuse halved the reads and was still **slower**, 22,812 vs 21,666 ns (`if(cp&1)` + the cross-loop `v_pair` disturbed IGC scheduling).
11. **K 2D-block load experiment (NEGATIVE RESULT)** and the toggles (`K_I8_VARIANT`, `V_SCALE_CACHE` no gain: L1 already absorbs the reloads, `K_DIAG`) were all deleted in the cleanup phase, keeping only the V transform win. The last gap is widening+zp broadcast, the floor for hardware OpenCL (micro avoids it with the gemmstone register schedule).
12. **block-level causal-mask skip** (beats-micro-256grf): mimicked micro's `if (causal_k_end > causal_q_begin)` and confirmed correctness by brute force over 9.3M elements, but **IGC flattened the branch instead of taking it** (cmp 20->36, sel unchanged, net inst +16). The ceiling is also small (mask `sel` 17/35, the remaining 16 are the fmax reduce). Removing the `k_mask` remainder add also only turned add 60->20 into mov 59->82, -3 inst.
13. **MIXED experiments** (5.5.4): paired Kc DWORD retention, KQ_FAST, KQ_TRIM all worse. "Do not conclude from REG128 alone that it caused the spill/slowdown."
14. **U4 token-major writer**: generate, for which the probe predicted -4.4%, was exactly tied. **sector floor**: 4 candidate fixes were measured and all rejected, ceiling 986 ns. Recording only the conclusion without a code change is also a valid closure.

### B. Correctness/gate holes

15. **The `use_ocl` hole in the mixed layout gate** (memory:pa-mixed-layout-gate-use-ocl-hole, fixed by T1 f81a75eef6 (formerly 96b457a4f0)): the "what does the page look like" check in `paged_attention_opt.cpp` was guarded by `use_ocl`, which means "which kernel reads it", so with `TEST_USE_SDPA_OCL=0` micro MIXED read a token-major page. **B70 blind spot**: on Xe2 the env and the staged backend always agree, so a remaining `use_ocl` read passes every B70 test (T2).
16. **by-channel layout defaulted to token-major without checking the reader** (T1): `by_channel_token_major_readable()` uses `sdpa_ocl_selected` as a proxy for "a decode reader exists" -> widening the predicate creates token-major pages on xe_hpg and the d-major GENERATE reader throws (predicate split in S3).
17. **A plain `inline` helper changes the whole kernel's ISA** (memory:igc-inline-helper-pipeline-switch, ocloc 26.22.38646.4 bmg): the pipeline changes even though IGC inlines the helper. **Only `__attribute__((always_inline))`** gives ISA identical to the hand-written code. Rules R7-R12: every helper is `SDPA_OCL_INLINE`; the lane is passed as `size_t lane` + `int lane_i` (no `(int)lane` inside a helper); no branching inside a helper on an argument that becomes a constant in some configs; no array declarations inside a helper (declare at the call site and pass a `__private` pointer); pass leaf variables for coordinates and compute them inside the helper; loop-carried outputs are call-site variables (out-param). `bidir_query_groups` stays inlined (C at gate0). Isolating a non-A cause: snapshots that apply the call-site hunks one at a time -> L2 on the target configs only -> a normalized per-IR-stage diff with `compile_one(keep_dir=)` (`_beforeUnification -> _afterUnification -> _optimized -> ISA`).
18. **A tiling-invariant violation is silently wrong output** (memory:sdpa-ocl-tiling-constraints; details -> 01-dpas-and-tiling.md): changing only `kq_sg_per_wg_keys` by env breaks the sg_per_wg invariant (number 3), so non-existent subgroups own value columns and the output is not written. The "fast-looking" timings of that sweep were invalid.
19. **PA harness data N(0,0.1) hides Q/K misreads** (memory:pa-harness-data-hides-qk-errors, T4 "WRONG earlier claim"): logits ~0.008 -> uniform softmax -> ref std ~0.005, and the tolerance 0.025/0.075 **accepts an all-zero output**. My claim that the distribution was U[-1,1] was wrong (`generate_input_data` has no callers; it is actually `generate_realistic_data` mt19937(1234) N(0,0.1)). The output buffer is `reset=false`, so it holds 0 or stale data. So the 24 `MISSING -> PASS` in T4 would have proven nothing -> only the NaN-poison fixture + the f16 U2 pair (tolerance 0.002, 70-90% of elements exceed it) are value-based detectors.
20. **Degenerate LCG data** (memory:gtest-lcg-power-of-two-degenerate-data, T8): `InputGenerateData(start, range, resolution, seed)` with `range*resolution` a power of 2 <= the row length has a period equal to the head size, so **every row is identical** -> identical logits -> softmax ignores the scale -> passes even on the old code. `(-1, 2, 32)` + head 64 is exactly that case. Replaced with resolution 31 (logit spread ~14, misread error 0.85 vs the bf16 threshold 0.025). **Replay the generator offline before trusting a test.**
21. **T4: SWA 241..256 output not written** (027a9a95ae): decode chose tmp_out/output with the **full seq_len** while finalization chose by the SWA-valid length, so the output was not written. The fix was confirmed by an exhaustive integer model (SWA 0..699 x seq 1..2099, 10.2M cases) with 0 violations. No known model has a window in (240,256] (gpt-oss 128, Gemma-3 512/1024, Mistral 4096). Latent coupling: decode is gated by `SLIDING_WINDOW_SIZE != 0`, host/finalization by `SWA_BLOCK_SKIP_ENABLED` (consistent because decode rejects scores).
22. **T7 block2d padding** (325f47dfa5): `block2d_layout_ok` did not look at padding. B70 **measured rule**: wrong if the token stride is not a multiple of 16 B (260/264 B), wrong if the first head start is off by 2 B, fine for a base misalignment of 4/16/32/48 B (stride a multiple of 16 B). The gate also keeps the documented 64 B/16 B base rule (spec + the old B580 phi-4 experiment; B580 may be different silicon). Residual: **dynamic** K/V padding cannot be proven by the host (4 DISABLED residual suites FAIL). Side finding (not fixed): the i8 BY_CHANNEL requantize at `pa_kv_cache_update_ref.cl:297` ignores `in_data_pitch` (misreads new tokens j>=1 of a padded key; narrow trigger: i8 BY_CHANNEL + padded key + past_len%16!=0). The fixup tier's rank-4 `axis_unpadded(X)` check was vacuously true at rank-2.
23. **T8 SCALE_DATA_T=half** (8300f14d92): the doc called it "latent", but it was live for a bf16 runtime scale (F1 compiles 400 bf16 runtime-scale sdpa_ocl kernels, which read the scale as half). After the fix, the existing `sdpa_opt.cl:154` PA scale bug was found (`SCALE_TYPE = INPUT3_TYPE` reads a runtime f16 scale as int32; unreachable in real models because the PA scale is a Constant), not fixed, documented. The reason for the multiplier 64: N(0,0.1) hides the scale.
24. **8b transform / block2d hardware rules** (→ 02-memory-io-prefetch-barriers.md; memory:sdpa-ocl-8b-transform-32row-min): the 8b transform is 32-row only; our helper is stricter than the spec. The `_32r16x4c` layout assumption, probed only at 64 B width x=0, was never verified at head 512 (it turned out to be irrelevant, #1).
25. **plain int4 / bf16 PA / jitter**: plain int4 has no nibble unpack at all (compile only, never dispatched); the plain i8 bias trick is currently correct (`kv_cache_compression.cpp:244` zp_dt = immad && !int4 ? i8 : query type); the `jitter.hpp` dup-macro assert was lost (T9 P2, not started). F2 bf16 compressed is a dynamicquantize layout failure, and `SDPAFusion/0` is the upstream 3D-XMX decomposition rule (T10: not sdpa_ocl).
26. **sink input**: the 30 `TEST_USE_SDPA_OCL=1` failures were the `token_type_ids` gap + 1 unexplained build failure (memory:sdpa-ocl-sink-input).
27. **`sg_ij` uniformity** (side finding from minicpm4, not fixed): micro uses `sub_group_broadcast(get_local_id(1), 0)`, sdpa_ocl the raw `get_local_id(1)` -> subgroup block read/write addresses and broadcast indices must be subgroup-uniform. `NUM_HEAD_SIZE_GROUPS` (`pa_kv_cache_update_ref.cl:227`) is an unparenthesised macro (the current uses are safe by accident; it truncates to 1 at head 320).
28. **Refactor scope verdict**: prove in units of "observable behaviour", not code changes. SHA instead of `HEAD`, no snapshot `--force`, and `corpus <dir>` **deletes** `<dir>/configs`, so rerunning r0/r1 with a different base destroys the proof reference point.

### C. Tool/process pitfalls are in 5.3 (E1-E16) and 5.4

### D. Attribution errors found on 2026-10-22 (numbering continues after #28)

29. **lfm2.5 2nd token slower than the reference build** (memory:lfm25-decode-gap-rms-rope-sg16): the suspect was the branch's kernels or host code. A diff of the most common per-decode-step enqueue sequence (LD_PRELOAD shim) showed 196 vs 184 kernels: 12 extra per-head q/k `rms` kernels, because the branch base predated the upstream FuseRMSRoPE pass. Controls on both sides: the reference with `OV_GPU_ENABLE_RMS_ROPE_FUSION=0` reproduced the branch's sequence, and the branch with the pass ported and its decode kernel disabled produced the reference's exact MD5. The host-API census per token was otherwise identical. **Rule: for a branch-vs-reference gap, diff the per-step kernel sequence and the host-API census before touching kernels; a missing upstream commit is a suspect.**
30. **"16 subgroups make head-128 decode 0.8x"** (memory:sdpa-ocl-decode-sg-m-rule): the SG knob also moved M from 2 to 4 through the register estimate (u4 reads four K tiles per message). Separate SG x M sweeps showed that 16 subgroups alone are 1.18x slower at 4K when M cannot grow (phi-4-mini). **Rule: after setting a knob, read every derived parameter back from the dispatch (LWS/GWS/SLM) before attributing.**
31. **"i8 BY_CHANNEL cache"** (memory:pa-kv-cache-u4-default-4bit): every measured model had 4-bit weights and therefore a u4 cache. A head-64 rule measured on u4 (16 subgroups 0.45-0.51x) was shipped for all caches and made i8 head 64 1.26x slower at 4K. **Rule: state the cache the model actually uses, and measure each cache type before generalising a tiling rule.**

---

## 5.7 xe_hpg (DG2/ARL-H) porting effort — S0-S9

Detailed playbook: `.claude/skills/xe-hpg-porting/SKILL.md`. The original plan is in the memory notes `sdpa-ocl-xe-hpg-plan` (hub), `-facts`, `-s7-performance` (top-priority supplementary directive, approved 2026-10-02), `-briefs-s2-s9`, `-plan-full`; working copy `test/sdpa_ocl_xe_hpg/plan/`. Hardware: DG2 is a separate PC (raptorlake-02), B70 is the development PC (raptorlake-01).

### 5.7.1 Stage table (as of 2026-10-06)

| Stage | Content | Status | Current branch commit (old hash in memory -> matched by title) |
|---|---|---|---|
| S0 | Hardware probe (DG2 compile+ISA, no enqueue) | DONE 2026-09-30, DG2 PASS rc0 | `test/sdpa_ocl_xe_hpg/probe/S0_RESULTS.md` |
| S1 | DG2 baseline (micro lane, HEAD 36fb814999) | DONE: U1 0/325/112, U2 0/287/108, U3 0/195/64, F1 133/2066/0, F2 15/101/7, SEL 18/18; UNEXPLAINED=0 | data on the DG2 PC |
| S2 | SG8 prototype (DPAS mapping H1, tiles, mini SDPA, K direction, DM spike) | DONE: H1 8/8 identity, NC 24/24 | `s2/S2_RESULTS.md` |
| S3 | Predicate split + fake-device tests | DONE | `5af20d13f1` (formerly 01bcf08592) |
| S4 | SG16-neutral refactor (key counts in `DPAS_K` units) + Xe2 invariance proof + tripwire | DONE: L1 same 2491 + modes 8937, ISA SAME 92+162 | `976fbf4b0c` (formerly 7940e525ca) |
| S5 | xe_hpg host jit (2D off, SLM/WG checks, tier mask, TEMP routing, `OV_GPU_ARCH_OVERRIDE`) | DONE | `a0c60df997` (formerly 0ae73cfa32) + `1799d40f79` (formerly d06fc2d3d6) + `79b856f497` (guard) |
| S6a | plain f16 static prefill SG8 core | DONE (confirmed by the user) | `10e4f00b68` (formerly 050605239a) |
| S6b | bf16, mask, causal, sink, dynamic, q<=1 | DONE | `087d2bd171` (formerly 27304008a1) |
| S6c | plain i8 KV | DONE | `b739dc5881` (formerly f972cc2d80) |
| S7a | PA PREFILL + entry performance checkpoint | **standalone raw-kernel 3% gate PASS (2026-10-04); product integration, dispatch/end-to-end and S6/SG16/B70 regression not done** | [chapter 06 §6.3](06-performance-gap-investigation.md), `test/sdpa_ocl_xe_hpg/s7a/perf/opt/CHECKPOINT_20261004_ASM_GATE3.md` |
| S7b | PA MIXED f16 + scalar Kc/Vc | Not started | - |
| S7c | PA features (sink, bidir, qq_bias, window, k!=v, runtime scale) | Not started | - |
| S8a/b/c | i8 BY_TOKEN / d-major i8 BY_CHANNEL (DM reader) / u4 | Not started | - |
| S9 | Default flip (DG2 + ARL-H 12.74, cache tag, docs) + 2 DG2 census runs + real models | Not started | remove the 2 `TEMP(S9)` sites |

In the committed HEAD `d20cac7`, `kHpgTiersReady` is `PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8`. The current uncommitted working tree adds a `PA_PREFILL` bit, separate PREFILL/MIXED routing and a `KV_TILED` pre-processing prototype, but this is not the vISA raw kernel that passed the standalone gate, and it has not been verified by a product build, accuracy or performance. Do not interpret this local change as tier readiness. In the committed state `TEMP(S9)` sits at exactly 2 places, `sdpa_opt.cpp:86` and `paged_attention_opt.cpp:1623`. Safe stopping point: **xe_hpg default behaviour unchanged until S9** (`TEST_USE_SDPA_OCL_HPG` off).

### 5.7.2 Performance policy after S7 (approved by the user 2026-10-02, memory:sdpa-ocl-xe-hpg-s7-performance)

- Performance priority **PA PREFILL > PA MIXED > plain SDPA q=1**. Numeric pass lines and the "accept the regression and only record it" policy are abolished. Each performance issue gets one of **FIXED / REFUTED / ACCEPTED·DEFERRED with the user's explicit approval**, together with the rationale, the limits and the next revisit condition. Structural/accuracy blockers are not closed as performance deferrals.
- Common SG8 config/tile/GRF/reader tuning is allowed even if it affects the completed S6 plain/q=1, but checking the affected regressions is mandatory. B70/Xe2 hardware regression and the compiled-kernel/L1/corpus/pset invariance proofs are not waived on the grounds that the work is DG2-only.
- Per-stage loop: (1) confirm accuracy/the actual dispatch -> (2) low-cost static check (reuse the existing corpus) -> (3) product-path screening of the approved representative cases -> (4) on a trigger, a controlled experiment with one change at a time -> (5) re-verification/user disposition. Do not automatically skip a measurable risk as "not a gate". If there is no tool, write "not measured/cannot judge".
- Example investigation triggers: new/increased spill, long-lived private arrays, duplicate gathers in a hot loop, abrupt changes at size/page/small-query boundaries, a faster kernel but a worse PA overall/E2E, the actual lane changing because the host rejects.
- **Meaning of the S7a follow-up result**: on 2026-10-04 the raw K/V standalone kernel passed the user-set gate OCL <= micro * 1.03 at 32 block-16 head sizes. That is the result of that measurement matrix, not a completed product replacement, a guarantee for every supported input, or a speed guarantee on other Xe devices. Tools, root cause and scope: [chapter 06 §6.3](06-performance-gap-investigation.md).
- `prof_us` of `sdpa_perf_hpg.py` is a cumulative primitive mean that includes warmup, in integer us -> do not call it a warmup-excluded median. gtest duration/XML time is wall time that includes compile/reference, not performance evidence.
- For every run, the executor/workload/budget/arm order/warmup/repetitions are **approved individually**. All new kits/logs go under `test/sdpa_ocl_xe_hpg/`.

### 5.7.3 DG2 real-hardware facts summary (MEASURED, memory:sdpa-ocl-xe-hpg-facts, S0/S2/S1)

| Item | Fact |
|---|---|
| Hardware | A770, device 0x56a0, IP 12.55.8, driver 25.13.33276.16; SG 8/16/32, SLM 65536 B, max WG 1024, 512 EU x 8 thread; OV `GPU.1` = OpenCL HWQ `0` |
| SG8 DPAS | f16/bf16 x M=1/2/4/8 compile successfully (`dpas.8xM`, hf/hf, bf/bf). A is `int/int2/int4/int8`, B `int8`, C `float..float8` |
| SG16 `short8` A | **Compiles without error, but the ISA has no DPAS** (no input loads either, only the stores remain, 128/128 results 0). The "compile failure" prediction was wrong -> **silent garbage output**. All SG8 work must check the `dpas` count |
| lane mapping (H1) | A: lane l = K(2l, 2l+1), low half = 2l. B: dword j = K(2j, 2j+1), lane n = column. C: lane = column, component m = row. 8/8 identity, permutation control NC 24/24 match |
| 2D block IO | read/write/prefetch all rejected at compile + extension not advertised (`cl_intel_subgroup_2d_block_io` is xe_hpc+ only). Subgroup buffer prefetch is also rejected. The 2D pragma only warns and leaves the macro undefined -> **cannot be turned off with `#ifdef`; it must be turned off with a jit switch (`block2d_io_allowed`)** |
| block IO | global uint/ushort block IO OK, **local uint/ushort block IO OK without a pragma**. `block_read uint` is wrong at an odd row pitch (err 49; the ushort-pair fallback is correct); `block_read_us` at a 2-byte offset is also wrong -> the contiguous K reader uses block_read only when the pitch is even + 4 B aligned |
| GRF | 32 B (Xe2 64 B): 128GRF = 4 KB/thread. `-cl-intel-256-GRF-per-thread` numGRF 128->256 (half the threads). Correction: "float8 C is 1 GRF on SG8" is wrong (8 GRFs of 32 B) |
| Tile/GRF choice (S2, mini SDPA bench) | **256GRF, 16x16, 4 sgs along the key direction (sg4x2 3353 us) spill 0**; 128GRF spills 3.8~11 KB in every configuration; k16q32/k32q32 spill even at 256GRF. K direction D=128 t_med: K0 contiguous 286 us (the note does not record whether it was adopted; K1 adopted), **K1 scalar gather 365 (adopted)**, K2 vload8+pack 370, K3 SLM transpose 449, K4 A=Q/B=K 344 (KQ only, not adopted). Limit: a small prototype, not a proof of optimality for the full product PA |
| split-matrix MAD | DG2 only (not on ARL-H) -> not used as a required path |
| tail rule | if `db*16+2*idx >= D`, set the dword to 0; guarding **only one** of K or Q is enough (wrong if neither is guarded) |
| F1 DG2 baseline failures 133 | all unrelated to sdpa_ocl: bf16 head 486/512 `CL_OUT_OF_RESOURCES` 128, bf16 runtime-scale 4 (also fail alone), f16 runtime-scale 1 (**depends on the order of earlier tests in the process**, reproduces with `=0` too) |
| f32 PA | rc=134 abort (`CL_OUT_OF_RESOURCES`) after 14 OK |
| DG2 micro-only pass list | U1 24 = micro_sdpa_prefill 13 + u4_mixed_micro 5 + update_shape 1 + sink {0,1,2,3,6} 5. **Before the S9 switch all of them must PASS on the ocl lane, otherwise it is a regression** (sink 5, prefill 13 -> S7a/S7c; u4 5, update_shape -> S8c) |

### 5.7.4 Tier table and routing rules

| Bit | Target | Status |
|---|---|---|
| PLAIN_F16_STATIC (1<<0) | plain f16, static, q>1, no mask/causal/sink/runtime scale | READY (S6a) |
| PLAIN_EXT (1<<1) | bf16, mask, causal, sink, runtime scale, dynamic, q<=1 | READY (S6b) |
| PLAIN_I8 (1<<2) | plain i8 KV (`supported()` rejects u4 and i8 without scale/zp) | READY (S6c) |
| PA_PREFILL (1<<3) + PA_MIXED_F16 (1<<4) | The committed host path handles the two PA stages together. The working tree has PREFILL-only routing and a `KV_TILED` prototype, but they are unverified and are not the product integration of the standalone raw-assembly gate. The S7a product gate is not done | S7a/S7b |
| PA_FEATURES (1<<5) | sink, tti, qq_bias, window, k!=v, runtime scale | S7c |
| PA_I8_TOKEN/PA_I8_CHANNEL/PA_U4 (1<<6/7/8) | compressed PA | S8a/b/c |

Routing pitfall: if `sdpa_ocl_selected` is true but `supported()` is false, the result is `none` (the opt kernel) -- it does not fall through to the micro tail (`paged_attention_opt.cpp:1615-1620`). That is why S5 uses `TEMP(S9)` to send ops rejected on xe_hpg back to the micro lane. `add_stage` swallows codegen exceptions, so an SG8 build failure is **silently demoted to opt** -> confirm with a census that the stage was added (the `sdpa_ocl` dispatch line). `SDPA_OCL_NEG_SG8=1..6` are negative controls that deliberately break the SG8 mapping (the sharp-softmax test must FAIL; `=4` forces the unaligned-K fallback and must PASS).

### 5.7.5 What can and cannot be done on B70

- **Cannot**: verify SG8 execution (B70 min SG=16, the iGPU has no XMX) -> real hardware means DG2. Upstream CI has no DG2.
- **Can**: host routing/jit generation tests, `OV_GPU_ARCH_OVERRIDE=xe_hpg` (ENABLE_DEBUG_CAPS only) + `SDPA_OCL_HPG_TIERS=all` dumps -> offline compile with `sdpa_ocl_ab.py hpg --device dg2 --grf256 [--define SDPA_OCL_SG8_ARM_READY]`, byte-identity proof of the SG16-neutral refactor (S4). **Do not interpret gtest results run on a faked arch; only the dumped sources mean anything.**
- The jit identity between faked-arch dumps and real DG2 dumps was planned to be checked with `pset` in S6a.
- pass1/pass2 detectors: looking only at first_error masks things, so 2D leaks are caught with the TSV `any_2d` column + pass2 (`ARM_READY` mutation, 2D tripwire 38/38). Offline `ab.py hpg` compile time per 1 run: a few seconds to 25 s.

---

## 5.8 Project timeline (git log + memory; hashes refer to the current branch `sdpa_ocl_dpas`)

Many of the old hashes the memory points to (e.g. `235639f1a2`, `1706b74354..0653092a5a`, `96b457a4f0`, `7940e525ca`, `050605239a`) are not contained in the current branch after the branch was reordered (`git branch --contains` returns nothing). Below are the current hashes whose titles match.

| Date | Stage/commit | Content |
|---|---|---|
| 2026-06-17~24 | `0ba4ebfcd0`, `864b2de687`, `1bc17f9186`, `27bd824c9b`, `2d9038bf46`, `feb3a58836` | first `sdpa_ocl.cl`: mha, head 128, 2D mask, cooperative Q fetch, config table + `choose_config`, GQA |
| 2026-07-05~07 | `13f7e52c8a` k block2d transform read | B580 int8 K-load redesign (1.69x -> 1.23x). Moved to B70, re-measured on B70 |
| 2026-07-13~30 | (memory) | kq_sg_tile_keys 32 investigation -> found the two axes sg_per_wg/wgTK; causal_k + window_k0_begin bounds (real llama model improved); 256GRF + tq32/pwk4/pwq2 overtook micro by -7% (07-30); `90a05f180f` V_I8_MULTIBLOCK (07-28, later a suspect for head 512, then judged unrelated) |
| 2026-08-08~12 | `53ba9bf880`, `ccb0a0ab81`, `dd55fce6b3`, `01b04aa769`, `ae1908b4b8`, `e1249f0164`, `3cf4c60692` | int8 by-token PA, new `sdpa_ocl_decode` (PA GENERATE, f16 -7.79% vs pa_gqa), int8 token-major BY_CHANNEL, int4 KV |
| 2026-08-13 | `d060b84ddc` (rename, default), `a1effb89ad` (kvup) | `sdpa_ocl` default lane, u4 head-64 whole-page read (3.92x), token-major u4 writer (-21.9%) |
| 2026-08-14 | `6f78fd496e` | block2d %64->%16 + base fixup + DKS_ACTIVE: gemma-4 head-72 9.28x. Same week: `365997e7b1` (08-18) decode head 72 |
| 2026-08-19 | `3705c3b124`, `8a27746c9c`, `1981d0d961` | bidirectional token_type_ids (prefill, MIXED, empty gate) |
| 2026-08-23 | `5cf2d5fc5a` | MIXED reads the current token from Kc/Vc |
| 2026-09-09~11 | (memory) | exact u4 MIXED (SV_TRIM/V_PREFETCH defaults): 145,603 vs micro 147,825; minicpm4 chaos conclusion (09-11) |
| 2026-09-10~21 | `e6450198b6`, `bda92b80af`, `6b0e442b09`, `c5b2811f54`, `b0dd980faa`, `78a552b414` | k_head != v_head, micro_math softmax, wide micro math, bf16, scalar runtime mask + unaligned head, qq_bias |
| 2026-09-23~24 | `9694bd61ef`, `142706ebd5`, `53296f18b0`, `2ce502f5ce`, `1677e8f28f`, `b72dc44907`, `631e0ce1bb` | Xe2+ allowed, **behaviour-preserving refactor** (legacy removal, header split, comments+docs, K/V helper extraction, host jit restructuring, known-issues doc). Code 6431->5179 lines (-19.5%), `sdpa_ocl.cl` 2767->996. L0/L1/L2/pset/xab proof per step |
| 2026-09-25~29 | `ad6e56dae9` T3, `027a9a95ae` T4, `f81a75eef6` T1, `f3167294bf` T2, `325f47dfa5` T7, `8300f14d92` T8, `270ba49532` T11 (comments) | known-issues fixes and their effects (5.6 B). T5 split off as upstream PR #38466. Closed for now by the user's decision on 09-29 (T9/T10/T11-perf/tk32 on demand) |
| 2026-09-30 | `5af20d13f1` S3, `976fbf4b0c` S4 | xe_hpg plan: predicate split, SG16-neutral refactor + Xe2 invariance proof. S0/S1/S2 on the DG2 PC |
| 2026-10-01 | `a0c60df997`, `1799d40f79`, `79b856f497`, `10e4f00b68`, `087d2bd171` | S5 (host jit, tier mask, TEMP routing), S6a (plain f16 SG8), S6b (bf16/mask/sink/dynamic) |
| 2026-10-02 | `b739dc5881` S6c | plain i8 KV SG8. S7 performance guidance approved |
| 2026-10-03~04 | S7a standalone PA PREFILL investigation | Initially OCL was 24–28x slower than micro. Separated the spill/reload, V gather/packing and loop-guard costs with VTune/GTPin/IGC ISA and matched source-injection A/B. The vISA inline-assembly raw K/V single attention kernel gate PASSed at 32 block-16 head sizes with +2.6234% worst. Kept distinct from the separate unverified `KV_TILED` prototype and from product raw-kernel integration; the product dispatch/end-to-end/regression gate is follow-up work |

---

## 5.9 Checklist extracted from this methodology (the skill `ocl-kernel-ab-methodology` condenses it into a playbook)

1. First classify the kind of change (behaviour-preserving / bug fix / host-only / performance / new feature) -> the proof recipe in 5.2.6.
2. State the hypothesis in one line, write the predicted value as a number, design **controls on both sides**, then run (one variable at a time).
3. Run the negative control **before** the fix to see the gap between prediction and reality first.
4. Check the setup: `--device_suffix`, binary, Debug/Release, embedding confirmed with `built`, clock pin.
5. Oracle: first layer, dtype ULP, bit-preserving vs order-changing classification, NaN-poison, non-degenerate data (check the logit spread).
6. Performance claims: metric (kernel device time vs e2e), pinned clock, MIN/median, matched arms from the same source/build, measured/assumed labels, re-measure after a hardware move.
7. Closure: memory update + resume prompt + a proposed one-line commit message (the user commits).
