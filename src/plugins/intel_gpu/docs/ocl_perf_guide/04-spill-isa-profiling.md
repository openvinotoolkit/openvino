# 04. Register spill, GEN ISA analysis, profiling

> Full per-tool commands (cliloader/VTune/GTPin/harness) and the interpretation limits of the DG2 cases are in [chapter 07](07-profiling-tool-cookbook.md); inline vISA is in [chapter 08](08-inline-visa-asm.md).

Scope: register pressure / spill (scratch, TPM) diagnosis, GEN ISA dumping and reading, the 256 GRF trade-off, A/B without rebuilding (ocloc splice), IGC pipeline traps (`inline` vs `always_inline`, unroll), the kernel embedding path, microbenchmarks, cliloader usage.
(→ 01-dpas-and-tiling.md: tile selection / 02-memory-io-prefetch-barriers.md: memory messages / 03-numerics-softmax-quantization.md: dequant numerics / 05-methodology-and-pitfalls.md: measurement methodology)

Notation: **MEASURED** = a measurement with the hardware and source stated, **ASSUMED** = hypothesis/estimate. Hardware abbreviations: B580 = Arc B580 (Xe2), B70 = Arc Pro B70 (Xe2, `ocloc -device bmg`), DG2 = Xe-HPG, iGPU = TGLLP etc.
All numbers come from the repo memory notes (`~/.claude/projects/.../memory/*.md`), `test/*.md`, the kernel sources and `docs/sdpa_ocl.md`. They were not re-measured.

---

## 4.0 One-page summary

1. **Spill is often a symptom, not the end result.** In head-72 prefill, SPILL=2688 B was a by-product of 128 scalar gathers; changing the load method gave spill 0 + 6.96x faster (B70 MEASURED, `test/sdpa_ocl_head72_analysis.md` §2.7). Conversely, turning on 256 GRF to reach spill 0 made it 47% slower.
2. **The ground-truth metric is `SPILL=` / `TPM=` / `SLM=` / `REG` on the runtime cliloader kernel-name line.** `ocloc` offline spill is exactly right in some cases (head-72: 2688 B matches) and wrongly 0 in others (tiling sweep, 4 cases). Choose candidates only by runtime measurement.
3. **Static ISA and runtime timing play different roles.** The ISA dump shows the machine-code structure; cliloader device time judges performance. Attributing by removing a specific operation with a code toggle can change what is being observed, because IGC reschedules/reallocates the whole program (K_DIAG=2: removing dequant made it slower, 28.1 -> 32.9 us, B580 MEASURED).
4. **A/B without rebuilding**: compile the runtime-dumped jit prelude (`OV_GPU_DUMP_SOURCES_PATH`) + the working-tree `.cl` with `ocloc` (`test/splice_head72.sh`, `test/isa_ab_*.sh`, `test/sdpa_ocl_ab.py`). The ISA was confirmed byte-identical to the runtime IGC dump (§4.6).
5. **A plain `inline` helper changes the IGC optimization pipeline itself.** Only `__attribute__((always_inline))` gives ISA identical to the inlined code (§4.8).

---

## 4.1 Kinds of spill and terminology

| Term | Where it shows up | Meaning |
|---|---|---|
| spill (fill/store) | IGC `.asm` header `//.spill size N`, `//.spill flag store`; cliloader `SPILL=N` (runtime kernel-name line) | bytes the register allocator could not fit in GRF and moved to scratch (= thread private memory) |
| private memory / TPM | `.asm` header `//.private memory size N`, `.zeinfo` `*private*/*scratch*`; cliloader `TPM=N` | **A private array in the source is allocated in scratch.** It can be caused not by a register shortage but by *accessing a private array with a runtime index* (§4.9-A). In that case it is separate from spill |
| numGRF | `.asm` `//.thread_config numGRF=128|256`, cliloader `REG128/REG256`, `.zeinfo grf_count` | GRFs per thread. 128 vs 256 trades off against thread/EU occupancy (§4.5) |
| SIMD | cliloader `SIMD16` etc. | subgroup size |
| `-abortOnSpill 4` | `.asm` `//.full_options` line (present in every dump) | IGC vISA option. **Do not judge spill with `grep spill`**: every dump matches because of this option string (the old `isa_ab_mixed.sh` counted it; `docs`/ab harness note). Read the spill size from the `//.spill size` header and `.zeinfo` |

It can also be queried with `CL_KERNEL_SPILL_MEM_SIZE_INTEL` (clGetKernelWorkGroupInfo); on DG2 the behavior was confirmed with the conditions 128 GRF > 0, 256 GRF < 128 GRF (`test/sdpa_ocl_xe_hpg/s2/t6.cpp:167`, P7 PASS, DG2 MEASURED).

### 4.1.1 VTune GPU Hotspots: actual use and interpretation limits

VTune was used for analysis on a real B70 (Xe2, `GPU.1`, PCI device `0xe223`). The following are observations confirmed from session records and saved results, kept distinct from confirming a source-level cause.

| Case | Observation | Scope of interpretation |
|---|---|---|
| `vtune_sdpa_gpu_hotspots_b70_20260714_111549`, `sdpa_ocl__prefill` | occupancy 99.0%, XVE active 47.8%, stalled 52.2%, average 5.581 ms / 2 instances | Even with high occupancy, about half of the XVE time was stalled. A real example that occupancy alone cannot determine throughput or the cause. |
| Follow-up OCL vs `sdpa_micro` comparison | OCL SBID stall 59.1% vs 34.2%, barrier stall 10.6% vs 2.5% | Formed the bottleneck hypothesis that dependency/barrier waits weighed more in OCL. SBID is a scoreboard dependency stall, not a direct measure of memory-bandwidth saturation. |
| Follow-up source correlation | source/DWARF mapping was fine, 0 stall PC records in the VTune result DB | Even with correct mapping, without sampled stall PCs a stall cannot be attributed to a specific OpenCL line or ISA instruction. This collection is not source-line bottleneck evidence. |

#### Collection environment and checks when reproducing

- At the time there were VTune 2026.2/2026.3 sessions, and on the B70 Xe driver the system `intel-metrics-discovery` 1.13.545 failed to initialize. Building Metrics Discovery 1.16.190 and putting its `lib` at the front of `LD_LIBRARY_PATH` made counter collection work. This path was the workaround of a temporary build at the time, not a general install location.
- On a 2-GPU system, specify the OpenVINO workload with `-d GPU.1` and give VTune the confirmed B70 PCI BDF with `-knob target-gpu=...`. The recorded collection used `target-gpu=0:3:0.0`. Device ordinals differ per system, so re-check the BDF.
- Collection form (replace the target workload and BDF for the current machine):
  ```bash
  vtune -collect gpu-hotspots -knob target-gpu=0:3:0.0 -- \
    env TEST_USE_SDPA_OCL=1 python <workload> <args>
  ```
  Verify the OpenVINO `GPU.1` selection and the VTune BDF selection separately. For a performance A/B, collect the same workload into separate result directories, and verify in the results that GPU metrics and the actually selected kernel were recorded.
- Xe/i915 counter collection can be blocked by kernel security settings. At the time, `dev.xe.observation_paranoid` and, on older i915, `dev.i915.perf_stream_paranoid` settings were needed. `kernel.yama.ptrace_scope` also affected source attach. Use privileged sysctl changes only when needed, check the organization's security policy, and restore the original settings after collection.
- To preserve the JIT kernel sources generated by OpenVINO, set `OV_GPU_DUMP_SOURCES_PATH=<dir>` together with `OV_GPU_MAX_KERNELS_PER_BATCH=1`. The latter guarantees a separate source file per kernel. These files are not a compile option that enables VTune source mapping itself; they are used to identify the mapped kernel and cross-check it against the ISA.
- Even though a `GTPin` instrumentation failure warning was printed, that collection still produced GPU metrics for the target B70. Do not judge the whole collection as failed from the warning alone; check the result's GPU/device data and the actual metric rows.
- Map the `InternalAddress` in VTune results only to `.text.<kernel>` of the **exact zebin** stored in that result archive. A separate IGC dump can differ in `.text` size (observed case: 128 B), so do not apply addresses across them as is.

#### From observation to a verified optimization

Seeing OCL's high SBID/barrier share, V-prefetch, which puts independent work between the V-read and the barrier, was tried first. In the follow-up benchmark on B70, V-prefetch alone went from 388.04 ms to 381.54 ms, with a re-verification average of 381.17 ms, and the output MD5 matched. A 32-row prefetch variant showed no difference from 16-row and was removed.

A separate f16 V-read merge experiment (`16r16x2c`) improved by 3.33 ms (0.64%) on the same workload, 521.01 ms vs 524.34 ms for x1c of the same build, and the output MD5 matched. The VTune metrics also moved: SDPA kernel span shortened by 2.9%/5.2%, Send stall from 22.0–24.8% to 21.0–22.7%, SBID stall from 43.9–45.5% to 41.5–43.5%. This is a single case of a small improvement and is no guarantee for other GPUs or models. See chapter 05 §5.5 for the detailed measurement discipline.

When re-collecting, the minimum items to record are the VTune version, driver and Metrics Discovery versions, GPU BDF, workload/repetitions, the actually selected kernel, collection knobs, metric definitions, and the result archive identifier. GPU Hotspots active/stalled ratios are for finding bottleneck candidates. Confirm optimization attribution with before/after collections under the same conditions, a device-time A/B and a correctness check; without stall PCs, withhold conclusions at the ISA/source-line level.

The separate 2026-10-03 DG2/A770 PA PREFILL investigation is in chapter 06 §6.3. That session observed XVE active 15.6% / stalled 81.8%, occupancy 47.9%, XMX active 2.0%, spill 7,872 B on the original OCL, while micro had similar occupancy but XMX active 24.4–24.5%. This data showed the initial kernel was underutilized but did not by itself prove a bandwidth bottleneck. After narrowing the loop-unroll, V gather/packing and K bounds-check candidates with VTune/GTPin, each was confirmed with source-injection ABBA and actual device time. The B70 case above and the DG2 numbers are results from different devices/workloads.

---

## 4.2 Diagnostic tool 1: cliloader (runtime ground truth)

Usage (the form used in all sdpa_ocl measurements, B70):
```bash
# setupvars.sh wipes "$@" with `set --`, so save the args before source (see §05)
SDPA_ARGS=("$@")
source <build>/ov_install/setupvars.sh
~/work/opencl-intercept-layer/install/bin/cliloader -d -dv \
  python <llm_bench>/benchmark.py -d GPU.1 -m <model> -n 1 -ic 4 -pf <prompt.jsonl> 2>&1 \
  | grep -E "sdpa_(ocl|micro)__prefill"
```
- `-d` = device performance timing, `-dv` = attach SIMD/REG/SPILL/TPM/SLM/GWS/LWS to the kernel name (source: the actual output format in `test/clintercept_report_dev.txt`, `~/work/opencl-intercept-layer`). `-n 1 -ic 4` is enough for kernel timing (~2 min); for the e2e 1st token use `-n 3 -ic 256` (`sdpa-ocl-beats-micro-256grf`).
- Example output (`test/clintercept_report_dev.txt`):
  `sdpa_micro__prefill_..._sa SIMD16 REG256 SLM=107008 GWS[ 112 x 1024 x 1 ] LWS[ 16 x 32 x 1 ], calls, time_ns, %, avg, min, max`
  `sdpa_ocl_decode__..._sa SIMD16 REG128 SLM=1152 GWS[ 16 x 128 x 2 ] LWS[ 16 x 8 x 1 ]`
  When spill is 0, `SPILL=` does not appear on the line (`sdpa_ocl_prefill_..._sa SIMD16 REG128 SLM=14464 (no SPILL)`, head-72 note §2.7).
- Report files: `test/clintercept_report_{ref,dev}.txt` compare ref (micro/opt) vs dev (sdpa_ocl) over a full gemma-4-26b decode run of 2040 tokens; the `pa-kvup-token-major-store-sector-floor` note used their per-kernel-group device-time totals (16.5 s total). The sdpa kernels in these files had no SPILL.
- Limit: cliloader `-d -dv` is host-bound and hides a 2.2 ms kernel gain in the total time (note quote: "`cliloader -d -dv` is host-bound and hides a 2.2 ms kernel win entirely"). Compare by the sum of per-kernel device times and do not confuse it with e2e. In the same file, wall 23.3 s vs device 16.6 s: 29% is not device time.
- **occupancy% is not a speed metric**: in head-128 prefill, the highest occ% (70%) was actually the slowest (6.86M ns, B70 MEASURED, `sdpa-ocl-kq-tile-keys-32-slower`).
- Verification trap: if "measurements that look like different configs are byte-identical", the toggle was not applied (args lost to setupvars `set --`; on the stateful path the PA kernel does not even launch). First check whether SLM/GWS/LWS are identical.

---

## 4.3 Diagnostic tool 2: IGC dump (static ISA, no observer effect)

### 4.3.1 Dumping at runtime
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> \
OV_GPU_MAX_KERNELS_PER_BATCH=1 TEST_USE_SDPA_OCL=1 \
<ov_gpu_func_tests> --gtest_filter=<case> --device_suffix=1
```
(Scripts: `test/dump_isa.sh`, `test/dump_isa_h128.sh`, `test/dump_isa_gemma4.sh`. Source: `test/dump_isa.sh:1-40`)
- `OV_GPU_MAX_KERNELS_PER_BATCH=1`: each kernel is compiled as a separate program, so there is 1 entry per dump (`*_simd16_entry_0001.asm`). Without it, several kernels share `entry_0002.asm` and identification is hard. Side effect: it hides `FUNC()` name collisions under batching (§4.7).
- File names are hashes, not kernel names. Identify by content: `grep -lE 'kernel sdpa_ocl__prefill' <dir>/*.asm`.
- `--device_suffix=1` is mandatory (2-GPU system: without it sdpa_opt runs on the iGPU and the comparison is invalid).
- Dump outputs: `.asm` (final GEN), `.visaasm`, `.ll` (`*_beforeUnification.ll`, `*_afterUnification.ll`, `*_optimized.ll`), reg info, `*_options.txt`, `*_internal_options.txt`, `.zeinfo`.
- **Do not use `cliloader --dump-kernel-isa-binaries`**: the output is raw GEN, not a zebin ELF, so `ocloc disasm` fails with "Invalid or missing ELF header" (`gpu-kernel-isa-dump`).

### 4.3.2 Runtime source dump (getting the jit prelude)
`OV_GPU_DUMP_SOURCES_PATH=./` (+ `OV_GPU_MAX_KERNELS_PER_BATCH=1`) dumps each kernel as a **standalone `.cl` with all `#define`s prepended** (`clDNN_program_*.cl`). This is the input to the §4.6 splice. This env var is the `GPU_DUMP_SOURCES_PATH` property (`internal_properties.hpp:187`) and needs the `OV_` prefix + an `ENABLE_DEBUG_CAPS` build (`gpu-tensor-dump-env`). The runtime appends `/* Build Log: ... */` at the end of the dump, so strip it before comparing (`BUILD_LOG_RE` in `sdpa_ocl_ab.py corpus`).

### 4.3.3 Offline compile with ocloc
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./occ \
ocloc compile -file X.cl -device bmg \
  -options '-cl-mad-enable -cl-std=CL3.0 -Dcl_intel_dot_accumulate -Dcl_intel_global_float_atomic \
            -Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_split_matrix_multiply_accumulate' \
  -internal_options '-cl-intel-greater-than-4GB-buffer-required -cl-intel-has-buffer-offset-arg -cl-store-cache-default=2 -cl-load-cache-default=4'
```
- `-device bmg` = B70/B580 (Xe2). **`xe2` is interpreted as the lnl-m iGPU**, so do not use it (`sdpa-ocl-kq-tile-keys-32-slower`). DG2 is `-device dg2` (`sdpa_ocl_ab.py hpg --device dg2`).
- `-cl-mad-enable` must be inside `-options`. At top level it is ignored.
- The authoritative source for build options is the dumped `*_options.txt` / `*_internal_options.txt`. sdpa_ocl uses the 4 `-D` above + `-cl-mad-enable -cl-std=CL3.0`; decode uses only `-cl-mad-enable -cl-std=CL3.0` (`sdpa_gen_ocl.cpp:1032-1041`, `sdpa-ocl-ab-harness`). Old scripts (`splice_head72.sh`, `isa_ab_kvup.sh`) compile with `-cl-std=CL2.0` and no internal_options; the latest `sdpa_ocl_ab.py` also passes internal_options to match production.
- What to read in the `.asm` header: `//.instCount`, `//.thread_config numGRF=`, `//.spill size`, `//.private memory size`, `//.full_options`. The greps the scripts use: `grep -oE '^//\.instCount [0-9]+'`, `'^//\.spill size [0-9]+'` (`test/splice_head72.sh:56-60`).
- Mistake to avoid: the `open(p,'w').write(f(open(p).read()))` pattern empties the file before reading -> the kernel compiles empty: "Build succeeded, `kernels: []`", no `.asm` (`igc-inline-helper-pipeline-switch`).

### 4.3.4 Disassembly (`iga64`)
Extract the kernel `.text` from the zebin ELF, then disassemble it with `iga64` (`sdpa-ocl-kq-tile-keys-32-slower`):
```bash
readelf -S -W X_bmg.bin | grep '] .text.sdpa'      # check offset/size
dd if=X_bmg.bin of=kernel.gen bs=1 skip=<off> count=<size>
iga64 -p=xe2 -d kernel.gen > x.asm                 # Xe2/Battlemage; a raw nGEN blob also works as is
```
- `iga64` is at `/usr/bin/iga64` (confirmed 2026-10-06). "iga64 CLI is not installed, only libiga64.so" in `gpu-kernel-isa-dump` is **stale**; `sdpa-micro-dpas-operand-mapping` (07-09) and the current environment agree.
- Usually, however, the `.asm` in the IGC dump is already text, so `iga64` is only needed (a) when extracting directly from a zebin, and (b) for micro's nGEN blob (`micro::Package::binary`, below).
- **sdpa_micro's real GEMM is not visible in the IGC dump.** The dump only has the wrapper; the GEMM is a `dpas.8x1 ... null` dummy + a `0xFADECAFE` sentinel placeholder. Compiling micro's `.cl` with ocloc yields a `DUMMY_DPAS` clobber and ~709 fake instructions -> do not quote micro numbers from it (`sdpa-ocl-u4-head64-page-read`). The actual blob was obtained by (temporarily) adding an env hook that writes `gemms[kq_id].binary` to a file right after `init_microkernels()` in `sdpa_gen_micro.cpp:~904`, and was read with `iga64 -p=xe2 -d micro_vs.bin` (the hook was reverted after use; results: `test/minicpm4_8b_sdpa_accuracy/static_micro/{kq,vs}.{bin,asm}`).

---

## 4.4 Reading ISA: signature -> meaning

Xe2 ISA syntax (B70/B580): memory uses LSC opcodes, not `send` (`load.ugm.d8u32.a64`, `store.ugm...`, `load_block2d.ugm.d8v`); `send` shows up for messages that are not LSC memory opcodes (barrier/fence/EOT etc.) (SLM access uses the `load.slm`/`store.slm` LSC opcodes). `(16|M0)` = all lanes, `(1|M0)` = SIMD-1 (only 1 of 16 lanes used). Line format `(pred) opcode (exec|Mask) dst src... {deps}` (`gpu-kernel-isa-dump`).

| ISA signature | Meaning | Remedy / measured case (hardware, source) |
|---|---|---|
| many `load.ugm.d8u32/d16u32 (1\|M0)` | SIMD-1 scalar gather wasting 15/16 lanes. Usually per-key scale/zp | int8 head-64: 128 of them. 1 wide load with lane-per-key, then `sub_group_broadcast` -> 128 -> 0, instr 4154 -> 3504 (B580 MEASURED, int8-perf) |
| `load.ugm.d8u32 (16\|M0)` x72 + `d16u32` x24 | mid-size per-lane gather (no wasted lanes, too many messages) | micro uses 27 `d32x16t/d32x32t` wide transposed block loads (B580, static ISA) |
| `load.ugm.d32xKt` (K=16,32) | wide transposed block load in 1 message | target form |
| `load_block2d.ugm.d8v` / `load_block2d.ugm.*` | 2D block read (incl. VNNI transform) | head-72 K/V after the gate relaxation: 256 scalar gathers -> ~13 block reads (B70 MEASURED, head72 analysis) |
| `dpas.8x8` count / density | systolic matmul. Normal prefill loop steady state 32 | head-72 NEW_kv2d_dks_on: dpas 52 = 2 x 26 (IGC peels the first iteration of the k0 loop; steady body 26 dpas / 13 block2d) (`splice_head72.sh` comment) |
| `dpas.8x1 ... null:f` + `0xFADECAFE` | **micro blob placeholder dummy**. Not actual compute | §4.3.4 above |
| many wide `mov` (`<2>`, `<4;1,0>` strided) | int8 -> f16 widen + VNNI repack mov-storm | ocl loop body mov 1250 (1181 wide) vs micro wrapper 352; of 679 mov in the S·V region, 609 are strided reshapes, 0 are pure renames (B580, int8-perf). IGC floor; source rearrangement does not reduce it |
| `mov rX<2>:hf ...:w` followed by `:uw<2>` repack | a half intermediate forces the dword-stride (`<2>`) layout | half dequant 144 instr/93 mov vs float 134/83 (microbench). **Do the dequant arithmetic in float, `(half)` once at the end** |
| `r[a0]` (indirect register addressing) | private array accessed by a lane-varying loop variable/runtime index | `r[a0]` 2 -> 274, **1.8x slower** (u4 writer, B70 MEASURED, `pa-kvup-u4-token-major-writer`) |
| `//.private memory size N` / cliloader `TPM=N` + scratch store/load in the hot loop | runtime-indexed private array -> scratch. ISA proof: store right after `math.exp` -> `cmp lt 0x7F800000` -> `sel 0x3F800000`, load right after `asr 4` (/SG) + `mul 64` (float x 16 lane stride), store 2 + load 2 inside the k0 loop | llama-3.2-1b MIXED `TPM=128`. Replaced with a select chain -> private ops 4 -> 0, inst 3646 -> 3628, device 0.8% (`sdpa-ocl-mixed-kc-vc-split`) |
| `//.spill size` > 0 / SPILL= | register spill. Common when many individual gathers are live at the same time | §4.9 |
| backward branch (goto/jmpi target label on a **smaller line number**) | a real rolled loop. A forward `goto` is a lane-predicated skip | awk to find loop body boundaries: `test/sdpa_ocl_ab.py`/session awk. k16 body lines 620-2981 (2361 lines), k32 629-5201 (4572) (B70) |
| sharp drop in `goto`/`join` count + `load.ugm.d32x8t` 51 -> 27 | the unroll is lost (the loop goes back to rolled) | §4.10 guard shape |
| change in `sync.nop/allrd/allwr` count | SWSB dependency annotation. +-1..5 jitter is common. A sync increase unrelated to barriers is usually a schedule change | In `isa_compare`, `sync` is in the heavy list, so jitter alone gives a C grade (74/86 cases). Can be ignored if the send/dpas/math/load/store/CALL counts do not change and no barrier was added in the source |
| `//.thread_config numGRF=256` / `-TotalGRFNum 256` | 256 GRF mode | §4.5 |
| zeinfo `disable_mid_thread_preemption` true -> None | **an IGC size threshold (~600 instr)**, not a functional change. Across 109,802 cached compilations, true if instCount <= 598, absent if >= 601 (none at 599/600) | When a small kernel crosses ~600, this key flips (`sdpa-ocl-ab-harness`, 2026-09-26) |
| `call`/`ret` present | a helper was not inlined | 0 call/ret is normal for sdpa_ocl |
| `math.exp`/`math.inv` count | softmax cost | head-128 f16 prefill k0 body: of 379 instr, softmax 180 (47.5%), dpas 32 (8.4%) (B70 MEASURED, `sdpa-ocl-beats-micro-256grf`) |

### 4.4.1 Region profile (splitting the ISA into regions)
Split the loop body into regions using landmark instructions (first dpas, `math.exp`, SLM barrier `send`, `store.slm`) and compute instruction ratios (e.g. K·Q dpas L4119-4661, exp L5161-5252, S·V dpas L5459-6433, int8-perf). This is a **static instruction count**, not time. As in §4.11 below, there are cases where static counts and time disagree, so always confirm with ablation/measurement.

---

## 4.5 256 GRF mode and the occupancy trade-off

### 4.5.1 Method
`-cl-intel-256-GRF-per-thread` build option (`sdpa_gen_ocl.cpp:1038-1041`, env `SDPA_OCL_256GRF`; decode uses `SDPA_OCL_DECODE_256GRF`, `sdpa_gen_ocl_decode.cpp:232`; **always on for xe_hpg**, since on DG2 every measured tile spills at 128 GRF (`sdpa_gen_ocl.cpp:1038-1040`, 195-196)). Doubling GRF halves threads/EU (on Xe2, 128 GRF = 4 KB/thread, 256 GRF = 8 KB; DG2 GRF is 32 B).
micro always compiles as REG256 when the gemm needs more than 128.

### 4.5.2 Decision table (all MEASURED)

| Case | Effect of 128 -> 256 GRF | spill | Verdict | Source |
|---|---|---|---|---|
| llama-3.1-8b f16 PA prefill q=4096, **default tiles** | 2.18M -> 2.73M ns (**25% slower**) | 0 -> 0 | 256 is not free | `sdpa-ocl-beats-micro-256grf` (B70, fixed at 2800 MHz) |
| same model, **large tiles + 256GRF** (tq32/pwk4/pwq2, sg_per_wg=8, wgTQ=64, wgTK=64) | 1,796,200 ns vs micro 1,930,454 (**7.0% faster**; 17.7% vs default ocl) | large tiles spill at 128GRF | gains only when **tuned as a pair with the tiles** | same note. Configs with `sv_sg_tile_scores >= 64` all spill 3.8-19 KB at 128 GRF |
| gemma-4 head-72 plain prefill (scalar-gather state) | 9.57 -> 14.12 ms (**47% slower**) | 2688 -> 0 | removing spill is not a gain. The bottleneck is the LSC message count | `sdpa_ocl_head72_analysis.md` §2.7 (B70) |
| gpt-oss-20b u4 head-64 MIXED (gather-bound) | 1,474,574,702 -> 2,502,030,737 ns (**+70% worse**) | 5952 -> 0 | for a kernel that hides gather latency with thread count, 256GRF is a loss. The winning fix (c) *increased* spill from 5952 -> 13760 B | `sdpa-ocl-u4-head64-page-read` |
| gemma-4 decode head-512 | 256GRF on par with sg16 (157.1 vs 158.6M), a loss at head-256 | - | not the answer | `sdpa-ocl-decode-tiling-sg-per-wg` |
| DG2 (SG8) plain prefill tile sweep | 256 GRF + 16x16 + sg4 along keys: spill 0. 128 GRF spills 3.8-11 KB in every config. k16q32/k32q32 spill even at 256 | - | HPG default 256 | `test/sdpa_ocl_xe_hpg/s2/S2_RESULTS.md` (small DG2 prototype, not a proof of optimality for the full product PA) |

### 4.5.3 Rules
- **Do not make spill bytes the objective function.** Scratch is L1-resident, so it is cheap compared with 192 scattered messages. First look at *why* many values are live at once (128 independent gathers).
- 256 GRF wins only "when large tiles spill at 128 + the tiles are grown together". Turning on 256 alone with the default tiles is a loss.
- Occupancy has two axes (WG count vs threads/WG). `sg_per_wg` 8 -> 16 was 2.07x faster in gemma-4 head-512 decode (head-256 1.26x), while the same knob was neutral/worse at head-128. The discriminator is `V_TILES >= 16` (`sdpa-ocl-decode-tiling-sg-per-wg`, B70 MEASURED). A knob that was "neutral" on one shape is not a dead knob.
- Changing only `SDPA_OCL_256GRF=0/1` does not give a 128/256 experiment on HPG (HPG always forces 256). A real comparison needs a separate code change after checking the build option string (`sdpa-ocl-xe-hpg-s7-performance`).
- SLM occupancy: Xe2 Xe-core SLM 128 KB, DG2 64 KB (`sdpa_gen_ocl.cpp:155-176`, `max_slm_bytes_for`). The hypothesis that the k32 variant lowers occupancy by growing SLM 17536 -> 25728 was **refuted by device measurement** (§4.11-3).

---

## 4.6 A/B without rebuilding: ocloc splice

### 4.6.1 Principle
jit constants (`kq_sg_tile_keys`, `USE_2D_BLOCK_IO_KV`, `HEAD_SIZE` ...) are `#define`s prepended to the source, so they can be changed without rebuilding OpenVINO. Take the prelude of the runtime source dump (up to the first `#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate`, or up to the first `#define UINT4_RANGE`), delete the `#define`s to be swept from the prelude with `sed` and supply them again via `-D`/`#define`, append `#include "<worktree>/sdpa_ocl.cl"`, and compile with ocloc (`test/splice_head72.sh:30-48`, `test/isa_ab_mixed.sh:1-30`).

### 4.6.2 Script list (all untracked, run by the user)

| Script | Purpose |
|---|---|
| `test/splice_head72.sh` | head-72: 6-row table of HEAD vs working tree x (scalar/kv2d) x (DKS_ACTIVE on/off) (`inst spill dpas block2d load.ugm baseFixup`) |
| `test/isa_ab_mixed.sh {dump,compile,ab,all} [case]` | PA MIXED kernel mode sweep (f16_dmajor, i8_bytoken_2d, u4_bychannel_1d, prefill, sink ...). Modes not targeted by the change must have an **identical instCount and opcode histogram** |
| `test/kvup_splice.sh {splice,check}` | `pa_kv_cache_update_ref.cl` splice. `NAME=VALUE` (replace), `+NAME=VALUE` (add a constant not in the dump), `-UNAME` (undefine; for `#ifdef` tests). Typo guard: a name not in the dump is an error. `check` first proves that the HEAD `.cl` reproduces the dumped ISA exactly |
| `test/isa_ab_kvup.sh` | kvup, all modes, HEAD vs working tree. Confirms 7 modes byte-identical |
| `test/splice_bidir.sh`, `splice_pa_cur_f16.sh` | bidir mask / PA f16 splice |
| `test/sdpa_ocl_ab.py` | generalized harness (§4.6.4) |
| `test/probe_decode_build.cl` + `probe_decode_run.cpp` | standalone for the decode kernel: reproduces the runtime-generated jit prelude by hand, compiles it with `ocloc` (+ISA inspection), and runs a synthetic PA GENERATE problem on the GPU for numerical verification. `probe_decode_build_bmg.{bin,spv}` are its outputs. Build: `g++ -O2 -o probe_decode_run probe_decode_run.cpp -lOpenCL` |
| `test/probe_m1_dpas.cl` | checks whether M=1/2/4 `intel_sub_group_f16_f16_matrix_mad_k16` is accepted by IGC, the dpas rep-count, and the 2D block transpose/transform forms (ocloc only) |
| `test/microbench/` | §4.12 |

### 4.6.3 Proof of reliability (mandatory procedure)
To trust a splice result, **first reproduce a known reference**.
- head-72: `HEAD_scalar` reproduces **spill 2688** = matches the value cliloader reported on the device (B70). The same table also proves that `SDPA_OCL_DKS_ACTIVE=0` is a byte-level no-op vs HEAD (identical instCount/spill/dpas/block2d/load.ugm).
- `ocloc(k16)` ISA == runtime IGC dump, exact match: dpas 32=32, load_block2d 16=16, send.ugm 98=98, total instr 3039=3039, spill 0=0, same hash 8d4bbcc78f6ee655 (B70, `sdpa-ocl-kq-tile-keys-32-slower`).
- kvup: `./kvup_splice.sh check`. However, `dump_ref` is a reference-build output and cannot be reproduced with the current `.cl` -> use `dump_dev` (`pa-kvup-token-major-store-sector-floor`).
- Limit (script comment): ocloc build options differ from OV, so **compare absolute instCount/spill only between rows of the same table**. dpas / block2d / load.ugm counts are structural and meaningful.

### 4.6.4 `sdpa_ocl_ab.py` (offline A/B system)
State directory `test/sdpa_ocl_ab/{snapshots,revs,cache}`. Levels:
| Level | Compared object | Meaning |
|---|---|---|
| L0 | source text to be embedded (`embed()`: reproduces kernels_db_gen.py) | comment/indentation changes are L0-identical (comments never reach the runtime) |
| L1 | `clang -E` token list | same tokens apart from whitespace rearrangement. Compares token lists so that `const int` vs `constint` are distinguished (selftest `merge`) |
| L1' | IGC front-end IR (`*_beforeUnification.ll`) | adding parentheses is L1'-identical/L1-different |
| L2 | ocloc ISA metrics: `//.instCount`, `numGRF`, opcode histogram (predicates stripped), `.zeinfo` execution_env (`slm_size`, `grf_count`, `barrier_count`, `eu_thread_count`, scratch/spill/private) | grade A=identical, B=heavy ops identical + no scratch increase + inst +-0.5% (needs measurement), C=everything else |
| `pset` | host-only change: jit (name,value) multiset equality | proof that the kernel is unchanged |
| `built` | after the build, whether the text is in the `.inc` (chunks joined), the linked `.so`, and `ov_gpu_unit_tests` | §4.7 |
| `hpg` | ocloc compile TSV for another device such as DG2 (`rc, sg, dpas, dpas_forms, simd, numGRF, spill, slm, inst`) | spill is **the ocloc compiler's reported value, not the DG2 runtime value** (`sdpa-ocl-xe-hpg-s7-performance`) |

Cost: 2332 configs + 6110 synthetic jobs, 32 cores `-j 28`: cold L1+L2 ~30 min, one side cached ~10 min, L1 only 2-3 min. The token-key cache `cache/tok/` skips compiling configs with identical tokens. `--l2-from-l1`: L1-identical jobs reuse the base compile (valid because ocloc is deterministic in its input tokens; provided no `#if` was added on macros that only the driver front-end predefines). Self-test: always rerun `selftest` after modifying the harness (negative control).
C-grade diagnosis: keep the IGC dump with `compile_one(text, kernel, keep_dir=dir)`, normalize `%name`/labels, and diff in the order `beforeUnification -> afterUnification -> optimized -> ISA` to find the **first stage that differs** (§4.8).

---

## 4.7 Kernel embedding path (.cl -> .inc) and verification

- `ocl_v2/*.cl` is not read from disk at runtime. `file(GLOB_RECURSE KERNELS "*.cl")` in `CMakeLists` embeds it at build time into `build/src/plugins/intel_gpu/graph/impls/ocl_v2/codegen/include/gpu_ocl_kernel_sources.inc` (**no `src/` after `intel_gpu/`**; the `.../intel_gpu/src/graph/...` copy is a stale leftover from 09-17). GLOB is configure-time, so **a new `.cl` file needs a cmake rerun**.
- The `.inc` splits long kernels into adjacent raw-string chunks (`)__krnl"` newline `R"__krnl(`), so grep misses tokens at chunk boundaries. Exact check: `python3 test/sdpa_ocl_ab.py built --base <rev|snapshot> [--release]` (joins the chunks for comparison + mmap-searches `bin/intel64/<cfg>/libopenvino_intel_gpu_plugin.so` and `ov_gpu_unit_tests` (which statically links the graph library and so has **its own kernel copy**: if only the unit-test relink is stale, the old kernel runs). Run it after linking finishes (a .so that is still being written gives a false negative). Under 1 s.
- In the two build trees (`build/` Debug, `build_release/`), the `.inc` and `.so` can get out of sync after an interrupted build. Check the `.so`.
- Codegen rules (`src/plugins/intel_gpu/src/graph/common_utils/kernels_db_gen.py`):
  - Only **top-level** `ocl_v2/*.cl` are kernel entries (not recursive). Sibling headers are also top-level `.cl` and are included by name.
  - A non-batch `#include "x.cl"` is inlined at build time, **1 time** per kernel (a second include is silently ignored unless it has `[[no_opt]]`). An include that fails to resolve is **silently replaced with `''`**. Batch-header includes are hoisted to the top and resolved at runtime.
  - Strips comments, minimizes whitespace, removes unused `#define`s, adds an `#undef` for every `#define`. Except for `#ifndef X`/`#define X` guard pairs **with no body** -> a classic include guard makes the second kernel in a batched program skip the header. Use a sentinel with a body (`#define X_INL 1`).
  - Several kernels share one OpenCL program (batching), so helpers/types use `FUNC(name)`/`FUNC_CALL(name)`. `OV_GPU_MAX_KERNELS_PER_BATCH=1` hides collisions.
- **Comments never reach the runtime, so a comment-only edit can be proven a no-op at L0** (`sdpa_ocl_ab.py l0`).
- Runtime source dump == `embed(base)` minus only the `#include "include/batch_headers/..."` lines (byte-level proof). So when `corpus` splits prelude/suffix, it also proves *which build the dump came from*. The index stores the base as the **full SHA**, not `HEAD` (so the base does not silently change at the next commit).
- Host-only (C++) compile check: `c++ ... -fsyntax-only <file>` using `CXX_DEFINES/INCLUDES/FLAGS` from `build/.../ocl_v2_obj.dir/flags.make` (~3 s). The target uses `-Wall -Werror -Wmissing-declarations`, but `-fsyntax-only` does not report unused functions in an anonymous namespace (only codegen gives -Wunused-function). It is a build-like run, so get the user's approval.

---

## 4.8 `inline` vs `always_inline`: IGC pipeline switch

**Facts** (ocloc 26.22.38646.4, bmg, MEASURED 2026-09-24, `igc-inline-helper-pipeline-switch`):
- A plain `inline` helper is inlined by IGC, but the **whole module** goes through a different optimization pipeline. Extracting a 1-line `inline size_t FUNC(pa_v_page_base)(...)` made 121 of 122 MIXED configs grade C (inst -72..+17).
- Control experiment: using an unrelated identity helper `inline int FUNC(id)(int x){return x;}` once -> **the same deltas reproduced** (1835 -72, 1818 -28, 1817 +4), and plain-SDPA configs that do not call that helper also moved (000 +18, 001 +4). So the cause is not the helper's content but "the existence of a function that is not alwaysinline".
- IR evidence: `*_afterUnification.ll` is identical except for value names; the difference starts at `*_optimized.ll` (`nocapture readonly` is inferred on kernel args only in the helper build).
- Fix: `__attribute__((always_inline)) inline`. Currently `#define SDPA_OCL_INLINE __attribute__((always_inline)) inline` at `sdpa_ocl_config.cl:268`. The front-end IR still contains the helper and its calls (clang does not inline them), but IGC treats alwaysinline functions differently. The other GPU plugin kernels in the repo do not use always_inline, so they are already on the "other pipeline".

Cases where the ISA moves even with `always_inline` (because the LLVM inliner first simplifies the clone body with the argument values; `docs/sdpa_ocl.md:138-175`):
| Trap | Cause | Remedy |
|---|---|---|
| widened parameter cast | the kernel's `size_t lane` (= `zext(get_sub_group_local_id())`) becomes `(int)lane` in the helper -> at clone time `trunc(zext x) -> x` reuses the entry-block lane id, while the inlined code does a fresh zext at each use. mov -2/-4 | the helper takes both `size_t lane` (for size_t arithmetic) and `int lane_i` (for int arithmetic), and the caller does the conversion. **Switching to `uint lane` makes the size_t uses worse (more spill)** |
| branch on a parameter that becomes constant | with `USE_BIDIR_GATE=0`, `bidir_active=true` -> the branch is removed at clone time -> different CFG, +16 `sync` | put that test at the call site or in an `#if` (the bidir query group loop stays inline) |
| passing combined coordinates | passing `VcD_x0 + sg_j0_sv` as one argument hoists it out of the unrolled loop and changes the order (+23 sync) | pass the leaves (`VcD_x0`, `sg_j0_sv`) separately and sum them inside the helper |
| **array inside the helper** | the inliner wraps the helper's own array in lifetime markers, changing stack allocation in a way the inlined code did not have. u4 head-512 MIXED (spill-bound): scratch 7680 -> 7552 B | scratch arrays (`kt`, `kw`, `vt_pa`, `vzp4`, `zpb4`, `v_pg`) are declared by the caller and passed as `__private` pointers |
| loop-carried output | with `q_pack` as a caller loop variable, SROA keeps the previous value as a phi; a helper-local vector drops it, shifting register allocation at u4 head 512 | use an out-parameter |

Principles (established during the refactoring, 6 commits `1706b74354..0653092a5a`): helper argument types match the caller exactly, unroll trip counts are macros, private arrays are indexed only by unroll constants, pointers carry an explicit `__private/__global`, no structs/globals. Helpers that use prelude macros reading shape_info (`MSK_*`, `KEY_COMP_OFF`, `VAL_COMP_OFF`) take `OPTIONAL_SHAPE_INFO_ARG`.

---

## 4.9 How spill / scratch was removed (only cases with evidence)

### A. Runtime-indexed private array -> select chain (TPM removed)
Symptom `TPM=128`: `float alpha[kq_query_blocks]` (2 float x 16 lane x 4 B) is accessed with the runtime index `alpha_qb` (derived from `sg_ij`) and moves to scratch. Store 2/load 2 inside the k0 loop.
```c
// sdpa_ocl.cl:845-862 (current)
float alpha_sel = alpha[0];
#pragma unroll
for (int t = 1; t < kq_query_blocks; ++t)
    alpha_sel = (t == alpha_qb) ? alpha[t] : alpha_sel;   // kq_query_blocks is a compile-time constant
#pragma unroll
for (int rr = 0; rr < 8; ++rr)
    av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr); // a runtime lane is an indirect reg move
```
Effect: private memory 128 -> none, in-loop private ops 4 -> 0, inst 3646 -> 3628, **device 0.8%** (llama-3.2-1b MIXED, source `sdpa-ocl-mixed-kc-vc-split`). Pre-existing (was also in HEAD).
General rule: **index private arrays only by unroll constants.** To select by a runtime value, use a select chain / `sub_group_broadcast`.

### B. Replace scalar gathers with block reads (spill disappears as a by-product)
head-72 plain prefill: `block2d_surface_ok()` folded the 3 width/pitch/base alignment rules into a single `row_bytes % 64 == 0`, so at head-72 (144 B) K/V fell back to **256 scalar gathers per subgroup per k0 iteration**. Relaxing it to `%16` (+ moving the base down to the 64 B boundary and correcting with `x += prem/elem, w += prem`) gave spill 2688 -> 0 (B70 MEASURED: 9,570,971 -> 1,030,869 ns, 9.28x overall; stage 1 (KV_2D only) 6.96x). Details → 02-memory-io-prefetch-barriers.md.

### C. Better load geometry (scale/zp hoist)
int8 head-64: reading per-key scale/zp inside the dequant loop gives 128 `(1|M0)`. Use 1 wide load with lane=key + `sub_group_broadcast` (a simple reg move because `krel%SG` is a compile-time constant). **The broadcast sits outside the per-lane `(head<d)` guard** (collective). OOB keys get scale/zp=0 (dequant result 0). 128 -> 0, instr 4154 -> 3504 (B580 MEASURED; the user confirmed a meaningful speedup on a real model).

### D. Tile/thread-count adjustment (MEASURED)
- Large tiles spill at 128 GRF -> the `256GRF + tile retuning` pair (§4.5).
- decode `Q_PER_WG` (M): gemma-4 head-512 M=8 gives SPILL=34432 B and is **3.14x slower** than pa_sdpa_opt; M=1 has spill 0. `live_grf_estimate()` (`sdpa_gen_ocl_decode.cpp:68-86`) counts the arrays live across the KQ loop (q_reg[M][K_TILES] half, k_sc/k_zp, k_corr, s[], m/l 5M, S*V accumulator V_TILES x float8 ...) and caps them at a **budget of 112** (below 128 GRF; compiler temps not included). At head-512 sg8, scores 136/164/220/332 vs spill 0/6400/15936/34432 B (monotonic), but **it cannot distinguish differences under ~15 GRF** ((8,M=1)=136 has no spill, (16,M=2)=126 has 640 B spill). So it is a coarse gate, not a predictor. Monotonic in M for the same SG_PER_WG.
- Narrowing the KV page key-tile (config A) vs reducing the thread count (config C): A is 146,417 ns (default 153,227), and C (sg8, wgK128) spills 11776 B at 372,628 ns — "shrinking the tile wins, shrinking the threads collapses". For B/C, **ocloc predicted spill 0 but spill occurred at runtime** (`sdpa-ocl-mixed-kc-vc-split`, B70).
- SV_TRIM / V_PREFETCH (u4 MIXED): skipping S*V for `cp*SUBGROUP_SIZE >= k_chunk` blocks in the unrolled cp loop (SV_TRIM) 151,677 vs 158,422 ns; placing V_PREFETCH right after the S_max aggregation barrier 145,603 vs 151,866 ns (**4.124%**); micro 147,825 ns (B70 MEASURED, `sdpa-ocl-mixed-exact-complete`). This note also states explicitly: "do not conclude spill or the cause of a regression from REG128 alone".

### E. Array lifetime / recompute vs keep
- With evidence: declare helper-internal arrays in the caller (§4.8). u4 writer: to keep `token_vals[]` in registers, the loop variable must be uniform (§4.10).
- **No evidence** (not described because there is no record of an attempt): cases where limiting the `#pragma unroll N` count reduced spill, and cases where splitting a kernel reduced spill. All 38 `#pragma unroll` sites in `sdpa_ocl.cl` are argument-less full unrolls, and `sdpa_ocl_decode.cl` uses `unroll_for` (= `__attribute__((opencl_unroll_hint)) for`, `batch_headers/common.cl:25`). No place in the sdpa_ocl family limits the unroll count (`#pragma unroll 1` exists only in `sdpa_ref.cl:341`, to avoid an FP16 normalization error, and is unrelated to spill). Kernel *splitting* was used not for spill but only for the PA kvup V workgroup split (§4.10, for performance).

---

## 4.10 unroll traps and guard shape

(→ 01-dpas-and-tiling.md: DPAS loop structure)

u4/i8 BY_CHANNEL writer in `pa_kv_cache_update_ref.cl` (B70 MEASURED, `pa-kvup-u4-token-major-writer`):
1. **A lane-varying loop variable takes registers away from arrays.** `for (t = par; t < tpb; t += 2)` -> it cannot be proven that `t>>1` is uniform, so `token_vals[]` moves to indirect register addressing (`r[a0]` 2 -> 274), **1.8x slower** than the shuffle version. Fix: uniform bounds such as `it/npo/np` and a uniform tail. A lane-varying *address* is fine (just a gather).
2. **The shape of a uniform guard decides whether the inner loop is unrolled.** i8 prefill (unsplit 8.2 us), 3 variants:

   | guard shape | static instr / ISA | time |
   |---|---|---|
   | loop bound set to 0 | unroll disappears (7850 -> 2590) | - |
   | `if (skip) return;` at the top of the inline helper | unroll disappears (`load.ugm.d32x8t` 51 -> 27, goto/join 24/17 -> 3/3) | **18.7 us (2.3x slower)** |
   | the same test **at the call site** | unroll kept | **5.6 us** |

   Rule: put guards at the call site, not in the helper body/loop bound. For the V loop too, the guard *wraps the loop*, not the body. Later `docs/sdpa_ocl.md:172-174` promoted this to a general rule ("a runtime or select-based bound, or a helper's early `return`, lost the unroll").
3. **Preserve per-expression type promotion**: `range = (max==min) ? 0.004 : (max-min)` subtracts in **half** and then widens, while `fabs(max*0.1f)` is float. The prefill helper narrows scale/zp to half and then quantizes; the requantize helper stays in float. Follow the arm being replaced exactly to get the same numerics (→ 03-numerics-softmax-quantization.md).
4. **For ablation, set the loop bound to 0** (`if (0)` makes `new_idx = j - token_pos_in_block` negative in the shared `for (j...)`: an OOB read, CL error -14). To turn off only one half: `for (int j = (int)token_pos_in_block; ...)`.
5. With `__attribute__((reqd_work_group_size(1,1,SUBGROUP_SIZE)))`, every enqueue with `lws[2] != 16` gets `CL_INVALID_WORK_GROUP_SIZE(-54)`. Diagnosis: `clGetKernelWorkGroupInfo(CL_KERNEL_COMPILE_WORK_GROUP_SIZE)` (a 40-line standalone query).
6. **Static instCount is not the metric**: the fixed u4 kernel has +643 static instructions and is faster. The dynamic count is the metric — pinning `current_token_pos_in_block` to a literal so that all loop bounds are compile-time predicted the real-model generate ratio within 0.7 pp.
7. IGC **peels** the first iteration of the k0 loop (head-72: dpas 52 = 2 x 26). When counting dpas, the steady-state body is half.

---

## 4.11 Falsified / counterintuitive cases (all MEASURED)

1. **observer effect**: removing only dequant in the int8 K path (K_DIAG=2): 28,124 -> 32,916 ns (slower). Removing the scale/zp loads shakes the IGC schedule/allocation. Only removing the whole K path (K_DIAG=1) was consistent (~9 us drop). -> Trust only static ISA analysis + structural changes (B580, `sdpa-ocl-int8-perf`).
2. **Reading K as a 2D block (non-transform 8b)**: spill went up to 5888/2944 and 30.8/30.3 us, slower than scalar (26.1 us). Later a 2-stage redesign using **transform (VNNI) read + hoisted scale/zp** gave 15.1 -> 12.8 us (B580). Lesson: the bottleneck was not the *number* of reads but the *geometry* (SIMD-1 scale/zp, mid-size messages). Reusing V's cp-pair read (8 -> 4 reads) had no effect/was worse in all 3 implementations (22.8 vs 21.7 us) -> V read count is not a lever.
3. **Why kq_sg_tile_keys 16 -> 32 is slower**: the first hypothesis (SLM 17536 -> 25728 B lowering occupancy 7 -> 5 WG/Xe-core) was **refuted by device measurement**. The real cause is that both axes, (a) `kq_wg_tile_keys = tile_keys x per_wg_keys` and (b) `sg_per_wg`, are "slower when larger" (B70 fixed at 2800 MHz, head-128 compressed prefill q=4096):

   | config | pwk | sg_per_wg | SLM | occ% | min ns |
   |---|---|---|---|---|---|
   | k16pwg4 | 4 | 8 | 12928 | 54.7 | **3,526,979** (reported faster than micro 4.17M; validity in doubt, see the caution below) |
   | k32wg | 4 | 8 | 17024 | 58.5 | 4,346,666 |
   | k16 (default) | 8 | 16 | 17536 | 63.6 | 5,102,600 |
   | k32 | 8 | 16 | 25728 | 65.3 | 5,587,600 |
   | k16pwg16 | 16 | 32 | 26752 | 70.2 | 6,864,687 |

   ⚠ There is a record that the pwg4/pwg2 timings may have come from invalid configs that violated invariant 3 by overriding only `kq_sg_per_wg_keys`, so "k16pwg4 beats micro" is **unconfirmed** (chapter 01 §6.1 "unresolved contradiction"). The valid-config re-sweep is in chapter 01 §6.2.

   Ruled out by ISA: spill (all 0, GRF 128), total instr (k32 has ~2% fewer), sync (k32 has fewer), V-load latency hiding, dispatch (GWS/LWS identical), dpas parallelism. **Nothing in the static ISA is unfavorable, yet it is slower** -> ISA alone is not enough; device measurement is mandatory. Mechanism hypothesis (ASSUMED): the key axis is the reduction axis, so more subgroups = more S_max atomic_max + S_sum SLM + barrier overhead. (The WG count is 2048 in every config, so this is not a grid-underfill story.)
4. **block-level causal mask skip**: implemented and verified (9.3M elements, zero change), but IGC did not take the branch and flattened it: cmp 20 -> 36, net instr +16. Removing the k_mask remainder add also took add 60 -> 20 but mov 59 -> 82, net -3. (B70, `sdpa-ocl-beats-micro-256grf`). **Branch intent in the source != IGC codegen.**
5. **K dequant in half**: slower on the real model too. float 134 instr/83 mov vs half 144/93 (because of the `<2>` dst, `:uw` repack +52). There are reasons why V's `vs_c/vz_c` can be cached as half (scalar-per-lane multiply, no word->half widen array), so do not "unify" them.
6. **store sector floor**: pa_kv_cache_update_ref head-256 u4 token-major is +12.1% vs d-major (6312 -> 7074 ns, B70). **ISA/message count/indirect addressing count were all equal or better (instr 4249 vs 4448, `r[a0]` 125 vs 188, store.ugm 50 vs 51), yet in time the store loop alone is 4.8x**. The cause is dirty cache sectors per thread (16 vs 2). Ablation table (probe, head 256, ns/dispatch):

   | | floor | old-token loads | data stores | total |
   |---|---|---|---|---|
   | d-major | 2201 | 723 | 423 | 3353 |
   | token-major | 1283 | 889 | 2014 | 3700 |

   4 candidate fixes were **falsified by measurement**: (1) sector-sharing co-location within a workgroup, `PA_K_SGS_PER_WG` sgs 1/2/4/8/16, no gain at all (sgs=2 +479 ns): **Xe2 does not merge sub-line writes from different threads**, (2) compile-time unroll+predication 3700 -> 3618 (2%), (3) partitions 16 -> 8 improves the ratio (0.90) but worsens the absolute (4337 vs 3700), (4) partition-tiled page layout rejected due to the risk of rewriting the readers. The ceiling is 986 ns/dispatch (0.024 ms/token): reaching it needs SLM staging + barrier, so the practical gain is parity. Conclusion: **no production code change.** Lessons: "an improving ratio can hide a worsening absolute" (parts=8); when the access *pattern* changes, the metric is sectors per thread, which ISA counts do not show.
7. **spill numbers contradict themselves**: at DKS_ACTIVE 8/6/5, instCount drops 7043 -> 5808 -> 5631 but spill *increases* monotonically 2688 -> 13568 -> 16064 (scalar fallback only; reached only at head 48/96). **ocloc static values, not measured on the device** (recorded as an ASSUMED risk in `docs/sdpa_ocl.md`). Because of this risk, the record says not to revert DKS_ACTIVE but to narrow its default.
8. **The reliability of ocloc spill prediction is inconsistent** (see §4.2): the head-72 splice matched 2688 B exactly, whereas 4 of the 256GRF sweep candidates had ocloc spill=0 but 3.3k-19k B runtime spill, and likewise kc/vc split configs B/C. The cause of the mismatch was not determined from the material in this repo (ASSUMED: possibly build option/internal option differences; the latest `sdpa_ocl_ab.py` passes internal_options, but no record of re-verifying this for spill was found). Rule: trust only the runtime `SPILL=`.

---

## 4.12 Microbenchmarks (`test/microbench/`)

A tool without observer effect. Compile isolated .cl files with ocloc `-device bmg` without editing/building the real kernel, and compare the instruction mix/mov counts (`test/microbench/README.md`).
- Static ISA: `./compare_isa.sh [k_dequant_float.cl k_dequant_half.cl ...]`. Results: K dequant float 134 instr/83 mov vs half 144/93; V dequant baseline 93 mov/cp-block, `A_char16` 93, `B_nozp` 68, `C_char2pack` 68, `E_reinterpret` 68, `F_vec_zp` 93 (only removing zp helps = symmetric int8; asymmetric is the common case, so its use is narrow). `k_dequant_bias*.cl`, `*_dpas.cl`: the 0x6480 xor bias trick and variants connected to DPAS.
- On-device layout probe: `./run_probe.sh {verify_k_transform, probe_v_layouts, probe_v_multiblock, verify_micro_kq_dpas}` — confirms by GPU execution which (key, head) each lane/byte holds (K 8b-transform read -> lane==head and no shuffle needed; V transform is already VNNI-aligned). `probe_dpas_api.sh`: checks which DPAS/2D-block builtins exist on bmg (int8 yes; mixed-precision `hf_i8`/`bf_i8`/`u8_hf` no -> an intermediate widen is mandatory).
- Limits: `bmg` is hard-coded (B580/B70); `dump/` is regenerable output.
- `test/probe_kvup_run.cpp`: GPU execution of the real kvup kernel + differential oracle. With `--bench-iters >= 500`, reproducible to +-1 ns. **A large ratio (12%) matches the real model within 1 pp (+13.2% vs +12.1%), but small ratios are not matched**: in generate the probe predicted -4.4% while the real model was at parity (6080 vs 6080) — the probe reports ~2.2x lower absolute time than reality. Judge small tunings on the real model. `test/kvup_negctl.py`: 15 negative controls (asserting anchor occurrence counts).

---

## 4.13 DG2 / GPU hang diagnosis notes

- The `CL_OUT_OF_RESOURCES` abort in the DG2 f32 PA tests was confirmed as a **GPU hang** by dmesg `i915 ... GPU HANG: ecode ... context reset due to GPU hang` with a matching PID (2026-09-28). NEO reports a hang-reset context as `CL_OUT_OF_RESOURCES`. The kernels involved are on the `pa_kv_cache_update_ref` vs `sdpa_opt__multi_tokens` path, not the sdpa_ocl path (`sdpa-ocl-known-issues-repro`). So a `clFinish` error is a symptom of the hang; check `dmesg` along with it.
- head 486/387 `CL_OUT_OF_RESOURCES` + core dump reproduces only in the full sweep (max ~41 KB WG SLM), not alone (estimated to be resource pressure, not determined). Initially blaming `DKS_ACTIVE` was an error: there was only a `=0` control and no matching negative control (→ 05-methodology-and-pitfalls.md).
- DG2 (xe_hpg) offline check: `OV_GPU_ARCH_OVERRIDE=xe_hpg` dump + `ab.py hpg --device dg2 --grf256`; the spill of representative plain configs, h64 7,968 B, h128 12,128 B, h256 28,064 B, is **the ocloc compiler's reported value**, not a DG2 runtime value or time. A large static spill is an "investigation candidate", not a confirmed regression (`sdpa-ocl-xe-hpg-s7-performance`).
- Small DG2 prototype: at 128 GRF every measured tile spills 3.8-11 KB; IGC sometimes optimizes even hand-written kernels with many registers down to 0 spill, so P7 was designed to compare the T3 kernel at 128 vs 256 (`S2_RESULTS.md`).
- On DG2 SG8, SIMD16 `short8` silently drops DPAS without an error (`sdpa-ocl-xe-hpg-facts`). Always check in the hpg TSV that the `dpas` column is not 0.

---

## 4.14 Spill triage playbook (step by step)

1. **Observe**: get the target kernel's name line with cliloader `-d -dv`. `SPILL=`, `TPM=`, `REG`, `SLM=`, GWS/LWS, device time. (Select the B70 device with `--device_suffix=1`/`-d GPU.1`.)
2. **Check the path**: first confirm that the kernel actually launches and that the configs were really applied differently (compare SLM/GWS/LWS). (On the stateful path the PA kernel does not launch; setupvars wipes the args.)
3. **Separate the kinds of spill**: `TPM`>0 and spill=0 -> runtime-indexed private array (§4.9-A). `SPILL>0` -> register pressure. Cross-check both in the .asm header (`//.private memory size`, `//.spill size`; `grep spill` is meaningless).
4. **Get the ISA** (no rebuild): runtime source dump + `OV_GPU_MAX_KERNELS_PER_BATCH=1`, ocloc splice. First **prove reproduction**: whether the ocloc result matches the runtime spill/instCount/dpas (otherwise see the §4.6.3 limits).
5. **Isolate the loop body**: fix the k0 loop body boundaries by the backward branch (label on a smaller line number), and tally dpas / block2d / `load.ugm` kinds (SIMD-1, d8u32, d32xKt) / wide mov / sync / `r[a0]` (§4.4).
6. **Cause hypothesis -> lever**:
   - many SIMD-1 gathers, N independent loads live at once -> load geometry (block read, lane-per-key + broadcast) (§4.9-B/C)
   - runtime-indexed array -> select chain / broadcast (§4.9-A)
   - lane-varying loop variable, indirect addressing -> uniform bound (§4.10-1)
   - the loop was unrolled / not unrolled -> move the guard to the call site (§4.10-2)
   - live set exceeds 128 GRF -> shrink the tile (M, tile) or 256 GRF + tile retuning; try the option that preserves the thread count first (§4.5)
7. **ISA A/B**: modes not targeted by the change must have an **identical** instCount + opcode histogram (`isa_ab_*.sh`, `sdpa_ocl_ab.py l2`). If grade C, diff the IR stages with `keep_dir` (§4.6.4).
8. **Judge by device measurement**: env toggle (A/B within one binary without rebuilding; 8 runs, min/median; prefill runs 2 times per invocation, so use min instead of avg), cliloader per-kernel time. Do not claim it got faster because the ISA improved. Judge against the production baseline (absolute value) on the same shape.
9. **Confirm the metric with ablation**: turn off the suspected phase with bound=0 (no `if(0)`) and break down the cost. If static counts are equal but time differs, suspect sector/cache patterns.
10. **negative control**: check whether the result reverts when the same change is *reverted*; do not attribute without a matching control.
11. **Correctness gate**: check NaN/Inf, the accumulator type (f32), the precision conversion order, and the dynamic shape (shape_info argument) impact. A tile override can produce *silently wrong output* (the 4 invariants of §01; accuracy fails with no crash and no spill).
12. **Confirm embedding**: after the build, use `sdpa_ocl_ab.py built` to check that the intended source made it into the .inc/.so/ov_gpu_unit_tests.

(Builds/runs are the user's job. The agent does not run the commands above; they are presented as blocks.)

---

## 4.15 Notes vs code mismatches / unconfirmed items

| Item | Status |
|---|---|
| VTune use | B70 GPU Hotspots and A770/DG2 GPU Hotspots/source-analysis were used (§4.1.1, chapter 06 §6.3). The absence of stall PCs in the B70 collection and the DG2 GTPin/native metadata problems are recorded as separate limits |
| `iga64` not installed (gpu-kernel-isa-dump) | stale. `/usr/bin/iga64` exists now |
| "head-64 note at `sdpa_gen_ocl.cpp:157-164`" in `docs`/notes | line numbers moved. The 256GRF description is now at `sdpa_gen_ocl.cpp:1038-1041`, the tile comment at `:195-196` |
| `SDPA_OCL_DKS_ACTIVE`, `SDPA_OCL_MAX_BARRIER_V_PREFETCH`, `SDPA_OCL_PA_CUR_*` | removed in the refactoring (2026-09). Those toggles in the notes are historical records |
| ocloc spill prediction mismatch | cause not determined (§4.11-8) |
| `test/dump_isa.sh` uses `ROOT=/home/gta/work/openvino-eddy` | old path hard-coded. The current repo is `/home/shingyuk/work/openvino-eddy` (`dump_isa_h128.sh` was updated) |
| TPM meaning | In the notes, `TPM=N` is used as cliloader's thread private memory size and recorded as equal to `//.private memory size N`. Not verified directly against the cliloader source |
