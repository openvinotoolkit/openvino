# 06. Performance gap investigation case: sdpa_ocl vs sdpa_micro

This chapter records the case of finding the product kernel's bottlenecks on Xe2/B70 and raising its performance, and the whole process on Xe-HPG/DG2 of investigating the initial 24x gap until a single attention kernel without a separate pre-pass met the measurement gate. The B70 product path and the DG2 standalone experiment path are different. Do not splice the numbers into one time series or generalize them to other GPUs.

Per-tool commands and interpretation limits are in [chapter 07 tool cookbook](07-profiling-tool-cookbook.md), inline vISA in [chapter 08](08-inline-visa-asm.md), and general lessons to carry over to other kernels in [chapter 09](09-general-lessons-for-new-kernels.md). The related design and accuracy principles are in [chapter 01 DPAS and tiling](01-dpas-and-tiling.md), [chapter 02 memory I/O](02-memory-io-prefetch-barriers.md), [chapter 03 numerics](03-numerics-softmax-quantization.md), [chapter 04 ISA and profiling](04-spill-isa-profiling.md), [chapter 05 methodology](05-methodology-and-pitfalls.md).

## 6.0 Summary and evidence boundaries

| Case | Initial observation | Confirmed result | Not yet proven |
|---|---|---|---|
| Xe2 Arc Pro B70 | On some heads/paths sdpa_ocl was up to 8.11x slower than sdpa_micro. The default f16 prefill configuration was also 13% slower on the representative case | The head-72 path became 9.28x faster, 1.14x faster than micro. The best f16 prefill setting is 7% faster than micro. The two recorded u4 MIXED workloads are 1.5% and 6.3% faster than micro | A guarantee that the same tile is best for every shape and model |
| Xe-HPG Arc A770/DG2 | PA PREFILL head 128, seq 4096: initial OCL 110–113 ms, micro about 4.62 ms | A standalone kernel matrix that uses vISA inline assembly and reads and packs the original K/V inside attention passed the approved 3% gate: 32 block-16 head sizes, 2,642 PASS records matching the current policy/source hash, 2,434 unique conditions, worst paired ratio +2.6234% | OpenVINO product raw-kernel integration, end-to-end performance, every head-count/padding/feature combination, Xe2/B70 regressions |
| head 512 PA PREFILL ratio drifting with prompt length (gemma-4, B70, 2026-10-24) | 0.644x micro at 8K, 0.694x at 32K, every tile alike | the 2800 MHz pin is overridden by PL2 in long prefills (slows both kernels); at a fixed 1800 MHz `sdpa_ocl` alone had a knee at 16K-20K = one KV head's K/V outgrowing L3, V miss latency. One-request-per-line V L3 prefetch: -11% at 32K, flat ratio 0.61-0.63x (§6.7) | other Xe2 devices; why micro hides its misses |
| gemma-4-26b decode gap growing with prompt length (B70 1800 MHz, 2026-10-28) | 2nd token vs master 0.99x at 278 tokens -> 1.13x at 133K | head-512 decode register-capped to M = 1 at kv group 8 (pages read 8x per kv head); a bit-identical low-register layout runs M = 8: 712.8 -> 161.5 us/call at 32K, e2e 0.84-0.99x master (§6.10) | i8-cache shapes measured on the probe only |
| lfm2.5-1.2b decode, branch vs reference build (B70, 2026-10-22) | branch 2nd token +3.5%/+2.7% on Linux (+20% reported on Windows) | two causes, each with both controls: the missing upstream RMS+RoPE fusion (+12 kernels and +2 flushes per token) and `sdpa_ocl_decode` at 8 subgroups for head 64; after the rebase and 16 subgroups the branch is 1.3-1.8% faster | the Windows magnitude (host-bound emulation on E-cores gave only +1%) |

The DG2 result is a kernel-only experiment gate. It does not mean that the work of replacing sdpa_micro in the product is complete. Product raw-kernel integration, confirmation of the actual dispatch, the product accuracy gate, end-to-end measurement, and the S6 and SG16/B70 regressions remain as separate steps. The raw data is in the 2026-10-03/04 records under `test/sdpa_ocl_xe_hpg/s7a/perf/`; this chapter summarizes, self-contained, the key evidence and limits worth reusing.

The current working tree also contains product code left over from the earlier `KV_TILED`/pre-processing experiment and a PREFILL-only tier change. This code is not the vISA raw-kernel policy that passed the standalone gate, and it has not been approved by a product build, accuracy, or performance. Distinguish the tier state of the committed HEAD from the state of the uncommitted working tree. Cleaning up this path and integrating the raw kernel into the product are follow-up work.

## 6.1 The first step that makes a performance comparison trustworthy

Before optimizing a large gap, confirm that what you measure is really the comparison you intend.

1. **Confirm the kernel that actually ran.** Use the OV_VERBOSE=4 dispatch census or the cliloader kernel names to confirm which arm ran: sdpa_ocl, sdpa_micro, or sdpa_opt. If an op rejected by the HPG tier gate fell back to opt, that run is not an OCL performance measurement. Paths that swallowed a build failure are also caught by the dispatch census.
2. **Pin the GPU and the memory location.** On a 2-GPU system, check the OpenVINO device suffix and the VTune GPU BDF separately. The A770 experiments used OpenVINO GPU.1 / --device_suffix=1 and VTune target 0:3:0.0. Unit-test inputs can default to usm_host on a dGPU. The 151.8 ms result with usm_host inputs had the PCIe path mixed in and was discarded; only results with the inputs copied to usm_device were used for comparison.
3. **Use the real native binary of the reference kernel.** An OpenCL source wrapper or an early clBuildProgram dump alone can miss sdpa_micro's fused nGEN GEMM. Confirm the final-linked PREFILL binary and the kernel name. With the wrong binary, neither accuracy nor time is valid.
4. **Give both the same work.** Match Q/K/V values and dtype, shape, cache layout, mask/causal, query/key ranges, subgroup/WG placement, warm-up, and enqueue cadence. If micro and OCL actually processed a different number of keys or a different amount of padding, compute that difference before comparing FLOP efficiency.
5. **Separate the kinds of time.** cliloader device kernel time, primitive pipeline time, and end-to-end first-token latency are different metrics. For example, if SDPA is 10–15% of first-token, a 7% kernel improvement may show up as only about 1% in first-token. When there is a pre-pass, measure the attention kernel time and the total time including the pre-pass separately.
6. **Fix the repetitions and the arm order.** With a small number of calls, do not trust a single mean. The B70 comparisons used a pinned 2800 MHz clock, several passes with low variance, and ABBA or independent re-measurement. The pin is not the clock: under a sustained load (long prefills) the PL2 power limit lowers it (§6.7), so sample `act_freq`/`throttle/reasons` and, for comparisons across sizes, pin below the limit. The DG2 standalone gate computed the OCL/micro paired ratio within the same round and checked the queued32 and wait-each cadences separately. Do not pass a threshold by rounding.

cliloader -d -dv/-ko is used to confirm the executed kernels, launch geometry, and device time. A run can be host-bound, so do not confuse small device-time differences with e2e results. The execution environment and A/B rules follow chapter 05 §5.3–5.5.

## 6.2 B70: not one performance problem but a different cause per workload

### VTune narrowed the direction of the investigation, but did not confirm a cause on its own

In the B70 GPU Hotspots collection, sdpa_ocl__prefill had occupancy 99.0%, XVE active 47.8%, stalled 52.2%. The follow-up OCL/micro comparison observed SBID stall 59.1% for OCL vs 34.2% for micro, and barrier stall 10.6% vs 2.5%. This was grounds for a dependency/barrier-wait hypothesis, but that collection had 0 stall PCs. Even with source/DWARF mapping working, the stalls could not be attributed to a specific source line or ISA instruction.

V-prefetch was tested on that hypothesis. On that B70 workload, V-prefetch alone went from 388.04 ms to 381.54 ms, the re-verification mean was 381.17 ms, and the output MD5 was the same. 32-row prefetch made no difference versus 16-row and was removed. A separate 16r16x2c V-read merge improved 524.34 ms to 521.01 ms, 0.64%, with the same MD5, but it is a small single-workload gain. This case and the VTune environment/interpretation limits are recorded in chapter 04 §4.1.1.

**Lesson:** GPU Hotspots active/stalled/occupancy are data for choosing the next experiment. If there are no stall PCs, or the addresses cannot be mapped to the exact zebin, do not conclude an ISA/source-level cause. Adoption is confirmed by a device-time A/B on the same input plus correctness.

### On head-72 the memory path was fixed before SPILL

For gemma-4 vision prefill f16 head-72, OCL was 8.11x slower than sdpa_micro. One head row was 144 B and the pitch was 2304 B, but the host gate bundled the width/pitch/base alignment conditions into the single condition row_bytes % 64 == 0 and turned K/V 2D block IO off. The only rule actually violated was the base alignment. As a result, K/V fell back to 256 scalar gathers per subgroup per key-loop iteration.

The fix had two steps.

1. Split the gate into the %16 rule, and added a base fixup that lowers the base to a 64 B boundary and corrects the x-coordinate and width. K/V block IO turned back on: 6.96x faster.
2. Processed only DKS_ACTIVE = ceil(head/DPAS_K), which the actual head needs, instead of the whole D_MAX. On head-72 the depth tiles went from 8 to 5 and Q SLM also shrank. Another 1.33x.

In total, cliloader device time improved from 9,570,971 ns to 1,030,869 ns, 9.28x. It was 1.14x faster than sdpa_micro. The detailed legality/padding argument is in chapter 02 §3.2.

This case refuted three simple readings of metrics. First, 256 GRF removed the 2688 B spill, but on this workload it was 47% slower, from 9.57 ms to 14.12 ms. The cause was that using twice the GRF reduced the resident threads, which made the scalar-load path even more expensive. Second, the spill disappeared as a side effect after the switch to block-read, so it was not the root cause. Third, the Q/A staging is outside the key loop and was not the main bottleneck in this case.

### The default f16 prefill tuned spill and tile as a pair

On B70, llama-3.1-8b, q=4096 f16 prefill, sdpa_micro took 1,930,454 ns and the OCL default setting 2,181,376 ns, 13% slower. The setting that used 256 GRF together with the tq32/pwk4/pwq2 tile took 1,796,200 ns, 7% faster than micro. First-token latency changed from 466.98 ms to 462.85 ms.

Turning on 256 GRF alone does not win. While increasing GRF to remove the spill of the large tile, the WG key/query placement also had to be re-matched. Conversely, a kernel like the gather-bound head-72 got slower at 256 GRF even when its spill became 0. Do not use REG256 or spill bytes as the sole objective. The B70 pinned-clock conditions and the remaining configuration table are in chapter 05 §5.5.3, the design in chapter 01 §4–7.

Other successes showed the same principle. In gpt-oss u4 head-64 MIXED, the cache row could not take the 64 B block2d path and scalar gathers increased. A whole-page uc16 read plus tile adjustment made it 3.92x faster, 6.3% faster than micro. In contrast, the causal key bound and the token-major K relayout are cases that reduced the actual work or the message geometry. These results are recorded in the tables of chapters 01/02/05 respectively.

## 6.3 DG2: from a 24x gap to a raw K/V single-kernel gate

### The first measured gap

The target was PA PREFILL, f16, head 128, Q/KV heads 32/8, seq 4096, past 0, causal, uncompressed cache, a single subsequence. On A770 DG2, TEST_USE_SDPA_OCL_HPG=1 dispatched the real SG8 OCL arm. The first controlled ABBA gave OCL 110.42–113.27 ms vs sdpa_micro about 4.616 ms, a gap of about 24x. A later reproduction gave OCL 126.74 ms vs micro 4.543 ms, 27.90x. These values come from different measurement runs, so do not add them up as consecutive deltas of one detailed optimization.

The initial host library was Debug, and the Release re-measurement was not completed. However, this comparison was OpenCL device kernel time, and the final acceptance was done in a separate standalone native-kernel harness. The first usm_host run at 151.8 ms was discarded and device-USM inputs were used. The accuracy gate passed, so the gap was a performance problem, not a correctness failure.

### Narrowing the root cause with profiling

| Measurement | OCL original | sdpa_micro | Conclusion that can be drawn |
|---|---:|---:|---|
| XVE active / stalled | 15.6% / 81.8% | 44.2% / 55.6% | OCL did not make sufficient use of the execution resources |
| XVE occupancy | 47.9% | 49.7–49.8% | Occupancy alone cannot explain a 24–28x gap |
| XMX active | 2.0% | 24.4–24.5% | OCL barely used the DPAS throughput |
| L3 read bandwidth | 246.86 GB/s | 783.61–857.93 GB/s | micro was fed more data, but this value alone does not prove DRAM saturation or a single cause |
| Runtime spill | 7,872 B | 0 B | The original OCL had a large register-pressure/scratch problem |

The role of each tool was as follows.

| Tool | What it did in this investigation | Interpretation boundary |
|---|---|---|
| CLIntercept cliloader -d -ko/-dv | Compared the actual dispatch names, per-enqueue device time, and launch information | Check separately that host-bound runs, profiling overhead, and wrong device/memory placement are not mixed in |
| VTune 2026.4 GPU Hotspots | Narrowed the bottleneck candidates with the XVE active/stalled, occupancy, XMX, Send, and L3 metrics | The ratios are not fractions of kernel time. Do not conclude bandwidth saturation from a bandwidth number alone |
| VTune source-analysis + GTPin | Collected memory latency, instruction count, and basic-block latency per native PC | The instrumentation trace and the PC/native mapping must be checked to be sound. Do not convert latency sums into wall time |
| IGC ShaderDump + ocloc + iga64 | Checked .asm/.zeinfo, scratch/spill, DPAS, global/SLM loads, dynamic branches and sends | An offline compile can differ from the runtime one. Confirm reproduction with the runtime options and the zebin actually loaded |
| __builtin_IB_read_cycle_counter phase probe | Generated hypotheses about the relative cost of the KQ, max/softmax, and S*V phases | The instrumentation code changes the ISA/scheduling, so do not use it for acceptance timing or exact latency attribution |
| component microbench | Checked approximate device ceilings and message costs of DPAS, L1/L3, SLM, ALU | Do not convert microbench bandwidth/throughput into overall attention speed |

The VTune 2026.4 collections were run after `source /opt/intel/oneapi/setvars.sh`, with the GPU target pinned by BDF. The A770 was OpenCL device 0 / OpenVINO `GPU.1` (`--device_suffix=1`), and the driver at the time was 26.27.39122.14. In the GTPin low-spill variant, without allowing SREGs there were cases with no data (“kernels not found”) even with exit 0. After `AMPLXE_MORE_GTPIN_OPTIONS='-allow_sregs 1'` we checked that real events and PCs were produced. In the micro final fused native, the ELF function `st_size` was smaller than the actual fused `.text` length, so GTPin source-analysis raised an assertion. Source-analysis was reproduced on a profiling-only copy of the binary with only the symbol size fixed; the native text and the product fuser were not changed. Wrong result directories and failed instrumentation were excluded from the numeric aggregation.

Another important correction was the early guess that “both K and V are scalar gathers”. The executed native ISA showed that K in that OCL was already a 32 B dword block read, and it was the V side that repeated lane gathers and packing. Across the whole workload, the boundary/alignment checks of both readers were also costly. Checking the executed ISA, rather than which source path was selected, corrected the cause quickly.

### Diagnostic A/B changing one thing

Without touching the real OpenVINO build or the product code, kernels injected as runtime source were compared with ABBA. Each delta below is a matched A/B with the same shape and harness. The source-injection results are diagnostic probes that confirmed the root cause, not a product gate.

| Change | OCL device time | Evidence |
|---|---:|---|
| Baseline | 126.74 ms | sdpa_micro 4.543 ms, runtime spill 7,872 B |
| Limit the S*V cp loop unroll to 1 | 126.79 → 57.57 ms | Same number of K/V reads; scratch fill and spill/reload reduced |
| Also limit the KQ db loop unroll | 57.57 → 51.01 ms | After both loop changes spill 704 B, scratch fill about 104x lower |
| Move the V boundary condition outside with a full-tile guard and use 64 B row reads | 50.97 → 28.20 ms | Block reads instead of several 16 B gathers and packing; V sends cut to 1/4 |
| Add a K full-tile guard | 28.18 → 19.00 ms | K/V send count unchanged, but dynamic goto/join about 85.8% fewer |

This result is an example of “bottleneck candidate → prediction → single change → bidirectional A/B”. The earlier diagnosis wrongly suspected K, but the ISA and the per-reader change results separated the K and V roles. Even after these adjustments, 19 ms was about 4.2x slower than micro, so the problem was not recorded as solved at this point.

### Cleaning up the experiment path and removing the pre-pass

In the follow-up standalone experiments, the problem of `MICRO_MATH=1` keeping an extra O accumulator live and increasing spill was turned off. In the measured comparison the probe went from 18.9 ms to 11.8 ms and spill from 1,376 B to 0. A temporary tile-major K'/V' relayout path turned each DPAS operand into a 256 B block read and brought the kernel down to about 5.1 ms. 256GRF + c40 tile reached 4.25 ms (kernel only) vs micro 4.51 ms.

This external pre-pass path was a useful investigation tool at a 4096-token length, but the fixed pre-pass cost was a handicap for short sequences. For h128 at seq 128–1024 the attention kernel was also 10–30% slower than micro, and the pre-pass took about 44–47 µs. So the target was not papered over with a pre-pass or with a micro fallback for short lengths.

The next design was a single raw kernel that reads the original K/V inside the attention kernel and feeds DPAS through register/native packing. That is, it runs without an external K'/V' buffer, a separate launch, or a micro fallback. This final candidate specifies the virtual register shapes and the operand mapping in vISA inline assembly and leaves the physical GRF allocation to IGC. This is an implementation property of the standalone gate. Integrating this asm into the OpenVINO product kernel and checking its behavior per IGC/compiler version is still separate work.

On Xe-HPG the best geometry differed across heads, head-counts, and length boundaries, so no single fixed tile was chosen. The standalone experiment policy used `(head, Qheads, KVheads, max_subsequence_length)`. When PA has several subsequences of different lengths, the longest subsequence length, not the total token count, was used as the selection input. This policy is a standalone candidate selector, not an OpenVINO product host policy.

### Step ladder: which evidence produced which next step

Below are the matched A/B results chained in time order. Shape, harness, and arm differ per step (product Debug OpenVINO → source-injection → standalone harness), so do not add them up in one line. The tools used are named as in the table of chapter 07 §7.0.

| # | Step | Result (A770, h128 32/8 seq4096 f16 causal) | Evidence from this step → grounds for the next step |
|---:|---|---|---|
| 0 | Product sdpa_ocl vs micro (cliloader, usm_device, ABBA) | OCL 110–127 ms, micro 4.5–4.6 ms (24–28×), accuracy PASS | Confirmed the executed kernel name/geometry, discarded usm_host. Defined as a pure performance problem |
| 1 | VTune overview + runtime spill | XVE active 15.6%/stall 81.8%, XMX 2%, spill 7,872 B | Execution units and DPAS underused + large scratch → check the ISA |
| 2 | ISA (IGC dump) | Dominated by scratch fill/reload; V is lane gather+pack, K is already a dword block read | Corrected the "K/V both gather" hypothesis; A/B with unroll and readers separated |
| 3 | source-injection ABBA: S*V unroll 1 → limit KQ unroll | 126.8 → 57.6 → 51.0 ms (spill 704 B) | spill/reload is more than half of the time |
| 4 | V outside the full-tile guard + 64 B row read | 51.0 → 28.2 ms (V send 1/4) | gather → block read |
| 5 | K full-tile guard | 28.2 → 19.0 ms (goto/join −85.8%) | 33% from control flow alone at the same send count |
| 6 | Moved to the standalone harness, `MICRO_MATH=0` | 18.9 → 11.8 ms, spill 1,376 → 0 B | live accumulator (A_tile1 64 GRF) |
| 7 | GTPin bb-latency + load ablation | V loader 49%, K loader 28% (BB cycles); ablation K 7 ms, V 6.8 ms | About 70% is operand load: 16 loads of 32 B per key row + word-mov VNNI pack (~80 instructions per operand) |
| 8 | tile-major K'/V' pre-pass | 11.8 → 5.1 ms | One 256 B block read per operand |
| 9 | tile/GRF sweep (24 at 128 GRF, 63 at 256 GRF) → c40 | 5.1 → 4.25 ms (micro 4.52) | 256 GRF + KQ 16×32 / SV 32×16. Further tile sweeps gave diminishing returns |
| 10 | Length scan | seq128/512/1024: c40 141/226/460 µs vs micro 117/179/411 µs, pre-pass 44–47 µs | **Short lengths and the fixed pre-pass cost are the new problem**. The user: no hybrid, remove the pre-pass |
| 11 | Test the short-length cause hypotheses (below) → try raw K/V staging → native asm | Table below | Analyze with the ISA why micro wins without a pre-pass |
| 12 | Per-head/per-length geometry policy + boundary gate | 32 head sizes, 2,642 PASS records | The section below and the gate scope |

### The causes of slowness at short lengths were refuted one by one

The c40 kernel was faster than micro at 4096 but 10–30% slower at ≤1K (pre-pass excluded). Below are the results of testing each hypothesis one at a time.

| Hypothesis | Test | Result | Conclusion |
|---|---|---|---|
| Causal tail work | Computed the WG's full-key padding ratio: seq128/512/1024/2048/4096 = 1.60/1.176/1.091/1.046/1.023. KQ skip and SV cp shortening as independent A/Bs + branch-only control + source-off control (ISA SHA256 identical) | combined/micro paired +3.14/+6.83/+9.58/−4.65% (128/512/1024/4096), combined/base −3.1/−4.4/−3.2/+0.9%. SV tail −2.5%, KQ tail ≈ 0 | **Refuted the hypothesis that the tail causes the whole gap.** Only a partial gain, and long lengths actually got worse |
| Initialization/idle (wait-per-call) | Kernel unchanged, only the harness enqueue changed: wait-per-call ↔ 32 queued | queued32: 128 → 24/23 µs, 512 → 129/117, 1024 → 409/369, 4096 → 4215/4443 | The large first-K delay of wait-per-call is not normal hot-loop latency. Still, 512/1024 remain ~10% behind |
| Occupancy/wave shortage | VTune overview | seq128 occupancy 21.7% vs 42.6%, idle 56.2% vs 14.6%. 512/1024 identical (45.2/44.9, 47.3/47.5%) | The wave shortage at 128 is confirmed; 512/1024 have **a different cause** |
| How micro feeds K/V | Analysis of the final-linked ISA and the wrapper source | micro also packs: KQ is global-next K load → current SLM operand → current DPAS → next K dword relayout → SLM write/barrier; VS is DPASW + `mov <2>:uw` pair packing. No separate global K'/V'/launch | Evidence that a structure without a pre-pass is possible. It is SLM single-buffer + barrier reuse (`slmBuffers=1`); do not call it "double-buffer" |
| DPASW/operand sharing | Opcode swap `kv_regular_dpasw` (DPAS vs DPASW+P half read) | 222.60 vs 222.63 µs (identical) | The opcode itself is not the cause (also consistent with dpasw ≈ dpas in the microbench) |

### Attempts to bring raw K/V into attention (seq512, h128 32/8, matched µs, all accuracy PASS)

Baseline: c40 (pre-pass K'/V') ≈ 131, micro ≈ 117. The raw-only candidates use only the original K/V, without a separate pre-pass.

| Candidate | Key idea | Result | Assessment |
|---|---|---|---|
| `k_first` SG8×8 gather rows | Cooperative load of raw K → SLM | 189.6 (pipelined 207) | Not adopted |
| `k_block` uniform block rows | Same structure + block read | 545 (spill 7,808 B) | **No wholesale retry of spilling block variants** |
| `k_sg16` / `k_sg16_block` | 16 SG | 228 / 295 | ~2.5x of micro, not adopted |
| `k_sg_sync` | WG barrier → subgroup barrier | 172.8 → 160.9 | Fewer barriers alone is not enough |
| `k_direct` | gather + 8×8 butterfly transpose (no SLM) | 185 | Not adopted |
| `qrows_transpose` | KQ as Q-rows/K-lanes | 159 | A structural change alone is not enough |
| `k_alias` | Alias K and the P scratch | Initial race (maxabs ≈ 0.9); only `k_alias_safe` with an added WG barrier is valid (157.7) | **A change of SLM ownership requires a WG barrier** |
| `v_shared` / `v_direct` | raw token-pair → register VNNI pack | 174 / 206 | The V pack cost is added as is |
| `independent` | raw K/V, query-lane direction | 452 | Slow from repeated online softmax over 16-key tiles |
| native vISA LSC+DPAS+pack bundle (`k16_nativepipe_rw`) | chapter 08 | Wrong at first (uniform SLM address) → 182–195 after the fix | Confirmed the potential of ordering control |
| Precompute raw addresses for 8 rows / whole-KQ single asm | | 184.2 → 179.7 → 171.6 | Small gains, all PASS |
| `kv_original_native` (original KQ direction + register mov relayout) | | 140.8 | First result in the 140s without a pre-pass |
| `kv_sv_whole_addrs` (per-cp addresses precomputed in C + whole-SV) | | 131.3 (micro 119.3) | ~10% slower than micro, also ~14% behind at 4096 |
| **`kv_k_d64`: K d32x8 → d64x4 gather + uq transpose** | Same 32 B/lane | 138.6 → 129.3 (micro 136.7) | **Turning point.** The native message grouping changed the transpose lowering |
| `kv_d64_partial_sv` + Q-dword load + output dword store + null-src0 | Accumulated small changes | −7.8% vs micro at 512 | Each change alone ≤5% |

Rules from this table:
- **Do not interpret the slowness of a candidate that combines many structural changes as a single cause.** The time difference is a mix of SLM staging, barriers, packing, and live ranges.
- Raw cooperative staging mostly lost to one of (a) a spill explosion, (b) added pack cost, (c) more barriers. What beat it was not a source-structure change but *native message grouping* (d64 gather) and *a change in tail code generation* (clamped whole-SV).
- "Good times" that came from wrong results (uq ADD, wrong SLM address, wrong K operand mapping) were all discarded.

### Short lengths: tiny geometry and per-length policy

For h128 32/8, a raw-only policy that handles seq ≤ 96 with Q8/K64/4SG/128 GRF (`tiny64_safe`) and longer sequences with Q32/K128/8SG/256 GRF (`kv_d64_part_v64`) was measured with seed2 over 32 lengths (queued32, warm6, 48 rounds, same-round paired ratio; all CPU accuracy PASS, spill 0).

| seq | raw µs | micro µs | paired | seq | raw µs | micro µs | paired |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.190 | 7.558 | −4.73% | 257 | 48.687 | 51.340 | −5.18% |
| 16 | 7.619 | 8.495 | −10.09% | 511 | 109.890 | 117.797 | −6.76% |
| 32 | 8.495 | 9.577 | −11.52% | 513 | 121.895 | 124.981 | −2.50% |
| 64 | 10.457 | 12.545 | −16.52% | 1024 | 359.530 | 369.440 | −2.63% |
| 65 | 13.182 | 13.834 | −4.48% | 2048 | 1225.703 | 1267.259 | −3.30% |
| 129 | 26.275 | 28.218 | −6.74% | 4096 | 4387.617 | 4441.423 | −1.22% |
| 255 | 45.017 | 46.166 | −2.49% | 8192 | 16592.823 | 16821.034 | −1.35% |

(This table is the queued32 result for h128 32/8 only. Separate defects appeared with wait_each and with other heads/head-counts; see below.)

Factors that worked at short lengths:
- The tiny setting was chosen not by a sweep but on **evidence about the wave/prologue gap** (occupancy 21.7% vs 42.6%, small WG count). The same tiny setting lost +6.05% at 127, so it could not be used for all lengths, and the policy switches at a length boundary (h128: tiny ≤ 64, then Q16 WG2/clamp).
- KQ skip in tiny produced NaN (chapter 08 §8.4-5). Keep the mask and clamp the raw loads.
- For each head such as h48/h112/h272, the geometry's *number of depths* (KQ depth 5/6/7/9/10), V width, and padded value group differed, so separate sources (`tiny_h48_2_2_v16`, `tiny_h112_exact`, ...) were needed.

### Residual head-count- and cadence-dependent defects and their fixes (small MHA/GQA)

The first policy left +2~5% defects for head-counts other than 2/2 (e.g. h128 2/2 seq129/257 gap200 +5.03/+5.02%). Factors tested one at a time:

| Attempt | Result | Adopted |
|---|---|---|
| V address algebra: 16 token clamps and stride multiplies → min of a common column/first/last address + fixed offset (`h128_q16wide_address`) | h128 2/2 seq256 gap: +4.06% → +1.70%, all 30 cases of 15 lengths × 2 cadences within 3% | ✔ |
| causal-bounded raw address clamp (`causal_k-1`) | Independent K/V clamp A/B: +1.64/+2.04/+1.08/−0.08% (h128 129 gap). Address locality is a contributing factor | Partial |
| SGQ32 → SGQ16 + query SG 2 (`h64_16_16_q16wg2`, K128/WGQ32 kept) | h64 16/16 seq256 gap +10.57% → −13.91% | ✔ (≤512) |
| Q64B cooperative load, `Qwide` (h96) | h96 129/257 gap: +0.3~0.9% → −1.0~−4.4% | ✔ |
| key8 KQ (SG8 keys / WGK128 + adjacent S pairs into the 16-key SV layout) | h512 seq128 +7.6% → −12.3%, h144/160/256 1/1 boundary PASS | ✔ (including a bound fix) |
| WG order changes (reverse/heavy-first/interleave), L1 bypass, streaming cache, prefetch | Mostly worse, gap unresolved | ✘ |
| Q8/K128 whole-SV, SGQ8×4 32 SG, K64/K96 (h128/96), large Q tile | Compile/duplicate-label errors or +5~29% | ✘ |
| `addc` 64-bit address, early reuse of the first K load, native atomic marshal | <1% or worse | ✘ |
| Confirm the bottleneck with phase instrumentation | Instrumentation changes the ISA, so unusable for acceptance | Reference only |

The key point is that these residual defects were **not functional defects but 1–5% latency/scheduling differences**. So the gate was closed by (a) freezing the candidate first and (b) checking that it *never once* exceeded 3% across all 15–20 boundary lengths × 2 cadences. An improvement on one shape (h128) hid a new defect on another shape (h64 16/16), and every head-count expansion brought new blockers (head80 16/16 +4.1%, head256 1/1 seq128 +4.4%, etc.). Treat head-count and head size as independent axes.

### Scope of the final raw-kernel gate

The PASS of the 2026-10-04 checkpoint applies to the following comparison procedure and input range.

- Device-USM inputs were used on A770/DG2. The PREFILL native binary that the product micro actually generates and final-links was used as the control. Cases where a wrapper or an early compile binary lacked the fused nGEN GEMM were not accepted as the control.
- Kernel time was compared with the same-round OCL/micro paired ratio from OpenCL device events. queued32/gap0 and batch1/wait-each/gap200 µs were measured separately. Each suite stated its warm-up and repetitions and ran serially, without competing GPU jobs.
- The output was poisoned with NaN, and then every row/channel was compared against a CPU double reference. Besides ordinary inputs, a sharp-softmax stress with Q/K standard deviation `sqrt(1.28)`, V standard deviation `0.1`, and scale `1/sqrt(head)` was also used. `maxabs < 1e-2` was maintained, and accuracy failures were stopped before timing. This stress input matches the logit distribution of the product gtest fixture but is not identical down to the fp16 input bytes, so an accuracy gate on the product fixture is still needed.
- The gate is limited to f16 Q/K/V/output, `k_head_size == v_head_size`, causal PA PREFILL, past=0, uncompressed cache, page block 16. The head sizes are the 32 multiples of 16 from 16 to 512. Not every head size was verified with every Q/KV head count. 24 head sizes were verified mainly at Q/KV=2/2 only, and the additional head-count matrix was checked only for some sizes. It does not cover every padding, subsequence arrangement, feature, or cache format.
- The gate was `OCL <= micro × 1.03` on the same input and cadence. This 3% is the approval criterion of that standalone experiment, not a default tolerance for every kernel. 2,642 PASS records matching the authoritative policy and the original source hash remained, 2,434 input/call conditions after deduplication. The slowest passing condition was head=128, Q/KV=16/2, seq=65, queued32 at 10.8975/10.616 µs = 1.026233509, i.e. +2.6234%. The same-round paired ratio was aggregated, not the ratio of separate medians of the two times.

Remaining problems at several boundaries were solved per source. Below is not the full selection table but the structural fixes that changed the result.

| Problem | Confirmed fix and lesson |
|---|---|
| H=112, Q/KV=2/2, seq=1 | An exact 7-depth KQ and the WGQ8/K64/128GRF tiny geometry fit better than the long-length candidate. After it became bit-identical to the CPU reference, it passed the short-length conditions. Small MHA/GQA is a separate boundary. |
| H=272 partial head | Changing the vector address calculation made the upper 16-channel band wrong (`maxabs=2.432`). A Q-only control and a fixed +32-byte upper-band load narrowed the cause, but the exact IGC lowering cause was not confirmed. The candidate that kept the original vector pointer and skipped only the missing upper band with a uniform native branch passed. |
| H=144/H=160 boundary, H=256 Q/KV=1/1 | In the candidate that reduced KQ to 8-key subgroups, when the KQ/SV producer-consumer widths differed, SV read unwritten SLM and produced NaN/Inf. After limiting the S*V chunk with `min(WGK, causal_k-k0)`, the independent boundary and sharp/multi conditions were re-checked. The policy was not fixed on one length passing. |
| H=80 padded V subgroup | The OpenCL outer-C guard produced NaN/Inf at seq=257. A separate candidate with the uniform guard inside the native asm passed. Control flow around synchronization/SLM is not judged safe just because it looks uniform in the source. |

The detailed candidates/source hashes/exclusion conditions are in `test/sdpa_ocl_xe_hpg/s7a/perf/opt/CHECKPOINT_20261004_ASM_GATE3.md`, `test/sdpa_ocl_xe_hpg/s7a/perf/opt/HANDOFF_20261003.md`, `test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/RAW_RESULTS.md`. The full selection policy in those files is a record for reproducing the standalone experiment of the time, not a product support table.

The `KV_TILED` OpenCL pre-pass branch left in the working tree is an earlier prototype, different from this raw assembly gate. The path that writes tile-major K'/V' externally and reads it back was useful for the long-seq investigation, but the fixed pre-pass launch/write/read cost was a handicap at short lengths, so it was excluded from the final 3% gate. The latest checkpoint also marks this product prototype as an inactive/old path and states that it is to be removed in the follow-up integration. The existence of the current uncommitted `PA_PREFILL` tier change does not count as evidence that the product raw kernel is complete.

### Failed candidates also left design rules

| Observation | Rule to apply to the next kernel |
|---|---|
| 256 GRF accommodated large accumulators but reduced the resident threads, and depending on the tile it was slower than 128 GRF or produced spill | Measure GRF count, SLM, tile, and subgroup count together. Do not adopt on SPILL=0 alone |
| Some variants that moved K/V to SLM or added global prefetch got slower, and L1 prefetch lost at several positions/distances | Do not add prefetch or SLM double buffering on a rationale alone. Check the staging structure of the real product micro in the ISA too, and avoid assumptions |
| Large query/key tile, unroll, SLM transpose, and KQ/SV fusion candidates produced accuracy errors or performance losses | Verify the full output of every candidate before timing. One PASS and a good sample are not adoption criteria |
| On some heads that introduced key8 KQ, SV read more cp than the WG had written and read unwritten SLM | Pass the producer/consumer tile bound as the same value. Test tails and partial tiles separately |
| The OpenCL outer-C variant of skipping the padded value subgroup produced NaN/Inf at seq 257, but the uniform native guard variant passed | Re-verify control-flow changes in regions with synchronization/SLM, including compiler reordering |
| On head-272, the inline address calculation/some vector payload changes broke the upper-channel results. The exact IGC payload-lowering cause was not confirmed | Record failures, not only successes, and do not generalize a variant whose cause is unclear |

## 6.4 A performance investigation procedure reusable for other GPU kernels

1. **Define the performance problem as one exact call.** Fix the operation, the actual dispatch, shape/layout/dtype, mask, data placement, device/driver, build, timing API, and execution cadence.
2. **First confirm that the measurements are comparable.** Check the correct native competitor, source/hash, the executed variant and GPU, warm/cold state, host/device memory, and identical useful work. Mark whether the problem is on the real product path or in a standalone probe. When the comparison is a branch against a reference build, diff the per-step kernel sequence and the host-API census before any kernel analysis; a difference of base (a missing upstream commit) is a suspect too (§6.6).
3. **Gather runtime evidence at low cost.** Save device time and geometry, SPILL/TPM, SIMD/GRF/SLM, and report the call count and min/median or the paired ratio. Occupancy is part of explaining utilization, not a verdict on speed.
4. **Tie each profiler to one question.** GPU Hotspots answers “where does stall/underuse show up”, ISA answers “which memory/DPAS/control instructions are actually there”, and GTPin answers “which events occur at the exact native PC/basic block”. If the question and the tool output do not match, do not add a heavier profile.
5. **Make the single most plausible cause falsifiable.** Instead of “memory-bound”, record a prediction such as “replacing this reader's lane gather with a 64 B block read lowers device time at the same K/V bytes”. Also set up the needed controls, such as a bit-identical/no-op arm.
6. **An A/B differs in one change only.** Avoid double changes such as changing only the source and missing the compile flags. Confirm, before and after, that the actual JIT config, final binary, and ISA signature changed as intended.
7. **Check the accuracy of a fast kernel first.** Check every output row/channel, NaN/Inf, tails, partial tiles, and dynamic/padded layouts. Confirm that the test data does not hide Q/K errors.
8. **Widen the performance gate.** Check sequentially not only the long representative model but also short/long, boundary, odd tail, multiple head-counts, and execution cadences. Report the experiment harness, the product dispatch, and end-to-end as different stages of proof.
9. **Leave the results reproducible.** Record measured/predicted, device/driver/clock, options/JIT constants, source/native hash, the exact comparison arms and commands, warm-up/repetitions, failed/excluded runs, coverage holes, and the next conditions.
10. **If the ratio changes with problem size, separate the clock from the kernel first.** Sample the actual clock and the throttle reasons during the run; repeat the size sweep at a fixed clock below the power limit; normalise time by the expected growth and look for a knee, then for a working set that crosses a cache size at that knee. Falsify reuse hypotheses with value-preserving changes (order, prefetch) so a bit-identical output check covers correctness (§6.7).
11. **Serialize GPU measurements.** Do not overlap separate benchmarks, compiler dumps, or VTune/GTPin collections on the same GPU. Create new tag/result directories, and if overlap or a wrong arm is suspected, discard that run and re-measure under clean conditions.

## 6.5 Questions the performance tools answer

| Tool | Good question | What that tool alone cannot tell |
|---|---|---|
| cliloader / CLIntercept | Which kernels actually ran how many times, and what are the device event time/dispatch geometry? | Overall end-to-end latency, source-level cause |
| VTune GPU Hotspots | Which candidate do the GPU active/stalled/occupancy/XMX/memory metrics point to? | A specific source line/instruction when there are no stall PCs; converting stall ratios directly into wall time |
| VTune source-analysis + GTPin | At the exact native PC/BB, where are load, latency, and instruction events concentrated? | The exact latency of the product native when the mapping/trace is incomplete or the instrumented scheduling changed |
| IGC ShaderDump / ocloc / iga64 | What are the memory messages, DPAS, spill/scratch, and control flow in the ISA the compiler produced? | Execution frequency, cache hits, latency hiding, device time. Cross-check offline ISA against the runtime build first |
| component microbench | What are the approximate ceilings and message costs of DPAS/load/SLM/ALU on that device? | Whether the whole kernel is limited by the same bottleneck |
| phase cycle counter | In one diagnostic build, how do the relative costs of KQ/softmax/SV etc. look? | Uninstrumented acceptance timing after the instrumentation changed the ISA and scheduling |

The exact commands, environment variables, collection modes, and failure cases of each tool are in chapter 07. Interpret every timing result only after correctness and confirmation of the actual dispatch. Spill bytes, instruction count, bandwidth, and occupancy are independent clues. A conclusion that a change in one of them caused a speed change must be supported by a matched device-time A/B.

## 6.6 Case: branch vs reference decode gap (lfm2.5-1.2b, B70, 2026-10-22)

The question was why the branch's 2nd-token latency was slower than a reference build (master plus the host-cost PR) for lfm2.5-1.2b (INT4 weights, u4 cache): about 20% on a Windows B60 machine and about 3% on the B70 Linux box. The steps, each with the evidence it produced:

1. **Per-step enqueue census first.** An LD_PRELOAD shim recorded every `clEnqueueNDRangeKernel`/`clFinish`/`clFlush` with timestamps and kernel names; splitting the stream at syncs and taking the most common decode step gave 196 kernels (branch) vs 184 (reference). The difference was 12 `rms_gpu_bfyx_opt` kernels: the per-head q/k RMSNorm of the 6 attention layers, which the reference folds into `rope_opt`. cliloader `-h` showed an otherwise identical host-API census per token (same `clSetKernelArg` count) plus 2 extra `clFlush` per token.
2. **Name the mechanism, then control both sides.** The reference had the upstream `FuseRMSRoPE` pass (option `GPU_ENABLE_RMS_ROPE_FUSION`), which the branch base predated. The reference with the option off reproduced the branch's kernel sequence exactly; the branch with the pass ported and its decode kernel off reproduced the reference's MD5 exactly. Only then was the fusion credited (about +0.04 ms per token).
3. **Split the remainder by kernel.** With the fusion on both sides, branch and reference differed only in the GENERATE kernel (`sdpa_ocl_decode` vs `pa_single_token`), and cliloader device time put the rest there: 20.9-22.7 us vs 14-20 us per call. The decode kernel ran 8 workgroups per partition (8 kv heads, M = 4, 8 subgroups) and was latency-bound; 16 subgroups made it 0.45-0.51x (chapter 01 §8.3).
4. **Test the platform hypothesis instead of assuming it.** "Windows is host-bound, so extra kernels cost more" was checked by pinning the run to the E-cores (decode became host-bound at 3.4 ms per token): the branch was only 1% slower there, so host-boundness alone does not explain 20%. The Windows magnitude stayed unexplained, and the exact control commands for measuring it there were handed over (`OV_GPU_ENABLE_RMS_ROPE_FUSION=0`, `TEST_USE_SDPA_OCL_DECODE=0`).
5. **Side effects are part of the result.** The upstream fused rope uses a 2-work-item workgroup and made a 1024-token prefill 0.36 ms slower (q 121.5 vs 71.8 us per layer) while the 32-token prefill got 0.2 ms faster; this was reported, not changed.
6. **Pre-existing failures need a one-variable control.** The PA gtest run had 13 failures; a test binary built from the pre-change source gave the identical failed and skipped sets, which is what makes them "pre-existing".

## 6.7 Case: head 512 PA PREFILL ratio drifting with prompt length (gemma-4, B70, 2026-10-24)

The shipped head 512 PA PREFILL kernel was 0.644x `sdpa_micro` at 8K keys but 0.694x at 32K (gemma-4-e2b; gemma-4-26b 0.730x ->
0.743x), and every tile tried showed the same shape. Details and all numbers: `docs/sdpa_ocl.md` "Paged-attention PREFILL: the
`d_max` 512 long-prompt knee"; prefetch mechanics: chapter 02 §7.6. The steps, each with the evidence it produced:

1. **Locate the knee with a clean length axis.** Prompts were cut from one 32K text (8.1K to 33.4K tokens in 7 steps) and per call
   time divided by keys^2. Both kernels rose from 16K-20K at the "2800 MHz" pin. That pointed at one KV head's K+V (2 KiB per
   token) outgrowing the 24 MiB L3 (12K tokens), but that was only a hypothesis.
2. **Falsify the cheapest reuse hypothesis.** If concurrent workgroups (consecutive query blocks of one head, drifting apart in key
   position) thrash L3, putting the 8 heads of a KV group into the same wave should cut the misses 8x. It was 4-7% slower at every
   length and the knee stayed: not reuse.
3. **Check the clock, not the setting.** A 5 ms sampler of `act_freq` and `throttle/reasons` showed `pl2` holding the GT at
   2133-2167 MHz for most of a 32K run, rarely at 8K. Pinned at 1800 MHz (0 `pl2`): `sdpa_micro` was flat, `sdpa_ocl` still had a
   +9.6% knee, and the ratio drifted exactly as before. The clock slowed both; the knee alone moved the ratio. A gap between two
   kernel variants (6-10% at 32K) vanished at the fixed clock too.
4. **Profile at the fixed clock.** VTune overview (needed `dev.xe.observation_paranoid=0`; chapter 07 §7.3): L3 miss 3% -> 18%,
   Send stall 20% -> 30%, DRAM read 64 GB/s (far under peak), L3 read per call = the computed one read per workgroup. Micro read
   2.8x more from L3 without stalling. Reading: miss latency on the V reads, not bandwidth.
5. **Prove the latency reading with a value-preserving switch.** The existing `PREFETCH_V` flattened the knee (121 ps from 16K) but
   cost 14% at 8K. Prefetch placed at L3 only cost the same as L1+L3 (not L1 pollution); 4x fewer demand messages for the same
   lines was slower (not the message count); the prefetch stride decided it: 64 / 128 / 256 B = +17.0 / +4.4 / -0.5% at 8K. Cost =
   line requests; the L3 line is 256 B.
6. **Adopt the narrow form, measure where it does not apply.** One request per line, L3 only, this tile's V, `d_max` 512 only:
   -0.4..-1.6% at 2K-8K and -11.2% at 32K at 1800 MHz, flat time / keys^2 from 12K; at the 2800 pin with PL2 32K 111.8 -> 100.1 ms
   and TTFT -1.3..-1.9%. At `d_max` <= 256, where one KV head's K/V fits L3, the same prefetch was 1-5% slower: kept off by the
   user's decision.
7. **Verify cheaply because nothing changes value.** 173 PA gtest outputs bit-identical with the switch forced on and off at every
   head size; the two suites' failure sets unchanged. A 4K TTFT +0.8% was arm order (reversed: +0.4% / 0.0%; kernel -1.2%).

## 6.8 Case: short-prompt PA PREFILL and f16 MIXED (B70 1800 MHz, 2026-10-27)

Two gaps of the same session, each closed by reading what the profiler counted rather than how busy the GPU was.

**Short-prompt PA PREFILL** (qwen3-8b, `d_max` 128, same workgroup as `sdpa_micro`): 24.7 against 12.6 us per call at 32 tokens,
253 against 214 at 1K, level from 4K. The handoff called it a fixed cost.
1. **VTune overview at 1K, both kernels**: XVE active 63.3 / 63.7%, XMX busy 57.8 / 59.1%, the same stall mix, but **XMX
   instructions 2.49 G against 1.85 G (+35%)** and ALU instructions +37%. Equal efficiency, more work: the kernel computed
   something `sdpa_micro` skips.
2. **Count the work by hand**: at 1K with 128-query x 256-key workgroups, the diagonal tile is computed whole and masked
   afterwards; keys past the workgroup's last query are 1.25x the needed key rows, the triangle inside it more.
3. **First cut, workgroup bound** (skip KQ of subgroups past `causal_k`, S*V blocks past it): 32 tokens 24.4 -> 16.9 us, 1K
   unchanged (253 -> 247). **Second cut, subgroup bound** (each KQ subgroup's last query, each S*V subgroup's last score row):
   14.1 / 216 us. Both skip exactly-zero work: 186 PA rows bit-identical; a bound 16 keys early fails exactly the 11 PREFILL rows
   at `d_max` >= 128.
4. **Breadth before the default**: the same skip cost `d_max` 64 6.8% at 4K and MIXED f16 4.7% at 8K (the uniform branch costs
   every tile, the saving is on one), so it ships for PA PREFILL at `d_max` 128 / 256 / 512 only.

**f16 MIXED** (qwen3.5-9b, d-major K page): 1,465 against 522 us per call. The ISA showed the per-key gather; the token-major page
(441 us) bounded what a block read could give. The d-major page read through a 64 B pitch surface and a 32b transpose
(chapter 02 §4.5) reached 508 us; VTune against token-major then showed **Send stall 7.8% vs 2.4%** and more L3 misses, i.e.
the transpose messages, and the 32-row transpose (one message for two depth tiles) took it to 460. A control that removed the Kc
branch from both arms (`SDPA_OCL_PA_CUR_F16=0`) moved both by ~10% (406 / 401): part of the "transpose gap" had been the Kc
code, which an f16 cache does not need at all.

Lessons: equal utilisation with more instructions of one type means redundant work, not scheduling; compare per-type totals
(XMX, ALU, Send) against the reference first. Bound a candidate with the ideal layout (token-major here) before tuning, and
remove a shared confound from both arms before attributing the remaining gap.

## 6.9 Case: a compiler upgrade regressed the KV-cache writer (B70 1800 MHz, 2026-10-27)

After the IGC 2.36 -> 2.41 upgrade the u4 token-major `pa_kv_cache_update_ref` PREFILL went from 0.75-0.86x master's writer to
1.3-1.6x (qwen3-8b 8K 322 vs 224 us), with `TPM=576` on every head size; master's kernel had none.
1. **Re-measure the baseline under the installed driver** (same numbers as the day before: the task was still needed).
2. **Name the memory**: runtime IGC dump of the kernel (`NEO_CACHE_PERSISTENT=0`, it sits in a batch with other kernels):
   `//.private memory size 576`, no `//.spill size`; `*_optimized.ll` keeps one `alloca [2 x [9 x half]]`. A private array, not
   register pressure, so 256 GRF and IGC knobs were tried once and changed nothing.
3. **Attribute with variants** (offline splice + ocloc, no build): reshaping one array split the slot into two (1216 B), dropping one
   helper left the other's 576 -> two helpers, one merged slot. Static indices in both removed it.
4. **Calibrate the probe before trusting it**: a 512-block PREFILL dispatch in the standalone probe matched cliloader (313 vs 322 us
   before, 168 vs 172 after); its GENERATE times are about half the real ones and small ratios did not carry over.
5. **Sweep every shape after the fix**: head-256 GENERATE got 7% slower than with the scratch. A TIMING-ONLY ablation (stores
   removed) put the cost on the store tail: 16 partitions writing the same 128 B rows in the same order. Rotating the start row per
   partition fixed it; the same rotation cost 3-8% at head 128 / 512, so it is gated to head 256.
6. **Prove bytes, not tolerance**: the basic gtest passed two deliberately broken writers; the per-row cache dump comparison caught
   both and showed the fix identical (chapter 05 H).
Result: PREFILL 0.49-0.77x master, GENERATE at or below master except head 256 (+3-4%, the store-sector floor of chapter 02 §10.2).

## 6.10 Case: decode gap growing with the prompt (gemma-4-26b, B70 1800 MHz, 2026-10-28)

The daily logs showed gemma-4-26b's 2nd token at 0.99x master for 278 tokens and 1.13x for 133K. Details and numbers: chapter 01
§8.4 and `docs/sdpa_ocl.md` "sdpa_ocl_decode". The steps, each with the evidence it produced:

1. **Reproduce the trend with both arms under one clock** (daily command, interleaved, load logged): dev/master 0.985 -> 1.126.
2. **Per-decode-token kernel time** (chapter 07 §7.1): two cliloader runs at 32K, `-ic 1` and `-ic 129`, subtracted per kernel
   family and divided by runs x 128. Everything matched except the head-512 decode call (712.8 vs 441.9 us); the SWA layers were
   faster than master. The dispatch line gave the cause directly: `GWS dim 1 / LWS dim 1` = 16 head groups = M = 1.
3. **Env sweep before code**: larger M at 128 / 256 GRF (spill and all) -> amortization wins, registers are the limit.
4. **Standalone probe as the bench**: with a 256 MB fill between launches it reproduced the in-model per-call times (679 vs 714 us
   at M = 1, 630 vs 624 at M = 8), so every kernel variant was timed and checked bitwise (`PROBE_KDIR` = a `git show HEAD` copy,
   `PROBE_DUMP`, `cmp`) without rebuilding the plugin.
5. **Data-movement-only changes, one at a time**, each bitwise-identical to the baseline on all 177 probe cases (and with sinks):
   spill 17.7 KB -> 0, then the unprefetched per-chunk V comp load turned out to be the short-context latency item.
6. **Every shape the rule touches**: the probe benched all capped decode shapes of the daily models (u4 and i8), which is how the
   head-64 / i8 short-context loss was found and excluded; unchanged shapes were proven ISA-identical with ocloc.
7. **Model checks with a valid oracle**: e2e on all switched models, with `--disable_prompt_permutation` and only prompts whose MD5 is
   stable inside each arm (chapter 05); one +1% short-prompt reading (e2b) went away when re-measured interleaved with `-n 5`.
8. **Tests that can fail**: new u4 GENERATE rows plus a negative-control build (twin store swapped, zp bias dropped): only the
   `sharp_softmax` rows failed, so those are the ones that guard the path.
Result: 2nd token 0.84-0.99x master over 278-133K tokens (was 0.99-1.13x); the other switched models 0.99-1.00x at short and
0.80-0.96x at long prompts against the branch before.
