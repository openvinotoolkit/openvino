---
name: xe-block-io-prefetch-barrier
description: Playbook for Intel Xe2/Xe-HPG OpenCL kernels' memory path - 2D block read/transform/transpose/prefetch legality (width/pitch/base, 8b 32-row, 32b transpose), base fixup gates, padded views, uc16 page reads, V prefetch, split barriers, K token-major KV-cache layout, OOB hazards. Use for sdpa_ocl, block2d, prefetch, barrier_arrive, scalar gather.
---

# Xe memory IO / prefetch / split barrier playbook

In-depth evidence and measurement sources: `src/plugins/intel_gpu/docs/ocl_perf_guide/02-memory-io-prefetch-barriers.md` (hereafter "ch02").
Related: DPAS/tiling `01-dpas-and-tiling.md`, numerics `03-numerics-softmax-quantization.md`, ISA/spill `04-spill-isa-profiling.md`, methodology `05-methodology-and-pitfalls.md`.

## 0. Rules before starting (AGENTS.md)
- Builds/gtests/ab/dumps/benchmarks are **run only by the user**. Give exact command blocks and receive the results (`--device_suffix=1` is mandatory, otherwise you measure the iGPU).
- No performance improvement claims: without measurement, label it ASSUMED. State the device (B580/B70/DG2) and the method (cliloader average ns, probe, ISA counts).
- Static metrics (instCount, message count, spill) turned out to be the wrong metric whenever the access pattern changed (ch02 §9.2, §10.2). Look at the message *kind* (block/scatter/transform) and the touched lines/sectors; the final verdict is time.
- Check numerical impact: NaN/Inf (unwritten slots, OOB rows), accumulator type, dynamic shape, padding.

## 1. Checklist before using a 2D block builtin (ch02 §2.1)
1. Is the subgroup size 16? DG2/ARL-H (xe_hpg SG8) have no 2D block at all -> turn it off in the jit with `block2d_io_allowed()` (`#ifdef` is not possible).
2. Does the variant exist? 8b transform is **32-row only** (no `16r`), transpose is **32b only** (`16r8x1c`, `32r8x1c`; IGC 2.41 has `16r16x1c`/`64b` transposes in libigc but OpenCL C does not declare them), 8b plain read needs a column count that is a multiple of 4. If unsure, `test/microbench/probe_dpas_api.sh`.
3. width >= 64 B (a multiple of 4 for 8/16b), pitch >= width and a **multiple of 16 B**, base 64 B aligned (or fixup), `coord.x` unit = element (8b byte / 16b half / 32b dword).
4. Clamp the surface height to the valid rows (NaN in unwritten page slots, the last k0 tile). Do not issue reads with height <= 0; fill with 0.
5. For the head tail, HW zero-fills with surface width = `d*sizeof(T)`. A widening fixup goes **backward (lower address)**.
6. **Confirm the lane/element meaning with a probe** (fill K[key][h]=key). Watch out for the trap where a probe without a sentinel reports an all-zero match.

## 2. Gate selection (host, `sdpa/sdpa_ocl_utils.hpp`)
| Target | Rule |
|---|---|
| Q, A (output), plain i8 K/V (no fixup) | strict `row%64==0`, `row>=64` + padding proof |
| plain f16 K/V, MIXED Kc/Vc | fixup `row%16==0` + `BLOCK2D_KV_BASE_FIXUP` (round-down 64 B, `x += prem/elem`, `w += prem`) |
| PA cache page f16/i8 | `block2d_page_ok` (`%16`) |
| u4 page | keep strict (mutually exclusive with the 1D uc16 read. Relaxing to `%16` would replace the proven 3.92x path) |
| decode page | strict `%64` (f16 `h%32`, i8 `h%64`, u4 K `h%128`) |
- **The fixup flag must be derived after the `SDPA_OCL_*_2D` override** for the override to be safe. `SDPA_OCL_KV_2D=1` is a lever for proving a hypothesis without a rebuild.
- Q/A have no fixup (32b transpose needs `prem%4`, and the cost makes it pointless): **forcing 2D risks wrong results**.
- rank-4 padded: passes if simple format && X unpadded (pad only multiplies the pitch). rank-2 PA token matrix: if static, check first_head/stride; **dynamic padding cannot be proven** -> strict false, fixup trusts its preconditions (stride%16, start%4).
- B70 measurements: wrong results come from a token stride that is not a multiple of 16 B and a start misaligned by 2 B. 4/16/32/48 B bases are fine (the device test proves only the pitch; the base rule is covered only by the host test `sdpa_block2d_gate`).

## 3. Pattern catalogue
- **8b i8/u4 K/V**: `transform_8b_32r16x1c` gives lane=column, uint=4 rows. For a page (16 tokens), clamp the height and use only uint 0..3 (the 50% over-read is an HW/OpenCL limit; micro gets 0 by controlling the descriptor directly). cp-pair reuse to reduce the read count gave no time improvement in all 3 attempts.
- **i8 K transpose**: giving `transpose_32b_16r8x1c` a byte width/pitch makes it a transpose that views the i8 surface as dwords (`sdpa_ocl_decode.cl:547`). 8 dwords = 32 head dims = 2 DPAS tiles -> half the K messages.
- **u4**: byte=channel pair -> the same depth permutation on A/B (`PA_K_U4_CHANNEL`), Kc is a DWORD surface (`read_32b_8r16x1c`), `kc_dword_ok` runtime check + scalar fallback.
- **f16 d-major K page (MIXED, `USE_2D_BLOCK_IO_K_PA_DT`, ch02 §4.5)**: re-declare the page as a 64 B pitch surface (row r = channels 2r, 2r+1, 16 keys each), `transpose_32b_16r8x1c` at x=8*(db&1), y=16*(db>>1) -> lane L = channel `PA_K_U4_CHANNEL(db,L)` (u4's permuted depth: Q staging/Kc reads reused), word halves = k_raw rows. `32r8x1c` (probed: lane L gets rows y+L and y+16+L, interleaved) = tiles db and db+2 in one message, 460 vs 508 us; 256 GRF only (4 KB spill at 128 GRF). Mask never-written keys by ANDing each extracted row with `sub_group_broadcast(key_ok, krel)` (an `if` cost 16 phi movs per depth tile). qwen3.5 f16 MIXED 1,465 -> 406 us (token-major page 401, micro 522).
- **head 32/64 u4 (row<64 B)**: `intel_sub_group_block_read_uc16` 1D whole-page read `PA_PAGE_R/I(ROW,t,c)`. `r` is a compile-time constant; for a runtime `c`, bias the base by `SUBGROUP_SIZE*c` + c=0 (prevents the `uchar16` subscript from becoming indirect addressing). Remove the per-key guard if only finiteness is needed (gpt-oss 3.92x, B70).
- **per-key scale/zp**: reading it in the innermost loop is lane-uniform, so a SIMD-1 load happens per key. Instead, do 1 lane=key wide load at the top of k0 + `sub_group_broadcast` (compile-time constant lane). Also hoist `block_indices[]`.
- **SLM**: micro's GEMM uses 0 SLM; the wrapper's Q/S staging is all of it. SLM size is a function of tile size, and occupancy headroom is not throughput. `slm_p` is key-indexed + head is the innermost vector.
- **Wide reads (`WIDE_READS`, PA PREFILL, ch02 §5.4)**: bit 1 K pair `16b_32r16x2c`/`16b_16r16x2c` per depth-tile pair, bit 2 V pair `transform_16b_16r16x2c` (needs `sv_value_blocks` even), bit 4 Q_slm `block_read_ul4` (`d64x64t`). All value-preserving. Defaults: d512 7, d64 5 (0 with a sliding window), d128/d256 1. Q_slm costs **8 deinterleave movs per read** (ulong4 -> int8), so it wins only where the loop is memory-bound (d512, long head-64 prompts) and loses when K/V stay in cache (short prompts, d128 +1.4..4%, 128-key window +10-12%). Stage Q_slm with per-lane `ulong` stores: `block_write_ul4` to SLM gave wrong Q on dynamic-pad head-64 rows.
- **DG2 (xe_hpg)**: local block IO works; global `block_read uint` only with 4 B alignment + even pitch (odd pitch gives wrong results). SG16 `short8` DPAS drops the DPAS without an error (check that the ISA has `dpas`).

## 4. Prefetch
- Where to use it: **loops whose load latency is exposed because there is no independent chain** (decode S*V's 16-page single-accumulator chain). It hurts loops that already interleave independent DPAS chains, like K (K prefetch -3.6%).
- Distance is non-monotonic (decode, 1/2/4/8/16 = -1.45/-1.87/**-2.10**/-0.56/**+2.21 %**). Distance only changes *where* it is issued; the count stays the same. Distance > ~4 evicts from L1.
- At the issue point use the **same surface/fixup/coordinates** as the actual read. A prefetch right before consumption is pure overhead (plain prefill lost about 11 us; removed).
- Cost: about 8 instructions per prefetch (a64 descriptor setup). It has no destination register, so it provides MLP that occupancy cannot buy.
- MIXED Vc prefetch (u4, right before softmax, `!from_cache && sg_i_sv==0`): -4.12% (B70 llama-3.2-1b, cliloader). Not measured on i8 BY_CHANNEL MIXED.
- MIXED **i8** V page prefetch (`PA_MIXED_PREFETCH_V`, this tile's V columns at iteration start) on every tiling incl. the d64 table: -10..-18% per call (qwen3.5 799 -> 709, llama-3.2-1b 630 -> 527); the K page prefetch 0..+1.6% (off). ch02 §7.7.
- PA PREFILL V prefetch (`v_tile_prefetch`, cooperative 16r16x2c over this key tile's V rows inside the split S_max barrier, as micro): -4.5% at head 128 (qwen3-8b 8K), but +10-14% at head 512; `v_prefetch` is a per-tiling flag (d_max <= 256). B70 pinned, ch02 §7.5.
- **A prefetch costs its cache-line requests** (ch02 §7.6, B70 1800 MHz, head 512): address prefetch of this tile's V at a 64 / 128 / 256 B stride = +17.0 / +4.4 / **-0.5%** at 8K; L3-only vs L1+L3 cost the same at 64 B; 4x fewer *demand* messages for the same lines was slower. The 64 B wide block prefetch asks for each 256 B L3 line 4 times on 1 KiB rows. B70 L3 line = 256 B (`clinfo`); 512 B stride kept a third of the gain.
- **`PREFETCH_V_L3`** (d_max 512, `39bdb18aeb`): `__builtin_IB_lsc_prefetch_global_uchar(V + row*pitch + line*256, 0, 2u /*L1UC_L3C*/)`, one per 256 B line dealt over all lanes, rows clamped to `k-1`, at the `PREFETCH_V` site; removed a 16K-20K L3-miss knee (one KV head's K/V > 24 MiB L3): -11% at 32K, -0.4..-1.6% at 2K-8K. At d_max <= 256 (K/V fits L3) it was 1-5% slower: prefetch only where the working set exceeds the cache. Next-tile K prefetch did not pay at any stride.
- Declaring `__builtin_IB_lsc_prefetch_*` in a kernel file: cache control as `uint` (compatible with `tile_ops.cl`'s `enum LSC_LDCC`), guarded by `!defined(GPU_OCL_TILE_OPS_H)`; an enum cannot be defined there (the codegen `#undef`s file macros between the kernels of a batch).

## 5. Split barrier
- Where to use it: independent work between `arrive` and `wait` that **does not read the SLM that barrier protects** (A_tile rescale by alpha `sdpa_ocl.cl:842-868`, V first-tile prefetch `sdpa_ocl_decode.cl:727-732`).
- Do not change the barrier count/placement. A conditional skip (the `continue` in `SV_TRIM`) keeps the barrier outside the loop.
- Measured standalone effect: 0.7%p of the 2.1% decode V prefetch gain. The standalone effect of the plain prefill split barrier is unmeasured (ASSUMED). `MAX_BARRIER_V_PREFETCH` was removed because its gain was unproven.
- PA PREFILL (`MAX_SPLIT_BARRIER`, 2026-10-23): S_max barrier → arrive / V prefetch / wait: -4.5% on top of the prefetch, +5.4% when reverted (qwen3-8b 8K); 0% with nothing in the window (d_max 512). MEASURED, B70 pinned.

## 6. KV cache layout decisions
- A K d-major page (`[head, block]`) has row=32 B < 64 B, so it cannot be block-read **as declared** -> per-key SIMD-1 scalar gather. Switching to token-major `[block, head]` reuses the prefill geometry (ISA: head 128 mixed total ops -49%, gather 128->0). Since 2026-10-27 the f16 page is block-read in place instead (§3, a 64 B pitch re-declaration + 32b transpose), at token-major speed without the layout contract change.
- The moment you change the layout, **every consumer** (writer, rotate, reorder, adaptive_rkv, decode, MIXED, test harness) is affected. green suite != layout-correct: check whether there is a test that reads the cache back (the rotate 512-pair corruption was invisible in the default suite).
- The decision is made in one place (`transformations_pipeline.cpp`); consumers derive from the cache's physical shape. A mismatch throws/rejects.
- i8 BY_TOKEN row pitch is `h` (not `h+4`: the +4 is the scale/zp array at the end). u4 K rows are not aligned to `h/2` (the `12h` page fits the INT4 allocation byte for byte).
- The price of token-major: the kvup writer dirties 16 sectors per thread (d-major 2). u4 head-256 generate +12% (MEASURED B70); 4 fixes all rejected by measurement (sector-sharing WG, unroll, fewer partitions). store cost ~ total sector writes. Store ORDER across threads does matter (2026-10-27, IGC 2.41): the 16 partitions writing the same 128 B rows in the same order queued; starting each at row pair `partition % 8` took GENERATE 5.75 -> 5.03 us (probe) / 10.1 -> 9.5 us (gemma-4-26b, master 9.2). Head 256 only: +3-8% at head 128, +5% at 512 (ch02 §10.3).
- Writer optimization traps: lane-varying loop variable -> registers lost (1.8x), put guards at the **call site** (in the helper body the unroll is lost, 2.3x), preserve type promotion per expression.

## 7. OOB / dynamic shape risks
- 2D block OOB = HW 0-fill, **global scalar OOB = CL_OUT_OF_RESOURCES or garbage** (Xe2). 
- A dynamic mask has its kind inferred in the JIT, so re-check runtime `MSK_D2==1`/`MSK_D3==1` in the kernel (static ones fold, so the codegen is identical).
- NaN in unwritten slots: even if the scores are masked with -INF, on the V side force scale AND zp to 0.
- When the K/V source split per subgroup within one k0 tile (page split), only V gave wrong results. Fixed with a **WG-uniform exact split + chunk clip** (mechanism not identified).
- Test data hides misreads: N(0,0.1) makes softmax uniform so Q/K misreads are invisible (`logit_scale_gain`); a power-of-2 LCG makes every row identical. Check data sensitivity before judging a wrong result.

## 8. Verification procedure (prepare commands for the user to run)
1. Test hypotheses first with an `SDPA_OCL_*_2D=0/1` env A/B (no rebuild). Change **only one thing** on the same build/workload. Controls on both sides are mandatory.
2. ISA/spill comparison via ocloc splice (no rebuild, `04-spill-isa-profiling.md`).
3. Correctness: `ov_gpu_unit_tests --device_suffix=1` (`smoke_paged_attention/*`, `paged_attention_feature_pad_test`, `sdpa_block2d_gate`), and classify with cliloader which kernel actually ran.
4. Performance: cliloader average ns + call count, state the device. Trust probes only for large ratios (small ratios on real models).
5. Report format: separate measured/assumed, regression risks (dynamic padding, DG2, pre-Xe2 fallback), test coverage gaps.

## 9. Common misjudgments (do not re-derive)
- "`load.ugm.d32x8t (1|M0)` is a scalar gather" -> no, it is a block load where one address spreads 32 B across the lanes.
- "Fewer reads is faster" -> failed 3 times on i8 V/K; geometry and dequant movs were the bottleneck.
- "Removing spill makes it faster" -> 256 GRF was +47% slower on head-72, +70% slower on u4 head-64.
- "Occupancy/SLM headroom is throughput" -> missed twice.
- "The prefetch hurts because the tile is big / it pollutes L1" -> no: it was 4 requests per 256 B line; at one request per line the same prefetch won 11% (ch02 §7.6).
- "Fewer messages for the same lines relieves Send stalls" -> 32-row x 2-block K/V reads were 4-14% slower.
- "A wider read that wins on one tiling/model is a safe default" -> the 16-row pair reads won 21-31% at d512 (Send-bound), but Q_slm lost at d128 and on a 128-key window, and the K pair alone flipped sign with the kernel variant (gpt-oss sink +1.2-1.8%). Sweep 32/1K/2K/4K/8K/32K and every variant (window, sink, image), per kernel hash.
- "Concurrent workgroups thrash L3; co-schedule the heads of a KV group" -> 4-7% slower at every length, knee unchanged.
- "d-major cannot be block-read" -> wrong: the f16 page is block-read through a 64 B pitch re-declaration + 32b transpose with a permuted depth (HW-verified and shipped 2026-10-27, ch02 §4.5); the i8 BY_CHANNEL 80 B-pitch variant (ch02 §4.4) is still unverified on HW.
- "A branch that only fixes the last page is free" -> it made IGC copy the DPAS operand at the merge (16 movs per depth tile, +11%); fold the fix into the producing instruction.
