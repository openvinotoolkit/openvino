# sdpa_ocl: OpenCL SDPA and paged-attention kernels for Xe2

`sdpa_ocl` is a plain OpenCL C implementation of scaled dot-product attention built on the Xe2
DPAS (`cl_intel_subgroup_matrix_multiply_accumulate`) and 2D block IO
(`cl_intel_subgroup_2d_block_io`) builtins. It replaces the oneDNN-microkernel based `sdpa_micro`
on Xe2 and later, for both the `scaled_dot_product_attention` primitive and the prefill/mixed
stages of `paged_attention`. A separate kernel, `sdpa_ocl_decode`, covers the paged-attention
GENERATE stage.

This document holds the design rationale, derivations and measurements behind the two kernels.
The code comments only keep the short "why" at each site.

The end-to-end performance investigation, including the B70 Xe2 gap closure and the later DG2 raw-kernel study, is summarized in [the OpenCL performance guide, chapter 06](ocl_perf_guide/06-performance-gap-investigation.md). That chapter distinguishes standalone kernel measurements from product integration and end-to-end evidence.

## Where it runs

| Primitive / stage | Generator | Kernel | Gate |
|---|---|---|---|
| SDPA prefill and single-token | `SDPAOclGenerator` | `sdpa_ocl.cl` | `sdpa_opt.cpp`: `SDPAOpt::dpas_sdpa_op_supported()` and `SDPAOclGenerator::supported()` |
| PA PREFILL / MIXED | `SDPAOclGenerator(prefill)` | `sdpa_ocl.cl` | `paged_attention_opt.cpp`: `choose_dpas_backend()` at construction, `can_use_sdpa_ocl_for()` per dispatch |
| PA GENERATE | `SDPAOclDecodeGenerator` | `sdpa_ocl_decode.cl` | `SDPAOclDecodeGenerator::supported()` |

Whether the DPAS kernel serves plain SDPA and PA PREFILL/MIXED is decided per device,
`paged_attention::sdpa_ocl_selected()`: `sdpa_ocl` on Xe2+ XMX (and xe_hpg with `TEST_USE_SDPA_OCL_HPG=1`, below), the
opt kernels on every other device (xe_hpg by default, xe_hpc, non-XMX). `sdpa_micro`, the oneDNN-derived DPAS kernel
that served the other XMX parts, was removed on 2026-11-02 together with `TEST_USE_SDPA_OCL`, which selected it; what
that changed off Xe2 is listed in "Known issues". The `sdpa_ocl` lane needs XMX and a oneDNN build (the
`ENABLE_ONEDNN_FOR_GPU` guards were kept, although the kernel no longer uses oneDNN), but not the oneDNN microkernel
support (`cldnn::query_microkernels_supported()`); `OV_GPU_ENABLE_SDPA_MICRO=0` (the config's `enable_sdpa_micro`, a name
from `sdpa_micro`) turns it off. Where `sdpa_ocl` refuses an op, the opt kernels run. The generator is created in every
impl's default constructor, the params constructor adds its stages, and every later decision reads `has_stage()`: a
loaded or cloned impl is default-constructed and only gets `_order` (indices into `_stages`) back, so it must not
re-derive the kernel from the device or the environment.

`TEST_USE_SDPA_OCL_DECODE=0` disables `SDPAOclDecodeGenerator`, so GENERATE falls back to the
`paged_attention_opt` single-token kernels. The K page layout does not follow it: since 2026-11-06 those kernels read
the token-major page of every kind ("Paged-attention cache layouts").
`SDPAOclGenerator::supported()` requires Xe2 or later (xe_hpg only during the bring-up, below) and
f16/bf16 Q and output. Plain SDPA
additionally requires equal K and V head sizes (`SDPAOpt::dpas_sdpa_op_supported()`), paged
attention does not.

Paged attention adds the PREFILL and MIXED stages together, so both variants are compiled even
when only one is dispatched. The MIXED stage reads the token-major K page only (since 2026-11-02),
so `can_use_sdpa_ocl_for()` keeps MIXED off `sdpa_ocl` on every d-major page: there it still
compiles, as the token-major reader, and `pa_multi_token` runs ("Paged-attention cache layouts").

Reachable configurations on Xe2 with XMX and no environment overrides:

| Kernel | Default | With user settings | Compiled, not dispatched |
|---|---|---|---|
| `sdpa_ocl` | PA PREFILL f16 (MICRO_MATH=1 except `d_max` 512); PA MIXED i8/u4 BY_CHANNEL token-major with raw f16 Kc/Vc; plain SDPA f16 prefill and single-token | PA MIXED f16 and i8 BY_TOKEN, token-major; plain i8 KV compression (always asymmetric + planar); plain bf16 | PA MIXED on any d-major K page (`OV_GPU_PA_K_TOKEN_MAJOR=0`, or a page the layout rule leaves d-major), plain int4 |
| `sdpa_ocl_decode` | PA GENERATE i8/u4 BY_CHANNEL token-major | f16 and i8 BY_TOKEN, token-major | — |

### xe_hpg bring-up (DG2, SG8)

xe_hpg (SG8, no 2D block IO) can take the `sdpa_ocl` lane only with `TEST_USE_SDPA_OCL_HPG=1`; without
it nothing about xe_hpg changes. Only the kernel arms that exist are served: `supported()` requires the
op's `HpgTier` bits (`sdpa_ocl_hpg.hpp`, `hpg_tier_required()`) to be set in `kHpgTiersReady`, which
has `PLAIN_F16_STATIC` (plain f16 SDPA, static shape, more than one query, no mask / causal / sink / runtime
scale), `PLAIN_I8`, `PA_PREFILL`, `PA_MIXED_F16`, `PA_FEATURES`, `PA_I8_TOKEN`, `PA_I8_CHANNEL` and `PLAIN_EXT` (the rest of uncompressed plain SDPA: bf16, a mask, causal, a sink, a runtime scale, a dynamic
shape, one query; the mask, causal, sink and softmax code is lane = query and shared with SG16, only the 16-key
`mask_full` vector of a full 2D mask is wider than the subgroup). Every other op is refused and runs the opt kernels,
as every op does with the switch off (until 2026-11-02 a temporary routing sent both to `sdpa_micro`). The SG8 arm (`SG8` in
`sdpa_ocl_config.cl`, i.e. `SUBGROUP_SIZE == 8`) feeds the DPAS an `int8` A operand, two halves per lane
(lane l = reduction elements 2l, 2l+1): the K tile is read by `k_tile_dword` (a dword block read per key row, a
two-ushort fallback for an unaligned or tail tile) and the S*V A operand is one 8-row dword block read of
`S_slm`; everything else (Q staging, softmax, V gather, output) is lane = query / value column and is shared with
SG16. `SDPA_OCL_NEG_SG8=1..3` (SG8 only) breaks one of the three mappings on purpose for the sharp-softmax test,
`=4` forces the unaligned-K fallback, `=5` / `=6` read the full / per-key mask with the wrong lane width / one lane off,
`=7` / `=8` / `=9` count the sink in every subgroup / read qq_bias transposed / end the sliding window one 16-key block
short (the PA feature tests must fail each), `=10` reads plain-SDPA K / V of batch 0, KV head 0 in the xe_hpg
PREFILL kernel (`smoke_sdpa_hpg_rows` must fail it), and `=11` / `=12` / `=13` read that kernel's mask from the first
query's row / at batch 0, head 0 / with the two 8-key halves of each 16-key block swapped (its mask cases must fail them).
`PA_PREFILL` and `PA_MIXED_F16` (every PA op needs both, since both stages compile) serve paged attention on an f16
cache. PREFILL reads the K/V inputs like plain SDPA; at heads 64, 96, 128, 256 and 512 it runs its own kernel ("xe_hpg PA
PREFILL kernel" below). MIXED reads the pages: the K page through the `int8` arm of `pa_k_tile_gather` (lane = channel pair, dword
loads: a token-major page gives the two channels of a key), the V page through the shared `pa_v_tile_gather` (lane =
value). xe_hpg K pages are always d-major (`k_token_major_device()` needs Xe2+), and since 2026-11-02 the MIXED stage
reads the token-major page only, so on xe_hpg MIXED runs `pa_multi_token`: `PA_MIXED_F16` now only lets the SG8 MIXED
stage compile with PREFILL. Its d-major arm (the two keys of a channel in one dword, repacked with 16-bit shifts) was
removed with it. The token-major arm runs on DG2 on a page a test forces token-major: `smoke_paged_attention_mixed_tm`
(heads 64 / 128 / 256 / 96 / 512, 21 cases each: past lengths on and off a page boundary, new tokens around a 32-query tile,
a long past, batches mixing MIXED, PREFILL and one-token subsequences, GQA / MQA, windows that start in the past, a
sink, all with a sharp softmax, and three cases whose page tails are NaN) passes on the HPG lane and asserts that
`sdpa_ocl` ran; `SDPA_OCL_NEG_SG8=36` / `37` / `38` (the two keys of a K pair swapped / the V pages of adjacent blocks
swapped / the odd V row of a pair read past k) must fail it: 36 and 37 fail 100 of the 105 cases, 38 exactly the 15
NaN-poisoned ones. 36 passes only the window-1 case (each query sees one key, whose softmax weight is 1 whatever K
reads), 37 only the one-page case (12 keys: both 16-key blocks take page 0, so the swap changes nothing).
The V page index of each 16-key block comes from one uniform load per block (`pa_v_pages`, `#if SG8`): the SG16 form,
a per-lane load of page `lane` and `sub_group_broadcast`, loads the same entries but gave wrong pages on DG2 under
some code shapes (MIXED with one KV head, heads 48 and 96), and which shapes failed moved with unrelated edits of the
S*V loop. The PA features (`PA_FEATURES`: a sink, token_type_ids, qq_bias, a sliding window, k != v, a runtime scale)
are lane = query code shared with SG16 and needed no SG8 arm; `smoke_paged_attention_feature_*` checks each one on an
f16 cache with a sharp softmax and asserts that `sdpa_ocl` ran the case where it serves the op (PREFILL; MIXED runs
`pa_multi_token` while xe_hpg pages are d-major). `PA_I8_TOKEN` (an i8 BY_TOKEN cache) has SG8 page readers in the
generic kernel: `pa_k_tile_q_gather` reads a lane's two channels of a key as one ushort of the token-major page and
dequantizes them with the key's (scale, zp), broadcast from lane kk of its 8-key group (`pa_k_comp_by_token`);
`pa_v_comp_fold` folds the V scales of a lane's two keys (one dword of the comp array) into the S*V A operand and returns
both zps as a `half2`, which `pa_v_tile_q_gather` broadcasts per key pair from lane kp. A compressed MIXED reads its new tokens from
the f16 inputs (`PA_CUR_KV_F16`, exact, as on Xe2): Kc with `k_tile_dword` (the plain-SDPA K read, its rows shifted by
past_len; any pitch or base, through its aligned-dword fallback) and Vc with `vc_tile_gather` (lane = value, one ushort
per key); `SDPA_OCL_PA_CUR_F16=0` reads them back from the cache instead, a lossy round trip. Until xe_hpg pages are token-major only PREFILL runs on `sdpa_ocl`; `smoke_paged_attention_comp_tm` (cases
0-16 on an i8 BY_TOKEN cache: every token of a page with its own scale and zp, a sharp softmax, partial 8-key groups, page
boundaries, heads 64 / 96 / 128 / 256 / 512, GQA / MQA, a window, a sink, k != v) forces the page token-major and asserts
that `sdpa_ocl` ran; `SDPA_OCL_NEG_SG8=19..22` fail 15 / 16 / 16 / 16 of them (all but the PREFILL-only case 13, and for
19 also case 5, whose K comp is uniform). `PA_I8_CHANNEL` (an i8 BY_CHANNEL cache) reads its token-major page in the
same SG8 gather: the page's comp is one (scale, zp) dword per channel, which a lane reads for its two channels next to
their bytes, per 8-key row block (in the depth loop, which SG8 does not unroll; `pa_k_comp_by_channel` is the SG16 lane ==
channel form). A row block at/past k reads no comp, since its page index is a clamped 0. The V page is per token, as
for `PA_I8_TOKEN`. `smoke_paged_attention_comp_tm` cases 17-30 distort K per channel (even / odd channels 4x apart,
offsets growing with the page) and V per token; `SDPA_OCL_NEG_SG8=23` / `24` (the pairs of a lane's two channels swapped /
scale and zp of a pair swapped) fail all 13 of them that read a page (all but the PREFILL-only case 27), and so do 20 / 21
/ 22. Cases 31 / 32 pad the inputs so that Kc is 2 B off a dword / has an odd row pitch (`k_tile_dword`'s fallback);
`SDPA_OCL_NEG_SG8=27` / `28` (the next new token's Kc / Vc row) fail 23 / 21 of the 33 cases: the others are
PREFILL only (13, 27) or keep a shift of the q newest of past + q keys inside the tolerance (3, 5, 9-11, 24, 25, 29, for
28 also 4 and 15). With `SDPA_OCL_PA_CUR_F16=0` cases 31 / 32 fail: the KV cache writer misreads rows 2 B off a dword
(Known issues). `PA_U4` is not ready: such an op runs the opt kernels, and `sdpa_ocl_config.cl` still `#error`s on
the u4 page readers and on the SG8 permuted-depth (u4) Kc read.
The PA unit tests ask `tests::pa_hpg_tiers()` for the bits of their case, so their expected backend follows the mask.
`PLAIN_I8` is plain SDPA on an i8 KV cache (always asymmetric, planar): the K tile comes from `k_tile_dword_i8`, which
dequantizes `(q - zp) * scale` per element from the hoisted per-key scale/zp into the same dword A operand (byte
loads, no alignment assumption), and V goes through the shared scalar `v_tile_gather`. A plain int4 cache, and i8 data
without the scale/zp tensors, are refused by `supported()` on xe_hpg. `NEG_SG8=1` swaps the head-dim pair there too.
The K dword read is guarded at run time (`k_dword_ok`: even row pitch and a 4 B aligned base), so a dynamic shape or a
sliced K view needs no host proof. The SG8 host jit is
still generated: the 2D block and 1D page flags are forced to 0 whatever the `SDPA_OCL_*_2D` overrides
say (`block2d_io_allowed()`), the tiling is 16 x 16 keys x queries with 4 x 2 subgroups up to head 128 and 8 x 2 / 16 x 2
at 256 / 512, so the S*V accumulator stays at 64 GRF per thread (the xe_hpg PA PREFILL kernel has its own tiling),
`MICRO_MATH` is off, the generic kernel's KQ depth loop and S*V key loop are not unrolled (`SG8_KQ_UNROLL` /
`SG8_SV_UNROLL` = 1), 256 GRF is
always requested, and `tiling_fits_device()` (used by both `choose_config()` and `supported()`, so
the `SDPA_OCL_KQ_*` overrides are judged too) keeps local memory within 64 KiB and the work-group
within 1024 work-items. `sdpa_ocl_config.cl` `#error`s on SG8 (`SUBGROUP_SIZE != 16`) at every spot
that is silently wrong there (2D block IO, the compressed-page and `PA_CUR_KV_F16` K/V readers), because DG2
compiles a `short8` DPAS operand without a DPAS.

Unrolled whole, as on Xe2, the two loops made IGC 2.41 load K / V of every depth tile / key block ahead, and the generic
SG8 kernel spilled at every head (cliloader on DG2: PA PREFILL head 64 / 96 / 256 6.5 / 9.4 / 25-32 KB, MIXED head 96 / 128
7.9-10.8 KB, plain SDPA head 64-256 4-28 KB); with 4 x 2 subgroups at head 256 / 512 the accumulator alone is 128 / 256
GRF (ocloc: 27 / 52 KB with the loops rolled), and `MICRO_MATH`'s second accumulator doubles it. With the three changes
none of those kernels spills, heads 64-512, PREFILL and MIXED, PA and plain. Their speed is not measured yet ("xe_hpg
follow-ups").

To look at the SG8 jit and compile it offline without a DG2: build with `ENABLE_DEBUG_CAPS`, run a
test group with `OV_GPU_ARCH_OVERRIDE=xe_hpg TEST_USE_SDPA_OCL_HPG=1 SDPA_OCL_HPG_TIERS=all` and
`OV_GPU_DUMP_SOURCES_PATH` (`test/sdpa_ocl_gtests.sh dump`), then `sdpa_ocl_ab.py corpus` and
`sdpa_ocl_ab.py hpg --device dg2 --grf256` (the `dpas` column must be nonzero for every in-tier kernel: DG2
drops a SG16 DPAS silently). The forged arch makes every kernel choice believe in xe_hpg,
so the results of such a run mean nothing; only the dumped sources do.

### xe_hpg PA PREFILL kernel

Every speed against `sdpa_micro` in this section was measured in real models on the same A770 before this branch removed
it (2026-10-30 to 11-04); the xe_hpg port re-measures against the opt kernels.

On xe_hpg, PA PREFILL at heads 64, 96, 128, 256 and 512 (f16, k == v, no alibi / runtime scale, with or without a sliding window,
a sink and token_type_ids, and Q / K / V rows the kernel can read in dwords) runs `sdpa_ocl_hpg_pa_prefill.cl` instead of
the generic body:
`SDPAOclGenerator`'s `hpg_pa_prefill_ok()` picks it, `hpg_pa_prefill_tiling()` gives it its tiling, and the jit
constant `HPG_PA_PREFILL` makes `sdpa_ocl.cl` include it in place of its own kernel (the constant exists only then,
so no other kernel's text changes). It keeps the generic loop (online softmax over workgroup key tiles, the causal
and sliding-window bounds, the `S_max_slm` / `S_sum_slm` exchange) and changes what DG2 needed to keep its DPAS fed:

| | Generic SG8 kernel | xe_hpg PA PREFILL kernel |
|---|---|---|
| Tiling | 16 x 16 keys x queries, 4 x 2 subgroups (64 x 32) up to head 128, 8 x 2 / 16 x 2 at 256 / 512 | head 64: 32 x 16, 4 x 2 subgroups (128 x 32); head 72 (plain SDPA): 32 x 16, 5 x 2 (160 x 32); head 96: 32 x 16, 6 x 2 (192 x 32); head 128: 32 x 16, 8 x 2 (256 x 32); head 256: 16 x 16, 16 x 2 (256 x 32); head 512: 16 x 16, 32 x 1 (512 x 16); each S*V subgroup holds the 16 queries of its KQ subgroup and 16 value columns |
| KQ | DPAS(K, Q): K repacked to the `int8` A operand | DPAS(Q, K): Q staged so that one `ulong4` SLM block read is the A operand, K read straight from the input as the B operand (one 32 B read per key row); the scores are transposed to lane = query through 64 dwords of SLM per subgroup |
| S*V | DPAS(P, V), V gathered per value | DPAS(V, P): V rows read as dwords and interleaved per key pair, the output tile is lane = query |
| V latency | loads in the S*V loop | the first `HPG_PA_V_LEAD` (6) key blocks of V are loaded before the wait of the P barrier and the loop keeps that lead: IGC 2.41 sinks loads that nothing pins toward their use, which left the V latency exposed |
| Mask | per element | skipped for a subgroup tile wholly below the diagonal and inside the window; with a window, causal and window are one unsigned compare (`0 <= query - key < window`): the two signed compares spilled 2.5 KB on IGC 2.41 |
| Workgroup order | subsequence order | heaviest, lightest, second heaviest, ... query block by the keys it visits, causal and from its first query's window start (`pair_query_blocks`, `prepare_internal_buffers()`) |
| Softmax rounding | the default rounding (`MICRO_MATH=0`, which gives `sdpa_micro`'s only with Xe2's table tiles) | the same |

Rows past the subsequence are clamped to its last key or query, so no read leaves it, and their scores are masked to
-INFINITY (P = 0 times a finite V). The sink and the image groups of token_type_ids follow the generic kernel: the sink
seeds the online-softmax state (max = the sink logit, sum = 1 in the subgroups of the first key column only;
`SDPA_OCL_NEG_SG8=7` counts it in every subgroup), and the causal and window bounds reach the image group of the
workgroup's last query and the group its first query's window cuts. Each query keeps its causal (and window) keys and its
own image group, one interval since both hold the query, so its mask is the same single unsigned compare as the
window's: a separate group test spilled 0.6-1.9 KB with a window (IGC 2.41). A static feature padding must keep every
head and token start a multiple of 4 B; a dynamic one is taken on trust, as by the block2d fixup tier (a crop view of a fused QKV tensor at head-size offsets has
it), except Q's: the kernel tests it (`q_dword_ok`) and stages a Q 2 B off with two ushort loads per dword
(`SDPA_OCL_NEG_SG8=4` forces that path; the suites must still pass). A dynamic K / V padding 2 B off reads wrong values,
as in the disabled `paged_attention_feature_pad_residual_test`. `SDPA_OCL_HPG_PA_PREFILL=0` switches the kernel off (the generic one runs), `SDPA_OCL_HPG_PA_V_LEAD=<n>` changes the
V lead, and a `SDPA_OCL_KQ_*` override also falls back to the generic kernel.

The kernel also serves plain SDPA (its PREFILL stage) without causality, a runtime scalar mask, a runtime scale, a sink
or a compressed / indirect KV cache, at the same heads (`hpg_plain_prefill_ok()`; a constant scale input is a literal; a
finite static scalar mask adds the same value to every score of a row, which the softmax ignores, so the kernel leaves it
out: CLIP's vision encoder has 0, in phi-3.5-vision, which took the generic kernel at 7.2x `sdpa_micro` before and takes
400 us against 372 now): the loop without the causal bound, so every workgroup
visits all k keys and only the keys past k are masked, rows from the 4D layout (`QRY_OFF` ..., batch =
`get_group_id(2)`, a workgroup per 32 queries), q and k from the arguments. Its Q, K, V and output must be rank-4 with the
head dimension innermost and unpadded, so every row starts at a multiple of the head size whatever the other axes'
padding (a crop view of a fused QKV tensor pads another axis); a padded head dimension takes the generic kernel.
`smoke_sdpa_hpg_rows` checks its rows with sharp data: GQA / MQA, batches, q != kv, `[b, s, h, d]` inputs and output,
K / V padded on the other axes, dynamic shapes, 1.1-1.5K tokens, heads 96, 72 and 512, static scalar masks of 0 and -3;
`SDPA_OCL_NEG_SG8=10` must fail every case with two
batches or KV heads. `SDPA_OCL_HPG_PLAIN_PREFILL=0` switches it off for plain SDPA. A tensor mask (f16, bfyx, unpadded)
is added in the raw score domain as in the generic kernel, 8 keys of the lane's query row per 8 x 8 score block just
before the block is used (`hpg_mask8()`: row 0 of a per-key mask, key 0 of a per-query one, batch and head broadcast
through `MSK_OFF`; one 16 B load when the rows are whole dwords, two-byte loads otherwise, which `SDPA_OCL_NEG_SG8=4`
forces), and no tile skips it; the generic kernel's float16 mask tiles would hold 64 GRF. The mask cases of
`smoke_sdpa_hpg_rows` (per key, full shared by the heads, full per head, full shared by the batches, with GQA and
batches) block about one key in four by a hash of every index the mask has: a first pattern of period 4 keys passed
`SDPA_OCL_NEG_SG8=13`, since an offset of whole 8-key blocks kept its blocked set. `=11` / `=12` / `=13` fail exactly
the cases whose mask they can see. With the tiling of the PA kernel (DG2 real models, device time per call, two
interleaved pairs, same method as above) flux.1-schnell's transformer attention (head 128, 24 heads, 1,536 tokens, no
mask) took 1,166-1,173 us against `sdpa_micro`'s 1,520-1,530 (0.77x; the generic kernel was 5.2x), its T5 encoder (head
64, 64 heads, 512 tokens, a mask per head) 430 against 570 (0.755x; was 2.9x), its CLIP encoder (77 tokens, a mask
shared by the heads) 48.8 against 61.0 (0.80x; was 1.55x), and gemma-4-e2b's vision encoder (head 64, a per-key mask)
1,178 against 1,613 (0.73x; was 4.1x). Plain SDPA also takes head 72 (phi-4-multimodal's vision encoder), which is not a
multiple of 16: `HPG_TAIL` = 8, so the K row and the Q row of the last depth tile are read as 16 B and the rest is zero
(no read leaves the head), Q staging takes that tile alone (`DKS_ACTIVE` 5 is odd), and the workgroup is 5 x 2
subgroups (160 x 32) whose S*V split covers 80 value columns: the last value subgroup reads columns 56-71 (`sv_col`) and
stores only 64-71 (`SDPA_OCL_NEG_SG8=18` stores them unshifted and must fail every head-72 case of
`smoke_sdpa_hpg_rows`). PA keeps heads that are multiples of 16 (the K page reads have no tail). phi-4-multimodal's
vision encoder (16 heads, a mask) took 2,064-2,067 us against 2,857-2,862 with 5 crops (0.72x) and 861-863 against
1,170-1,172 with 2 (0.74x); the generic kernel was 10,354 / 4,182 (3.6x).

The kernel is the `skip_e1p6` design of the DG2 study: a standalone harness on an A770 (driver 26.35, IGC 2.41.5),
paired against the `sdpa_micro` binary of the same driver at head 128, 2 / 2 heads, with a CPU double-precision oracle
and NaN-poisoned outputs, at queued 32-deep and with 200 us gaps. With a V lead of 4 it passed the s364 gate (9 shapes
from 4129 to 8193 tokens, single and multi-subsequence, sharp data) 18 / 18 with a worst paired ratio of 1.0136; with
the lead of 6 it passed 18 / 20 on the ordinary s360 lengths (2048 and 2049 tokens with gaps at 1.034 / 1.030) and
26 / 28 on the multi-subsequence s361 set. Without the lead, the worst s364 row was 1.06x `sdpa_micro` on IGC 2.41. Prompts up to about 2K tokens are slower than `sdpa_micro` with this tile ("xe_hpg follow-ups"). The
product kernel itself has not been timed yet: that waits for a DG2 that is not shared.

The other heads' tilings were picked by device time per call in real models (llm_bench, f16 KV cache, cliloader, A770,
driver 26.35, 2026-10-30), against `sdpa_micro` in the same run. Head 64: 4 x 2 subgroups of 32 x 16 (128 x 32) took
165.7-168.8 us at 1K tokens on minicpm4-0.5b against `sdpa_micro`'s 174.9-177.1, and 264.9-267.8 us on llama-3.2-1b
against 263.5-266.0 (1.004-1.005x in two interleaved pairs, a lower total per run; 32 tokens: 55.2-56.5 / 55.8-57.5
against 60.8-61.2 / 61.7); `sdpa_micro`'s own 64 x 128 workgroup tile was 1.24x, 128 x 64 0.98x / 1.04x. Head 256:
`sdpa_micro`'s 256 x 32 workgroup (16 x 2 subgroups of 16 x 16) took 532-540 us on gemma-2-9b against 472 (1.13x; 32
tokens 1.07x); 32 value columns per S*V subgroup spilled 4.7-9.4 KB in every variant (a V lead of 1 included), 512 x 32
took 558 us, and the V lead (4-6) and the KQ depth unroll (1-4) moved it within 2%; the KQ depth loop is unrolled by 2
there (`HPG_PA_KQ_UNROLL`). Head 512: 32 x 1 subgroups of 16 x 16 (512 x 16, 16 value columns per S*V subgroup, no
spill, 42 KB of SLM) took 537 us against 591-596 on gemma-4-e2b's global layers at 1K tokens (0.90-0.91x; 32 tokens
0.94x) and 1,437-1,455 against 1,517-1,537 on flux.1-schnell's VAE (plain SDPA, one head; 0.94-0.96x); the generic
kernel was 7.4x / 4.2x. 16 x 1 subgroups of 32 x 16 with 32 value columns spilled 6.1-6.6 KB (4.9 KB with a V lead of
1) and were 2.2-2.5x. The 512 x 16 tiling sits near the IGC codegen hazard in "xe_hpg follow-ups" (Known issues): a V
lead of 4 leaves one 8 x 8 accumulator tile as garbage now and then with a dynamically padded input, the default 6 passed
the 84 head-512 cases in five uncached runs.

With a sink and token_type_ids (same method, 1K and 32 tokens): gpt-oss-20b (head 64, sink, a 128-key window in every
other layer) took 450.4 us on its full layers and 238.5 on its window layers against `sdpa_micro`'s 453.4 / 266.4, and
62.3 against 64.3 at 32 tokens; gemma-3-4b-it (head 256, text prompts, so no image group) 287.7 against 340.8 and 64.3
against 84.2. With an image in the prompt (274 and 1,024 tokens), gemma-4-26b-a4b-it's head 256 window layers took 811.6
against 2,345 and 411.7 against 1,804; gemma-4-e2b-it's, in three interleaved pairs, 217.7-218.9 against 212.7-214.1 on
16 layers and 246.8-251.9 against 238.8-240.7 on 11 (1.02-1.05x; 0.96-0.97x at 274 tokens).

`smoke_paged_attention_prefill_h128` covers it: lengths around the 32-query and 256-key tiles, several subsequences per
call, GQA and MQA, padded rows (one padding that sends the op to the generic kernel), sliding windows (1, 17, 100, 256,
300 keys and one wider than the prompt), and every case again with a sharp softmax; `SDPA_OCL_NEG_SG8=9` must fail its
window cases but the widest. `smoke_paged_attention_prefill_heads` runs the same cases at heads 64, 256, 512 and 96,
plus the tile edges 127 / 128 / 129 at 64, 16 / 511 / 512 at 512 and 191 / 192 / 193 at 96.
`smoke_paged_attention_feature_sink` and `_bidir` hold its sink and image-group cases at the three heads (groups across
its query blocks and key tiles, from token 0, wider than the window, in a second subsequence); `SDPA_OCL_NEG_SG8=7` must
fail every sink case and `SDPA_OCL_BIDIR=0` every image-group case. The harness draws Q, K and V from one seed, so with
equal head counts each query's own key dominates a sharp softmax and a mask that keeps it hardly moves the output
(`SDPA_OCL_BIDIR=0` passed a 400-token case at head 256 with 2 / 2 heads): the wide image-group cases use GQA.
`sdpa_micro` fails `_bidir/9` (two subsequences, 4 / 1 heads): it reads token_type_ids by the position in the
subsequence, without the subsequence's start.

## Source layout

| File | Contents |
|---|---|
| `impls/ocl_v2/sdpa_ocl.cl` | The kernel: extension pragmas, batch-header includes, the helper-header includes and the kernel body |
| `impls/ocl_v2/sdpa_ocl_config.cl` | Every compile-time macro: dtype and DPAS selection, derived tiling, DKS, mask/bidir predicates, PA page geometry, u4 labelling, 1D page mapping, expression macros, and all host/kernel invariants as `#error` |
| `impls/ocl_v2/sdpa_ocl_mask.cl` | Attention-mask helpers: the bidirectional image-group boundary scans and the full 2D mask tile |
| `impls/ocl_v2/sdpa_ocl_qk_load.cl` | Q staging, the per-k0 K hoists (plain comp, PA pages, PA comp, u4 1D page) and the K tiles (block reads, Kc, gathers) |
| `impls/ocl_v2/sdpa_ocl_v_load.cl` | V page base, Vc prefetch and the V tiles (block reads, comp fold and dequant, Vc, gathers) |
| `impls/ocl_v2/sdpa_ocl_decode.cl` | The decode kernel (self-contained) |
| `impls/ocl_v2/sdpa_ocl_hpg_pa_prefill.cl` | The xe_hpg PA PREFILL kernel (heads 64 / 96 / 128 / 256 / 512, and 72 for plain SDPA), included by `sdpa_ocl.cl` in place of its kernel when `HPG_PA_PREFILL` is set |
| `impls/ocl_v2/sdpa/sdpa_gen_ocl.{hpp,cpp}` | `SDPAOclGenerator`: gate, tiling (`choose_config()`), jit constants (one function per concern), arguments, dispatch |
| `impls/ocl_v2/sdpa/sdpa_gen_ocl_decode.{hpp,cpp}` | `SDPAOclDecodeGenerator` |
| `impls/ocl_v2/sdpa/sdpa_ocl_utils.hpp` | Host helpers shared by the two generators: environment switches, block2d rules, paged-attention page geometry, sink and I/O layout jit |

The kernels are embedded at build time by `common_utils/kernels_db_gen.py`, and a few of its rules
shape the header split:

- A non-batch `#include "x.cl"` is inlined once per kernel. A second include of the same file is
  dropped, and an include that cannot be resolved is silently replaced by nothing. Each header
  therefore ends with a sentinel (`#define SDPA_OCL_<NAME>_INL 1`) that `sdpa_ocl.cl` checks with
  `#error`.
- Every `#define` gets an `#undef` appended, except a bodyless `#ifndef X` / `#define X` pair. A
  classic include guard would therefore make the second kernel of a batched program skip the
  header, which is why the headers have none.
- Kernels share one OpenCL program when batched, so helper functions must be `FUNC(name)` /
  `FUNC_CALL(name)` decorated. `OV_GPU_MAX_KERNELS_PER_BATCH=1` hides name collisions.
- The `.cl` list is globbed at configure time: a new `.cl` file needs a CMake re-run.
- Comments never reach the runtime (the minimizer strips them), so a comment-only change is
  provably a no-op on the embedded text.

### Kernel helpers

The K/V/Q load variants live in the headers as `FUNC()` helpers; the kernel body keeps the
skeleton: setup, pointer bumps and surface fixups, SLM declarations, key-range arithmetic, the k0
loop control, the qB/pA SLM reads, both DPAS nests, masking and the running max, every barrier, the
softmax, the alpha select chain and the epilogue. Each call site picks its variant with a one-level
`#if` chain, and each helper is guarded by the same condition, so a configuration that does not
use a helper does not even see its tokens.

The helpers were extracted with the requirement that every configuration compiles to the same ISA
as the inline code (verified per step on the corpus, ocloc metrics identical). IGC and the LLVM
inliner make that stricter than "it inlines":

- **always_inline** (`SDPA_OCL_INLINE`). A plain `inline` helper is inlined too, but the module then
  goes through a different optimization pipeline: the first one-line helper changed the ISA of 121
  of 122 MIXED configurations, and an unrelated identity helper reproduced the same deltas on
  configurations that never called it. The IR after unification was identical modulo value names;
  only the later stages differed (the kernel arguments gained inferred `nocapture readonly`).
  `__attribute__((always_inline))` gives the inline ISA; the call still reaches IGC, it is just
  treated differently.
- **The inliner simplifies the helper body with the actual arguments while cloning it**, earlier
  than the inline code gets simplified, so anything that folds against an argument can move the
  result:
  - A cast of a widened parameter. The kernel's `lane` is `size_t` (the zero-extended
    `get_sub_group_local_id()`); `(int)lane` inside a helper folds `trunc(zext(x))` to the
    entry-block lane id, while the inline code later gets a fresh zero-extension at each use (2-4
    more movs hoisted to the entry). Helpers therefore take `size_t lane` for size_t arithmetic and
    `int lane_i`, converted by the caller, for int arithmetic. A `uint lane` instead made the size_t
    uses worse (more spills).
  - A branch on a parameter that is a compile-time constant in some configuration: under
    `SDPA_OCL_BIDIR_GATE=0` `bidir_active` is `true`, the branch was pruned at clone time and the
    CFG came out different (+16 `sync`). Such tests stay at the call site; the per-query bidir group
    loop stayed inline for this reason.
- **Coordinates are computed from their leaves inside the helper.** Passing `VcD_x0 + sg_j0_sv` as
  one argument hoisted it out of the unrolled loop and changed instruction order on the MIXED Vc
  read (+23 `sync`); passing `VcD_x0` and `sg_j0_sv` does not.
- **No arrays inside helpers.** The inliner brackets a helper's own arrays with lifetime markers the
  inline code never had; on the spill-bound u4 head-512 MIXED kernel that moved scratch allocation
  (7680 -> 7552 bytes). Scratch arrays (`kt`, `kw`, `kt_dt`, `vt_pa`, `vzp4`, `zpb4`, `v_pg`) are declared by
  the caller and passed as `__private` pointers.
- **Loop-carried outputs belong to the caller.** `q_pack` is written element-wise by the u4 staging;
  as the caller's loop variable SROA keeps its previous value as a phi, which a helper-local vector
  dropped (register allocation moved on u4 head 512). It is an out-parameter.
- From the start: arguments keep the caller's types, unrolled trip counts are macros (a runtime or
  select-based bound, or an early `return` in a helper, lost the unroll in
  `pa_kv_cache_update_ref.cl`), private arrays are indexed only by unrolled constants (IGC 2.41
  keeps an array with a runtime index in scratch even when the index is subgroup-uniform), pointer
  parameters carry an explicit `__private` / `__global`, and there are no structs or globals.
- Helpers that use a prelude macro which reads `shape_info` in dynamic-shape builds (`MSK_*`,
  `KEY_COMP_OFF`, `VAL_COMP_OFF`) take `OPTIONAL_SHAPE_INFO_ARG` and are called with
  `OPTIONAL_SHAPE_INFO_TENSOR`.

## sdpa_ocl

### Tiling

The host picks the tiling in `choose_config()` (per-head-size tables through `solve_tiling()`,
`SDPA_OCL_KQ_*` overrides) and `solve_sv_split()`:

- KQ: each subgroup computes a `kq_sg_tile_keys x kq_sg_tile_queries` score tile; a workgroup has
  `kq_sg_per_wg_keys x kq_sg_per_wg_queries` subgroups (`sg_per_wg`). `kq_sg_tile_keys` is 16 or
  32 (`k_mask` and `mask_tile` hold at most two 16-key entries, one i8 transform read covers 32
  keys).
- S*V: the same subgroups are re-split into `sv_sg_per_wg_scores x sv_sg_per_wg_values`, each
  owning `sv_sg_tile_scores` queries and `sv_sg_tile_values` value columns.

The invariants below are checked by `#error` because a violation does not fail, it silently leaves
part of the output uncomputed: the S*V split uses exactly `sg_per_wg` subgroups, its score tiles
cover `kq_wg_tile_queries`, and its value tiles cover `V_HEAD_SIZE`.

`solve_sv_split()` derives the S*V split for a fixed KQ tiling: the value and score tiles cover
`vd_max` and `kq_wg_tile_queries`, the tiles are multiples of the subgroup size and of the 8 DPAS
rows, and every subgroup's score rows nest inside the KQ query tile it rescales (`alpha[]`).
Preferring the widest value split reproduces the tuned tables for every `k_head_size ==
v_head_size`. It serves two cases:

- `k_head_size != v_head_size`. The table sized the S*V split for `d_max`, so it is re-derived for
  `vd_max`, leaving the KQ side (and with it the paged-attention query-block stride) alone. For
  `d_max >= 128` the query tile is 32, and a 32-wide V head splits at most 2 ways, which leaves
  score tiles of 4 or 2 rows, below the DPAS minimum. Tier 2 retries with a 64-key x 64-query KQ
  workgroup tile, which resolves every such pair to (16, 8, 2, 8). The hole was not benign: the
  MIXED fallback, `pa_multi_token`, then read a token-major BY_CHANNEL K cache d-major (it reads it
  token-major since 2026-11-06), so a rejected shape produced NaN rather than a slower result. Tier
  2 is unreachable for `k_head_size == v_head_size`.
- The `SDPA_OCL_KQ_*` overrides. `sg_per_wg` comes from the KQ side and is what the dispatch
  launches, so the S*V split has to be re-derived for the same count. An early sweep that overrode
  `kq_sg_per_wg_keys` alone left subgroups that no longer existed owning value columns, and their
  part of the output was never computed.

`SDPA_OCL_256GRF=1` compiles with 256 GRF, as `sdpa_micro` does whenever its microkernel needs more
than 128. The tiling sweep found that every configuration with `sv_sg_tile_scores >= 64` spilled
(3.8-19 KB on device) at 128 GRF. 256 GRF halves the threads per EU, so it only pays where it
removes spill.

`DKS = D_MAX / DPAS_K` where `D_MAX` is `K_HEAD_SIZE` rounded up to a power of two. At head 72
(`D_MAX` 128) only `ceil(72 / 16) = 5` of the 8 depth tiles hold data, 4.5 tiles' worth: the rest
would load nothing and add zero, 1.78x the necessary KQ DPAS and K loads. `DKS_ACTIVE` bounds the KQ
depth loop and the Q staging to the real tiles; `D_MAX` itself stays, because the S*V split
(`sv_sg_tile_values * sv_sg_per_wg_values == d_max`) and the alpha nesting are derived from it.
`sdpa_micro` never paid this, since its `ugemm_kq` takes the reduction extent as a runtime argument.
For u4, `DKS_ACTIVE` is rounded up to even (the depth permutation pairs tiles).

A caveat measured with ocloc on the head-72 configuration: shrinking the depth loop makes the 2D
block path strictly better, but makes the scalar fallback path spill more, monotonically:

| DKS_ACTIVE | instCount | spill bytes |
|---|---|---|
| 8 | 7043 | 2688 |
| 6 | 5808 | 13568 |
| 5 | 5631 | 16064 |

It only matters where `DKS_FULL < DKS` and a scalar fallback is still taken.

### Paged-attention PREFILL tiling

PA PREFILL on Xe2 at `d_max` 64, 128, 256 and 512 does not use the table row: `solve_pa_prefill_tiling()`
gives it the workgroup of `sdpa_micro`'s PA prefill config for that head size, compiled with 256 GRF (one
workgroup per Xe-core). MIXED has its own tiles at `d_max` 128, 256 and 512 ("Paged-attention MIXED tiling");
plain SDPA (except f16 `d_max` 128 and, on Xe2, `d_max` 64 and 512: "Plain-SDPA PREFILL tiling" below), `d_max` 32
and xe_hpg keep the table. The PA query-block stride follows the stage, which is why `get_query_block_size()` takes it.

| `d_max` | KQ subgroup tile, subgroups | Workgroup, keys x queries | `sdpa_micro` | Local memory |
|---|---|---|---|---|
| 64 | 16 x 64, 8 x 2 | 128 x 128 | `xehpc_h64`, same | 53,760 B |
| 128 | 16 x 64, 16 x 2 | 256 x 128 | `xehpc_h128`, same | 107,008 B |
| 256 | 16 x 32, 8 x 4 | 128 x 128 | `xehpc_h256`, same | 102,912 B |
| 512 | 32 x 16, 8 x 4 | 256 x 64 | `xehpc_h512`, same | 100,608 B |

At `d_max` 128 the 128 x 32 table tile reads every K/V row once per 32 queries, 4x as often as this tile, and its
loop has a quarter of the DPAS per barrier. On qwen3-8b (32/8 heads, head 128, u4 cache, B70, 2800
MHz, cliloader device time of all 36 layers) it was 1.30x `sdpa_micro` at 2K tokens and 1.39-1.42x
from 4K to 32K. Since attention grows with the square of the prompt and the rest linearly, its share
of the first-token latency went from 13% (2K) to 66% (32K), and with it the e2e gap: +2.0% at 2K,
+6.0% at 4K, +10.5% at 8K, +17.3% at 16K, +24.0% at 32K. The same build with
`TEST_USE_SDPA_OCL=0` matched master, so nothing else contributed; `pa_kv_cache_update` was 15-18%
faster than master's.

The tile alone brought it to 1.03-1.14x. Four loop changes, on by default with these tilings only
(`add_prefill_pipeline_jit()`, one switch each; `PREFETCH_V` at `d_max` <= 256, `d_max` 512 has `PREFETCH_V_L3`
instead, see below), took it below `sdpa_micro`:

- `SOFTMAX_BRANCHLESS`: exp2 then `select()` instead of `ok ? exp2(..) : 0`, which IGC compiled as a
  branch around the exp2 of every (mb, qb) block, so the blocks could not interleave.
- `PREFETCH_V`: this tile's V rows are prefetched after the S_max barrier, where `sdpa_micro` does.
- `MAX_SPLIT_BARRIER`: that barrier split into arrive / wait around the prefetch, as in `sdpa_micro`.
- `CAUSAL_MASK_LAST_ONLY`: the mask loop runs on the last key tile only.

Per call at 8K, adding them in this order (`sdpa_micro` 7,481-7,494 us in the same sessions):

| Step | 8K us | vs micro | Switch off again |
|---|---|---|---|
| tiling alone | 7,980 | 1.065x | |
| + `SOFTMAX_BRANCHLESS` | 7,914 | 1.056x | +2.6% (measured with `PREFETCH_V`) |
| + `PREFETCH_V` | 7,555 | 1.008x | |
| + `MAX_SPLIT_BARRIER` | 7,218 | 0.965x | +5.4% |
| + `CAUSAL_MASK_LAST_ONLY` | 6,977 | 0.931x | +3.3% (+4.2% at 32K) |

The last one holds because a workgroup's queries all lie in its last key tile (the query tile
divides the key tile and both are aligned), so no earlier tile has a causally masked key. With a
sliding window an earlier tile also runs the mask loop while it holds a key the window drops for
the workgroup's last query (`k0 + SLIDING_WINDOW_SIZE <= wg_j0 + kq_wg_tile_queries - 1`); bidirectional
groups still disable it. Without it IGC flattened the per-block `causal_block_clear` branch into 64
compares and 64 selects per subgroup per tile, more with the window predicate: phi-4-mini (24/8 heads,
`SLIDING_WINDOW_SIZE` 262144, which no prompt reaches) went from 4,813 to 3,753 us per call at 8K
(`sdpa_micro` 4,191) and from 68,485 to 53,489 us at 32K (`sdpa_micro` 61,307). The PA suite rows
`{1000}` / window 300, `{700, 300}` / 256, head 96 `{520}` / 100, head 64 `{900}` / 200, head 256 and
head 512 `{700}` / 200 fail when the window term is dropped, and `{600}` / 4096 and the windowless head
256 / 512 rows do not; they are uncompressed, because a compressed cache relaxes the tolerance to
25e-3, which a missing window mask passes.

Result per call, against `sdpa_micro` in the same session (N = 3, two rounds, spread 0.2-0.4%): 2K
553.6 vs 550.0 us (1.007x), 4K 1,904 vs 1,984 (0.960x), 8K 6,977 vs 7,494 (0.931x), 16K 25,557 vs
27,808 (0.919x), 32K 101,281 vs 110,942 (0.913x).

None of the four changes a result, and with `MICRO_MATH` the `d_max` 64 / 128 / 256 tiles have `sdpa_micro`'s
key-tile boundaries, so their output is expected to be bit-identical to `sdpa_micro`'s (the MiniCPM4-8B capture
found all 32 layers equal with this tiling). Checked end to end: with `TEST_USE_SDPA_OCL_DECODE=0`
(so GENERATE runs the same kernels as master) the 32-token output MD5 equals master's for all 20
runs (5 prompt lengths x 4 iterations), while the table tiling differs in 15 of them.

At `d_max` 512 the table tile spilled 11,648 B at 128 GRF. On gemma-4-e2b (8/1 heads; 28 layers of
head 256 with a 512-token window and 7 global layers of head 512) its 512 layers took 34.4 ms per call
at 8K against `sdpa_micro`'s 9.15 ms (3.8x), and the first token was 2.27x master's at 32K. 256 GRF
alone brought the table tile to 1.03x, and the 128 x 64 tile with the loop changes except `PREFETCH_V`
to 0.94x. `PREFETCH_V` on its own made that tile 10% slower (a 128-key tile of 512-channel V rows is
128 KiB), so `v_prefetch` is set for `d_max` <= 256 only. That first tile was 128 x 64 because `xehpc_h512`'s
256 x 64 workgroup needs 32-key subgroup tiles, then a known issue, or 16 x 4 subgroups of 16 keys = 1,024
work-items, which a 256 GRF kernel cannot dispatch (`CL_INVALID_WORK_GROUP_SIZE`). It kept the table's 128-key
workgroup and 64-value S*V tile, so the online-softmax rescale points and the accumulation order were the table's: the gemma-4-e2b
32-token output MD5 equals the table tiling's in 20 of 20 runs, and `SDPA_OCL_MICRO_MATH=0`, which there
changes only the 512 layers, changes it. Per call against `sdpa_micro` at 2K / 4K / 8K / 16K / 32K:
1.006x / 0.956x / 0.931x / 0.962x / 1.012x.

The `d_max` 512 tile is now `xehpc_h512`'s 256 x 64 from 32-key subgroup tiles, without `MICRO_MATH` (`micro_math`
false in `solve_pa_prefill_tiling()`). `MICRO_MATH`'s second S*V accumulator cost the 128 x 64 tile 22% and the
256 x 64 one 21-26%: per subgroup the output tile is 512 values x 64 queries / 32 subgroups = 64 GRF, `A_tile1` is
another 64, and the ISA showed the S*V loop waiting on every V block read right after issuing it, where without
`A_tile1` it reads V one or two key blocks ahead (loop 1,021 vs 790 instructions, spill 1,344 B vs 0, 128 x 64 tile).
Splitting the S*V into value-block groups keeps the result and removes the spill but halves the independent DPAS
chains: 2 groups 5.7% faster at 8K and 0.5% at 32K (3.7% slower at 32K on the 256 x 64 tile), 4 groups 16% slower, so
not used. With `SDPA_OCL_MICRO_MATH=1` the 256 x 64 tile's output is bit-identical to `sdpa_micro`'s on the three
windowless head 512 rows below (8.0 M elements), which rules out an indexing fault of the 32-key tiles there; without
it 301 to 5,384 elements differ by at most 3.05e-5, against 10% of them with 128-key tiles. Error against the CPU
reference is unchanged: max 1.2e-4 / 2.4e-4 (one f16 step). `{300, 457}` 8/2 and a sharp-softmax `{600}` 8/1 row
(`logit_scale_gain` 128) were added; with the alpha rescale removed (128 x 64 tile) only the sharp row fails (0.09),
the near-uniform softmax of the others stays under 2e-3. `PREFETCH_V` makes the 256 x 64 tile 2-13% slower. Per call
against `sdpa_micro` (gemma-4-e2b, two rounds):

| `d_max` 512 tile | 2K | 8K | 32K |
|---|---|---|---|
| 128 x 64, `MICRO_MATH` | 1.000x | 0.940x | 1.012x |
| 128 x 64 | 0.788x | 0.733x | 0.816x |
| 256 x 64, `MICRO_MATH` | 0.897x | 0.790x | 0.841x |
| 256 x 64 | 0.718x | 0.644x | 0.694x |
| 256 x 64, `PREFETCH_V_L3` (below) | 0.702x | 0.635x | 0.622x |

Every one of these tiles lost ground from 8K to 32K. That was an L3-miss knee of `sdpa_ocl`'s own, which `PREFETCH_V_L3`
removes; the B70's power limit also lowers the clock in long prefills, but it slowed both kernels alike: "Paged-attention
PREFILL: the `d_max` 512 long-prompt knee" below.

The daily models, B70 at 2800 MHz (llm_bench `-n 3`, 32 output tokens; kernel = `sdpa_ocl` / `sdpa_micro` PREFILL
device time per prefill from cliloader; 2K / 4K / 8K / 16K / 32K prompts, gemma-2 to 8K and llama-2 to 4K by their
context; before = this branch without these tilings). The MD5 column runs `TEST_USE_SDPA_OCL_DECODE=0`; equality is not
expected with a window, a sink or token types, which `MICRO_MATH` excludes, and qwen3.5 / 3.6 already differ between two
master runs. (b) master also runs a `ref_matmul` this branch does not; against the branch before these tilings it is
0.992 / 0.988 / 0.972 / 0.948 / 0.874. (c) with the 256 x 64 `d_max` 512 tile; with the 128 x 64 one and `MICRO_MATH` the
kernel column was 0.777/0.838/0.873/0.926/0.992 and 0.506/0.565/0.689/0.809/0.932, and the TTFT column, measured again
in the same session, 0.970/0.981/0.977/0.979/0.977 and 0.886/0.904/0.931/0.940/0.960. The gemma-4 TTFT column is with
`PREFETCH_V_L3`, measured again (one round, 2K-32K; e2b two rounds) against the build without it in the same session:
0.969/0.966/0.949/0.926/0.890 and 0.883/0.894/0.913/0.898/0.882 (at 32K 3,092 to 3,034 ms and 4,638 to 4,577 ms); the
kernel column is without it.

| Model | Heads, head | Kernel vs `sdpa_micro` | TTFT vs master | Before, vs master | MD5 = master |
|---|---|---|---|---|---|
| qwen3-8b | 32/8, 128 | 1.008/0.962/0.930/0.921/0.913 | 0.981/0.997/0.983/0.969/0.952 | 1.010/1.069/1.103/1.169/1.239 | 20/20 |
| llama-3.1-8b-instruct | 32/8, 128 | 0.967/0.932/0.904/0.890/0.891 | 0.999/0.986/0.982/0.963/0.947 | 1.021/1.041/1.090/1.153/1.240 | 20/20 |
| minicpm4-8b | 32/2, 128 | 1.010/0.954/0.930/0.914/0.910 | 0.998/0.991/0.985/0.970/0.953 | 1.014/1.040/1.087/1.151/1.213 | 20/20 (32K: empty output) |
| mistral-7b-instruct-v0.2 | 32/8, 128 | 1.003/0.963/0.929/0.921/0.914 | 0.995/0.988/0.980/0.971/0.952 | 1.021/1.056/1.102/1.168/1.237 | 20/20 |
| llama-2-7b-chat-hf | 32/32, 128 | 1.024/0.967 | 0.993/0.986 | 1.061/1.091 | 8/8 |
| qwen3-vl-4b-instruct | 32/8, 128 | 1.019/0.962/0.932/0.919/0.914 | 0.991/0.995/0.975/0.958/0.945 | 1.024/1.089/1.133/1.204/1.276 | 20/20 |
| phi-4-mini-instruct | 24/8, 128, window | 0.959/0.932/0.896/0.876/0.872 | 0.994/0.993/0.978/0.958/0.941 | 1.005/1.009/1.038/1.111/1.174 | not expected |
| phi-4-multimodal-instruct | 24/8, 128, window | 0.966/0.935/0.923/0.891/0.893 | 0.980/0.990/0.988/0.968/0.957 | 0.986/1.024/1.061/1.136/1.207 | not expected |
| phi-3.5-mini-instruct | 32/32, 96, window | 0.849/0.834/0.808/0.796/0.797 | 0.981/0.972/0.938/0.911/0.883 | 0.988/0.982/0.996/1.051/1.097 | not expected |
| phi-3.5-vision-instruct | 32/32, 96, window | 0.935/0.827/0.827/0.831/0.825 | 0.974/0.959/0.947/0.927/0.898 | 0.990/1.030/1.018/1.079/1.187 | not expected |
| llama-3.2-1b-instruct | 32/8, 64 | 0.865/0.854/0.860/0.868/0.871 | 0.975/0.968/0.951/0.928/0.910 | 1.009/1.021/1.025/1.019/1.026 | 20/20 |
| minicpm4-0.5b | 16/2, 64 | 0.882/0.876/0.848/0.857/0.866 | 0.951/0.943/0.919/0.892/0.891 | 0.978/0.999/0.998/0.989/1.012 | 20/20 |
| gpt-oss-20b | 64/8, 64, sink, window | 0.888/0.842/0.795/0.771/0.756 | 0.996/0.985/0.972/0.953/0.920 | 0.999/1.017/1.002/1.003/1.004 | not expected |
| gemma-2-9b-it | 16/8, 256, window | 0.774/0.774/0.776 | 0.971/0.962/0.941 | 0.992/1.014/1.025 | not expected |
| gemma-3-4b-it | 8/4, 256, window, token types | 0.543/0.554/0.555/0.655/0.580 | 0.927/0.966/0.928/0.928/0.904 | 0.946/0.964/0.953/0.965/0.972 | not expected |
| qwen3.5-9b | 16/4, 256 (8 of 32 layers) | 0.831/0.820/0.816/0.807/0.827 | 0.989/0.994/0.991/0.984/0.970 | 0.991/1.004/1.011/1.023/1.054 | master varies |
| qwen3.6-35b-a3b | 16/2, 256 (10 of 40 layers) | 0.822/0.824/0.809/0.809/0.823 | 0.796/0.867/0.913/0.734/0.905 (b) | 0.802/0.878/0.939/0.774/1.035 | master varies |
| gemma-4-e2b-it (c) | 8/1, 256 window + 512 | 0.703/0.699/0.669/0.661/0.697 | 0.953/0.974/0.949/0.918/0.873 | 1.030/1.115/1.252/1.588/2.260 | not expected |
| gemma-4-26b-a4b-it (c) | 16/8, 256 window + 512 | 0.494/0.521/0.600/0.638/0.708 | 0.881/0.893/0.914/0.898/0.870 | 0.927/1.010/1.194/1.521/2.205 | not expected |

Measured and not adopted, at 8K: K-tile prefetch (next tile +0.2%, first tile +0.4%), skipping the
`k_mask` add (0%), reversed workgroup order to start the long query blocks first (0% at 8K, +0.7%
at 32K), a 32 values x 16 scores S*V split as `sdpa_micro`'s (+4.7%), one load per row for the
epilogue sums (+2%, +6.6% at 2K).

### Paged-attention PREFILL: short prompts (causal tile skip)

With the same workgroup as `sdpa_micro` the short prompts still lost: qwen3-8b (`d_max` 128) 24.7 against 12.6 us per call
at 32 tokens and 253 against 214 at 1K. VTune at 1K showed the same XVE activity (63%) and XMX-busy share (58% against 59%)
but 35% more XMX instructions (2.49 G against 1.85 G): the diagonal key tile was computed whole and the triangle past each
query masked afterwards. `CAUSAL_TILE_SKIP` skips that work per subgroup. A KQ subgroup whose keys all lie after its last
query (or at/past `k`) skips its depth loop; its `S_tile` stays 0 and the mask makes it `-INFINITY`. An S*V subgroup skips
the key blocks after its last score row, whose scores are exactly 0. Both conditions are uniform per subgroup and the skipped
work adds exactly 0, so no result changes: the 186 `basic` PA rows are bit-identical, and a skip 16 keys too early fails the
11 PREFILL rows at `d_max` >= 128 (`basic/93`-`96`, `98`-`101`, `145`, `171`, `172`). With image token types
(`BIDIR_MASK`) a group can reach past its query, so only the workgroup's `causal_k` bounds the skip there.

| PREFILL, us per call (32 / 1K / 4K tokens) | `sdpa_micro` | before | `CAUSAL_TILE_SKIP` |
|---|---|---|---|
| qwen3-8b, `d_max` 128 | 12.6 / 214 / 2,372 | 24.7 / 253 / 2,340 | 14.1 / 216 / 2,269 |
| phi-3.5-mini, head 96 (`d_max` 128) | 12.8 / 249 / 2,745 | 24.7 / 264 / 2,389 | 15.8 / 224 / 2,229 |
| gemma-3-4b, `d_max` 256, window / full layers | 19.8 / 165 / 759, 19.0 / 159 / 1,429 | 21.3 / 114 / 514, 20.0 / 117 / 1,084 | 15.0 / 115 / 517, 14.0 / 111 / 1,018 |
| gemma-4-e2b, `d_max` 256 window / 512 (4K) | 684 / 1,947 | 550 / 1,690 | 539 / 1,679 |

On at `d_max` 128 / 256 / 512 (the tilings above). At 64 the branches cost more than the skipped work from 4K on
(llama-3.2-1b 1,283 to 1,371 us, gpt-oss-20b's full layers 2,362 to 2,457), although 32 and 1K gained (13.1 to 11.6, 131 to
129), so it stays off there. In MIXED it was within 0.5% on qwen3.5-9b, llama-3.1-8b and gemma-3-4b at 8K except llama-3.1-8b
with an f16 cache (+4.7%), so it is PREFILL only. `SDPA_OCL_CAUSAL_SKIP=0/1` forces it.

End to end with the default (u4) cache and 256-token chunks, where PREFILL runs on the first chunk only, the 32-token first
token went against master from 0.987x to 0.957x on qwen3-8b (16.30 to 15.93 ms; 36 layers x 10.6 us is 0.38 ms), 1.014x to
0.985x on gemma-3-4b, 0.982x to 0.975x on llama-3.1-8b and 0.997x to 0.994x on phi-3.5-mini (same method as the MIXED
runs, the "before" from the run of the same day); at 1K / 2K, which MIXED dominates, the ratios stayed within 0.01.

### Paged-attention PREFILL: the `d_max` 512 long-prompt knee

The shipped `d_max` 512 tile went from 0.644x `sdpa_micro` at 8K keys to 0.694x at 32K, gemma-4-26b's from 0.730x to
0.743x, and every other `d_max` 512 tile showed the same 8K-best, 32K-worse shape; head 128 models did not. Measured
at a fixed clock, the growth with length had two parts: the clock, which slowed both kernels, and a knee of `sdpa_ocl`'s
own, which alone moved the ratio. Setup: B70, gemma-4-e2b-it (8/1 heads, 7 global layers of head 512 among 35),
cliloader device time per call of the head 512 kernel, prompts cut from one 32K text so that only the length changes
(8,105 / 12,171 / 16,227 / 20,298 / 24,385 / 28,421 / 33,367 tokens), two rounds; the clock and the throttle reasons
were sampled every 5 ms during each run (`tile0/gt0/freq0/act_freq`, `throttle/status`, `throttle/reasons`).

**The clock.** `min_freq` = `max_freq` = 2800 reads back, but under a long prefill the firmware lowers the GT clock with
throttle reason `pl2` (the PL2 power limit; `power1_cap` 230 W, package temperature at most 64 C, so not thermal). In a
32K run the clock sat at 2133-2167 MHz for most of the prefill (`pl2` in 68% of the busy samples with `sdpa_ocl`, 78%
with `sdpa_micro`), in an 8K run rarely (4%). Pinned at 1800 MHz no sample showed `pl2` and the clock never moved. Per
call time / keys^2 (ps):

| Kernel, clock | 8.1K | 12.2K | 16.2K | 20.3K | 24.4K | 28.4K | 33.4K |
|---|---|---|---|---|---|---|---|
| `sdpa_micro`, 2800 pin | 137.5 | 134.0 | 135.0 | 138.7 | 142.0 | 144.5 | 145.0 |
| `sdpa_ocl`, 2800 pin | 87.0 | 87.1 | 86.7 | 91.6 | 95.8 | 98.8 | 100.4 |
| `sdpa_micro`, 1800 | 175.7 | 173.8 | 172.0 | 172.0 | 171.7 | 172.0 | 170.8 |
| `sdpa_ocl`, 1800 | 110.8 | 109.0 | 108.0 | 111.9 | 114.5 | 117.2 | 118.4 |

`sdpa_micro`'s rise at 2800 MHz (+5-8%) was all clock. `sdpa_ocl` rose +15% there and kept +9.6% at a fixed clock (flat
to 16K), so the ratio went 0.628x to 0.693x at 1800 MHz as it had gone 0.633x to 0.693x at the pin. The model's two head
512 kernel hashes (4 and 3 of the 7 global layers) also differed by 6-10% at 24K / 32K at the 2800 pin and by 0.2% at
1800 MHz: layers that run later in a prefill ran deeper into PL2.

**The knee.** 16K to 20K keys is where one KV head's K and V (2 KiB per token at head 512) outgrow the 24 MiB L3
(`clinfo` global memory cache; line 256 B). VTune GPU Hotspots, overview, 1800 MHz, the head 512 kernel:

| | `sdpa_ocl` 8K | `sdpa_micro` 8K | `sdpa_ocl` 32K | `sdpa_micro` 32K |
|---|---|---|---|---|
| XMX active | 64% | 41% | 59% | 41% |
| L3 read | 1,231 GB/s | 2,174 GB/s | 1,116 GB/s | 2,343 GB/s |
| L3 read per call | 9.0 GB | 25.1 GB | 147 GB | 446 GB |
| L3 miss ratio | 3.0% | 1.1% | 18.2% | 4.6% |
| DRAM read | 36 GB/s | 23 GB/s | 64 GB/s | 187 GB/s |
| Stall on Send | 20% | 0.4% | 30% | 0.8% |

`sdpa_ocl` reads each K/V tile from L3 once per workgroup: 9.0 GB per call at 8K against 9.1 GB computed (1,016
workgroups x about 4,180 causal keys x 2 KiB), so the 4 query subgroups that read one K slab and the 4 that read one V
slab meet in L1. `sdpa_micro` reads 2.8x more from L3 and does not stall on it. DRAM read stayed far under the 608 GB/s
specified peak, so the knee is miss latency, not bandwidth. Why `sdpa_micro` hides its misses is not profiled; it is 1.6x slower
per tile, which leaves more time per miss (ASSUMED).

**Hypotheses tested**, per call of the head 512 kernel against the shipped kernel at 1800 MHz (two rounds, within 0.3%
unless noted). "Address prefetch" is `__builtin_IB_lsc_prefetch_global_*` with one address per lane, as
`sdpa_micro`'s `cooperative_prefetch_2d`; "L3 only" is cache control `L1UC_L3C`, "L1 and L3" `L1C_L3C`:

| Change | 8K | 32K | Reading |
|---|---|---|---|
| Workgroup order with the 8 heads of a KV group innermost, so concurrent workgroups read the same K/V tiles | +5.2% | +4.1% | knee unchanged (92.5 to 104.3 ps): not L3 capacity or reuse across workgroups |
| `PREFETCH_V` (block prefetch of this tile's V, 64 B wide) | +14.2% | +2.6% | time / keys^2 flat from 16K (121 ps): the knee is V misses; the prefetch costs |
| Address prefetch of this tile's V, 64 B stride, L3 only | +17.0% | +4.5% | |
| the same, L1 and L3 | +15.0% | +4.4% | not L1 pollution |
| Address prefetch of the next tile's K, 64 B stride | +15.7% | +10.7% | |
| 32-row x 2-block K / V / both reads, the same lines in a quarter of the messages (one run, 8K) | +3.8% / +14% / +9% | | not the message count |
| Address prefetch of this tile's V, 128 B stride | +4.4% | -5.8% | |
| **the same, 256 B stride (one address per line)** | **-0.5%** | **-11.0%** | **adopted: `PREFETCH_V_L3`** |
| the same, 256 B, L1 and L3 | +1.2% | -8.9% | filling L1 does cost |
| the same, 512 B stride (every other line) | -0.1% | -3.5% | half the lines, a third of the gain |
| the next tile's V, 256 B | +0.2% | -10.1% | |
| this tile's V and the next tile's K, 256 B | +4.0% | -8.6% | K misses are not worth their requests |
| the next tile's K and V, 256 B | +3.9% | -6.8% | |

So the cost of a prefetch follows the number of line requests it adds to L3, not the message count and not where the
lines land: the 64 B wide block prefetch asks for each 256 B line of a 1 KiB V row four times, which is also why
`PREFETCH_V` made this tile 2-14% slower. One request per line costs nothing measurable at 8K.

**`PREFETCH_V_L3`** (`v_tile_prefetch_l3()`, `sdpa_ocl_v_load.cl`). Where `PREFETCH_V` would run, after the S_max
atomics and inside the split barrier, the workgroup prefetches this key tile's V rows into L3 only: one
`__builtin_IB_lsc_prefetch_global_uchar(..., LSC_LDCC_L1UC_L3C)` per 256 B line, the lines of `kq_wg_tile_keys` rows dealt
over all lanes (1,024 lines over 512 lanes at head 512), rows clamped to the last key so that no address leaves the
tensor. The byte form needs no alignment. `v_prefetch_l3` is set at `d_max` 512 only (`solve_pa_prefill_tiling()`);
`SDPA_OCL_V_PREFETCH_L3=0/1` forces it on any PA PREFILL and replaces `PREFETCH_V` when on; MIXED has no V input surface
and `#error`s. The builtin is declared here with its cache control as a `uint`: `tile_ops.cl` declares it with
`enum LSC_LDCC`, which is compatible, but an enum cannot be defined in this file, because the codegen `#undef`s every
macro of the file after each kernel (an include guard does not survive) and a batch can hold several `sdpa_ocl` kernels
in one program. Both prototypes in one translation unit compile (ocloc, bmg).

Per call time / keys^2 (ps) at 1800 MHz:

| | 2K | 4K | 8K | 12K | 16K | 20K | 24K | 28K | 32K |
|---|---|---|---|---|---|---|---|---|---|
| 256 x 64 | 144.6 | 119.9 | 110.8 | 109.0 | 108.0 | 112.0 | 114.7 | 117.3 | 118.5 |
| + `PREFETCH_V_L3` | 142.2 | 119.0 | 110.3 | 107.4 | 105.4 | 105.6 | 105.6 | 106.1 | 105.2 |

0.61-0.63x `sdpa_micro` from 8K to 32K at 1800 MHz, where it was 0.63x to 0.69x. At the 2800 pin, PL2 included, per call
against `sdpa_micro` at 2K / 8K / 32K: 0.702x / 0.635x / 0.622x (0.712x / 0.638x / 0.694x without; 32K 111.8 to 100.1
ms), and at 4K 1,584 against 1,604 us. First-token latency against the build without it, same session, llm_bench
`-n 3 -ic 32`: gemma-4-e2b 0.984 / 1.008 / 0.999 / 0.991 / 0.981 and, a second round, 1.005 / 1.008 / 0.993 / 0.999 /
0.985 at 2K / 4K / 8K / 16K / 32K; gemma-4-26b 0.998 / 0.999 / 1.001 / 1.000 / 0.987. The 4K +0.8% followed the arm
order (the new build ran second each time): run first, it was +0.4% and 0.0%, and the head 512 kernel itself is 1.2%
faster at 4K with the other PA layers unchanged.

A prefetch changes no value: the 173 `basic` PA rows give bit-identical outputs with `SDPA_OCL_V_PREFETCH_L3` forced on
and off at every head size (a temporary output dump in the test, compared bit for bit; the program sources showed the
switch taking effect). The PA suite and the `*sdpa*:*rope*:*kv_cache*:*update_shape_test.paged*` suite fail the same
13 and 60 tests as before.

In place of `PREFETCH_V` at `d_max` <= 256, where one KV head's K/V still fits L3 at 32K (head 128: 16 MiB), it was
slower except on gemma-3's full-attention head 256 layers (1800 MHz, two rounds):

| Model, layers | 8K | 32K |
|---|---|---|
| qwen3-8b, head 128 | +3.8% | +3.9% |
| llama-3.2-1b-instruct, head 64 | +5.1% | +5.2% |
| gemma-3-4b-it, head 256 window layers | +2.8% | +2.9% |
| gemma-3-4b-it, head 256 full layers | -0.8% | -1.0% |
| gemma-4-e2b-it, head 256 window layers | +2.6% | +1.1% |

Not covered: other Xe2 devices (B580, Lunar Lake) were not measured. The 256 B step is the B70 L3 line; on a device
with 64 B lines it would request one line in four, which costs speed, not correctness.

What carries over: check the clock (`act_freq`, `throttle/reasons`) before reading a ratio across problem sizes, and
pin below the power limit for scaling questions; normalise the time by the expected growth (keys^2 here) to find a
knee, then look for a working set that crosses a cache size there; count a prefetch's line requests, not its messages,
and issue one request per line; test reuse hypotheses with a change that keeps every value (workgroup order, prefetch),
so that a bit-identical output check covers correctness.

### Paged-attention PREFILL: wide reads (IGC 2.41)

Every number above was measured with IGC 2.36.3 (compute runtime 26.22). With IGC 2.41.5 (26.35) `sdpa_micro`'s head 512
PA prefill kernel lost its 2,752 B spill and became about 40% faster (gemma-4-e2b, 32K keys, 1800 MHz: about 190 to 114
ms per call), while `sdpa_ocl`'s became about 7% slower (117 to 125 ms), so the shipped tile went from 0.62x to 1.08-1.10x
`sdpa_micro` (gemma-4-26b 1.12-1.24x) and the gemma-4 first-token lead vanished. The head 64 / 128 micro kernels had no
spill and did not move. VTune at 8K, head 512 kernel: XMX active 60.9% against `sdpa_micro`'s 64.4%, stalled on Send
26.5% against 5.1%. Per subgroup and key tile the kernel issued 64 K block reads and 64 V block reads of 512 B and 96 SLM
reads of 256 B; `sdpa_micro`'s final binary (iga64) reads K in 2 KB and V in 1 KB blocks and SLM in 512 B messages.

**`WIDE_READS`** (bits 1 / 2 / 4; `wide_reads` in `solve_pa_prefill_tiling()`: 7 at `d_max` 512, the other tilings
in "Wide reads at `d_max` 64 / 128 / 256" below):

- K: one `16b_32r16x2c` read per depth-tile pair (`16b_16r16x2c` at 16-key subgroup tiles), block-major, so `k_raw`
  for `db` and `db + 1` are the two halves of the same registers the narrow reads filled.
- V: one `transform_16b_16r16x2c` read per value-block pair (`vb[cd]`, `vb[cd + 1]`). There is no 16-bit `16r16x4c`
  transform (ocloc, bmg).
- Q_slm: read as `ulong4` (one 512 B SLM message instead of two 256 B ones), so it is staged in that layout, with
  per-lane stores. `intel_sub_group_block_write_ul4` stored it wrong on the dynamically padded head 64 PA rows
  (`basic/29`-`34`) with `SDPA_OCL_WIDE_READS=7`, also with `Q_slm` 64 B aligned; the per-lane stores write the same
  layout and pass.

Per call at 1800 MHz (cliloader, two rounds, spread under 0.1%), head 512:

| | e2b 8K | e2b 32K | 26b 8K | 26b 32K |
|---|---|---|---|---|
| `sdpa_micro` | 7,537 us | 114.2 ms | 16,346 us | 258.3 ms |
| narrow reads | 1.077x | 1.095x | 1.162x | 1.121x |
| K (1) | 0.997x | 0.993x | 0.962x | 0.927x |
| K + V (3) | 0.896x | 0.897x | | |
| K + V + Q_slm (7) | 0.854x | 0.852x | 0.804x | 0.773x |

At 32K that is about 79% (e2b) and 77% (26b) of the B70's XMX f16 peak (256 engines x 256 FLOP per clock), against 68%
and 60% for `sdpa_micro`. Measured and not used: 32-row V reads (`transform_16b_32r16x1c`, `_32r16x2c`, a cp-block pair
per read) 6-10% slower alone and 3-10% slower than the 16-row pair with K + Q_slm; a block prefetch of the K pair one or two
pairs ahead +51%; a block prefetch of the next cp block's V +2-3%.

The reads change no value: the 173 `basic` PA rows are bit-identical with `SDPA_OCL_WIDE_READS` 0, the default and 7
(every head size); with the V pair read deliberately swapped the four head 512 rows fail and the head 256 row passes. The
PA, `*sdpa*:*rope*:*kv_cache*:*update_shape_test.paged*` and `*large_head*:*kv_cache_rotation_content*:*shared_kv_cache*`
suites fail the same 13 / 60 / 0 tests with it on and off.

First-token latency against master (`fde316cf48b`, same IGC), llm_bench `-n 3 -ic 32`, 1800 MHz, the daily text prompts
at 2K / 8K / 16K / 32K: gemma-4-e2b 0.979 / 0.981 / 0.981 / 0.964 (narrow reads 0.988 / 1.013 / 1.011 / 1.023),
gemma-4-26b 0.988 / 0.969 / 0.941 / 0.932 (0.997 / 0.994 / 0.987 / 1.015). At 278 and 1,024 tokens with the daily image
the head 512 call is 0.1-0.4 ms and `sdpa_ocl` is 1.7-2.3x / 0.91-0.93x `sdpa_micro` there; 26b's lead at those lengths
(0.76x / 0.87x) is the image mask (`sdpa_micro` checks group membership per (query, key) element, see "Bidirectional
image groups"). Not covered: other Xe2 devices.

#### Wide reads at `d_max` 64 / 128 / 256

Defaults: `d_max` 64 takes K + Q_slm (5), or nothing (0) when the layer has a sliding window
(`add_tensor_block_io_jit()`); `d_max` 128 and 256 take the K pair alone (1). The K read there is `16b_16r16x2c` (16-key
subgroup tiles, 1 KB per depth-tile pair). The V pair exists only at `d_max` 256 (`sv_value_blocks` 2); `d_max` 64 / 128
split the values 16 per subgroup, so it is inactive there.

What each part costs (IGC ShaderDump, one key tile of the main loop): the K pair halves the K block reads (8 to 4 at
`d_max` 128, 4 to 2 at 64) and changes nothing else. The Q_slm read halves the Q SLM messages (64 `d32x64t` to 32
`d64x64t` at 128), but the `ulong4` result has to become the `int8` DPAS operand, 8 `mov (16) r<1>:d r<2;1,0>:d` per
read: 256 more movs per key tile at 128, 128 at 64. So Q_slm pays only where the loop waits on memory, and its gain grows
with the key count (llama-3.2-1b, Q_slm alone: +2.4% at 1K keys, -2.9% at 32K); it loses where K and V stay in cache:
short prompts and small sliding windows. The K pair alone and K + Q_slm do not add up, and the K pair's sign depends on
the kernel variant (gpt-oss-20b, with a sink input, is 1.2-1.8% slower with it at `d_max` 64).

Per call against the narrow reads at 1800 MHz (cliloader `-d -dv`, `-n 3`, two rounds; spread up to 0.25% from 2K up,
0.9% at 1K and 1.5% at 32 tokens, where a call is 12-28 us):

| | default | 32 | 1K | 2K | 4K | 8K | 32K |
|---|---|---|---|---|---|---|---|
| llama-3.2-1b (64) | 5 | +2.4% | -0.4% | -1.4% | -1.8% | -3.1% | -5.6% |
| minicpm4-0.5b (64) | 5 | | -1.5% | | | -3.0% | |
| gpt-oss-20b (64), full layers | 5 | | +1.0% | | -2.8% | -3.3% | -4.7% |
| gpt-oss-20b (64), 128-key window layers | 0 | | (5: +11.1%) | | (+11.9%) | (+11.2%) | (+9.9%) |
| qwen3-8b (128) | 1 | +0.2% | +0.1% | -0.3% | -0.5% | -0.8% | -1.0% |
| phi-3.5-mini (96, `d_max` 128) | 1 | +0.6% | +0.1% | -0.1% | -0.2% | -0.2% | -0.2% |
| llama-3.1-8b / minicpm4-8b / phi-4-mini (128) | 1 | | 0 to +0.1% | | | -0.25 / -0.7 / -0.4% | |
| gemma-3-4b (256), full / 1024-key window | 1 | +0.9 / -0.2% | -2.0 / -2.0% | -2.1 / -2.2% | -1.9 / -2.3% | -2.0 / -2.1% | -2.1 / -2.4% |
| gemma-2-9b (256) | 1 | | -3.7% | | | -5.2% | |
| qwen3.5-9b (256, image in the prompt) | 1 | | -3.1% | | | -4.4% | |
| gemma-4-e2b (256, 512-key window) | 1 | | -1.9% (image) | | | -1.0% | |

Measured and not used: Q_slm at `d_max` 128 (alone +1.4-4.0%, with K +3.8-16%) and 256 (alone +0.1-2.2%); the V pair at
256 (+0.6% full, -0.1% window); K + Q_slm on gpt-oss-20b's window layers (+10-12%) and the K pair alone there (0 to
+0.7%). gemma-3-4b with the daily image (275 and 1,025 tokens, bidirectional image groups, full / window layers) is
-0.3 / +0.1% and -1.6 / -1.9%; its vision encoder (plain SDPA) does not change. The cache precision (f16, i8, u4) does
not change the PREFILL kernel or the gain (llama-3.2-1b 8K -3.0 / -3.1 / -3.1%). With GenAI's continuous-batching
default scheduler (256-token chunks) only the first chunk runs PREFILL; the other chunks run MIXED, which these reads do
not touch.

The reads change no value: the 173 `basic` PA rows are bit-identical with `SDPA_OCL_WIDE_READS` 0, the default, 1, 5
and 7. With the two halves of the K pair deliberately swapped, the default changes exactly the rows that read K in
pairs: 32 PREFILL rows at `d_max` 64 without a window, 128 (head 96 included), 256 and 512, and none of the `d_max` 64
rows with a window; `SDPA_OCL_WIDE_READS=1` also changes those (`basic/87`-`90`, `97`). Only 16 of the 32 rows fail the
test's tolerance (the compressed-cache rows stay inside it), so a bitwise comparison is the check. The PA,
`*sdpa*:*rope*:*kv_cache*:*update_shape_test.paged*` and `*large_head*:*kv_cache_rotation_content*:*shared_kv_cache*`
suites fail the same 13 / 60 / 0 tests as before, with the reads on and off.

First-token latency against the build before (532e4893fa), llm_bench `-n 3 -ic 32`, 1800 MHz, two rounds interleaved,
median of 6: llama-3.2-1b 1.006 / 0.999 / 0.989 / 0.961 at 1K / 2K / 8K / 32K (1K within the run-to-run range,
18.46-19.05 ms before), gpt-oss-20b 1.001 / 0.997 / 0.986 at 1K / 8K / 32K, qwen3-8b 1.001 / 0.998 / 0.994, gemma-3-4b
1.001 / 0.999 / 0.996, gemma-2-9b 0.998 / 0.998 / 0.989 at 1K / 2K / 8K, qwen3.5-9b 1.000 / 0.999 at 1K / 8K. Not
covered: other Xe2 devices, and `d_max` 128 / 256 kernels with a sink input (no such model was at hand).

### Plain-SDPA PREFILL tiling (`d_max` 128)

Plain (non-PA) multi-token SDPA with f16 Q / K / V at `d_max` 128 on Xe2+ (`solve_plain_prefill_tiling()`): a static
stage with at least 128 queries and 2K keys takes the PA PREFILL `d_max` 128 workgroup (16 x 64 subgroup tiles, 16 x 2
subgroups = 256 keys x 128 queries, 256 GRF, `sdpa_micro`'s `xehpc_h128`) with `WIDE_READS` 1, `PREFETCH_V`,
`SOFTMAX_BRANCHLESS` and `MAX_SPLIT_BARRIER`. Every other f16 `d_max` 128 multi-token stage, shape-agnostic ones
included, keeps the 128 x 32 table tile with those same loop changes. An `is_causal` stage of either tile also takes
`CAUSAL_TILE_SKIP`, and a non-causal table-tile stage without a mask its KQ branch ("Plain-SDPA causal tile skip" below). `CAUSAL_MASK_LAST_ONLY` stays PA only: it rejects lower-right
causal and an attention mask by `#error`, which is how the pair failed to build for the plain kernel on PTL-H
(`CL_BUILD_PROGRAM_FAILURE`; ocloc for ptl-h gives the same -11 with `CAUSAL_MASK_LAST_ONLY` 1 alone and builds the
skip alone). Plain SDPA never emits `MICRO_MATH`. bf16 and compressed K/V keep the bare table: neither
was measured with these paths.

The loop changes do not change a result: with the table tile, all 42 bench cases (static, dynamic K, dynamic Q + K)
gave bit-identical outputs with and without them. The 256 x 128 tile accumulates in another order (outputs differ; max
|out - `sdpa_opt`| 1.1e-3, the same as the table tile's).

**One tiling per compiled stage.** The dispatch re-derives the tiling from the runtime shape
(`get_dispatch_data_func()` calls `make_problem()`), and the kernel has `reqd_work_group_size`. So a stage compiled
for dynamic shapes must not choose by query / key count: `make_problem()` takes `shape_agnostic` (compile time
`params.is_dynamic()`, dispatch `compiled_shape_agnostic(kd)`, the stage's SHAPE_INFO argument, which is saved with a
blob) and leaves the counts unread for such a stage. The first version of this policy read them anyway. With a dynamic
Q + K (a stateful LLM's SDPA) and 512 queries over 4K keys, or a static Q with a dynamic K and 1K runtime keys,
compile time and dispatch disagreed, and every call failed with `CL_INVALID_WORK_GROUP_SIZE` (B70; the dynamic
`sdpa_gpu_test_128_8_512_2048_1` rows and `sdpa_gpu_causal_mask.prefill_32q_8kv_512q_2048kv` fail that way).

Measured on B70 (GPU.1, clock not pinned; one cliloader process per case, 100 warm-ups + 400 calls, min device time,
sides interleaved per case, 3 rounds, median), lower-right causal, f16:

| 16 chunks of 512 queries over 512..8192 keys (an 8K prompt), sum | before | this tiling | `sdpa_micro` (master) |
|---|---|---|---|
| 32 / 32 heads | 8.357 ms | 5.881 ms | 5.903 ms |
| 28 / 4 heads | 5.839 ms | 5.330 ms | 5.358 ms |

Per chunk from 2K keys this is 0.68-0.76x the bare table tile (32 / 32 heads; 28 / 4: 0.89-0.97x) and 0.81-0.94x the
table tile with the loop changes; below 2K the loop changes alone give 0.90-0.97x (32 / 32) and 0.98-1.02x (28 / 4).
The table tile stays faster than the large one at 1K keys (1.02-1.06x) and 512 (1.29x), and at fewer than 128 queries
(64 queries over 4K keys: 1.15-1.32x), hence the threshold; at 1.5K the large tile would be 0.94-0.99x. The PTL-H B390
measurement that introduced this policy (PR-39316 shapes) gave 0.994 / 0.995 of `sdpa_micro` with PR-39316, from 0.52
/ 0.74. Remaining against `sdpa_micro` on B70 at that point: short static chunks (1.00-1.15x at 512-1.5K keys; 0.98x
at 8K), static self-attention prompts (2K: 1.11x, 4K: 1.05x; `sdpa_micro` skips the causal tiles), and decode
(`sdpa_ocl_mixed`, unchanged: 1.07-1.15x). The causal tile skip below closed the first two, the decode tiling after it
the third.

A shape-agnostic stage keeps the table tile because no single tile wins across shapes. With dynamic Q + K, the 256 x
128 tile is 0.75-0.95x the table at 128+ queries over 2K+ keys and at 4K-8K-token prompts, but 1.16-1.47x at 16-64
queries over 4K keys (a later chat turn, speculative decoding) and 1.13-1.29x at 512-1K-token prompts. The table's
loop changes there: 32 / 32 heads 0.61-0.93x the bare table from 4K keys or 1K-token prompts; 28 / 4 heads 0.90-0.96x,
except 1.02-1.07x for 512 queries over 2K-8K keys and 16 queries over 4K. Kernels under ~30 us vary 20-30% between
processes and were left out of these ranges. Taking the large tile for long dynamic shapes needs a second
shape-agnostic stage chosen in `execute()` ("Performance opportunities").

`SDPA_OCL_PLAIN_MIRROR=0/1` forces the table / the 256 x 128 tile on every such stage, static and shape-agnostic (the
f16 gate stays), `SDPA_OCL_PLAIN_TABLE_FLAGS=0` the bare table (both need a `SDPA_OCL_NO_GETENV=OFF` build). Bench:
`test/xe3_test/b70_repro` (kit README), `sdpa_micro_prefill_config_bench` there is not part of the tree.

### Plain-SDPA PREFILL tiling (`d_max` 64)

Plain (non-PA) multi-token SDPA with f16 Q / K / V at `d_max` 64 (`k` and `v` head 33-64) on Xe2
(`solve_plain_prefill_d64_tiling()`). Without an attention mask input: 32-key x 32-query subgroup tiles, 4 x 2 subgroups
(128 keys x 64 queries), 256 GRF, with `PREFETCH_V`, `WIDE_READS` 1 (the K pair), `SOFTMAX_BRANCHLESS` and
`MAX_SPLIT_BARRIER`, and on a non-causal stage the KQ branch (`CAUSAL_TILE_SKIP`, "Plain-SDPA causal tile skip"); a causal
stage does not take the causal tile skip (below), as it does not on PA `d_max` 64. With a mask input (tensor or
single-element; not a const `attn_mask_val`): the table tile (128 GRF) with those four loop changes. The choice reads the
op's mask input, never a query or key count, so static and shape-agnostic stages, at compile time and dispatch, agree.
Before, these stages ran the bare table tile (16 x 32 subgroup tiles, 8 x 2 subgroups, 128 GRF, none of the loop
changes), which was 1.29x `sdpa_micro` on stable-diffusion-3.5-large-turbo's joint attention (38 heads, 1024 image + 333
text tokens, no mask, a dynamic model, so the shape-agnostic stage: 602.5 vs 469.0 us per call in the model, cliloader,
1800 MHz) and made its transformer step 4.5% slower than master. bf16, compressed K / V, a `v` head of another `d_max`,
and other devices keep the bare table (not measured). `SDPA_OCL_PLAIN_D64=0` keeps the bare table (A/B; it reproduces
the old output bits).

Candidates were swept on B70 (GPU.1 pinned at 1800 MHz; `SDPA_OCL_KQ_*` / loop-flag env overrides of one build; one
cliloader process per case and side, 100 warm-ups + 400 calls, min device time, sides interleaved per case, 3 rounds,
median, round-to-round spread <= 3.6%; inputs at 1/1024 resolution) over 17 head-64 shapes: non-causal (SD3.5, CLIP 577,
Whisper 1500, ViT-B 197, 64 / 333 / 4K tokens, head 40 SD1.5 self- and cross-attention, a 2D mask, static and dynamic)
and causal GQA (128-token and 2K prompts, a 512-query chunk over 4K keys, 16 queries over 1K, 64 / 8 heads at 4K).
Against the table: the PA PREFILL `d_max` 64 workgroup (16 x 64 subgroup tiles, 128 x 128, 256 GRF) was 0.78x on SD3.5
(this tiling 0.65x); 16 x 64 tiles in a 64 x 128 workgroup were best at head 40 (0.61x) but 1.12-1.65x at 64-333 tokens,
with a mask and on causal prompts / steps; 16 x 32 tiles in a 64 x 64 workgroup at 128 GRF were 0.71x on SD3.5 but
1.02-1.21x at 197 tokens, a 128-token causal prompt, with a mask and on 16-query steps; 32 x 16 tiles were 1.12-1.15x at
head 40 and on 16-query steps. This tiling was faster than the table on every shape. At 128 GRF it was 1.4-2.7x slower.
Without the K pair it was 0.94-1.15x with the KQ branch (faster at head 40 only) and 1.00-1.06x on causal stages. The KQ
branch made the non-causal shapes 0.88-0.99x; the causal skip made a 512-query chunk 1.06x and a 16-query step 1.11x (a
2K prompt 0.98x), so it stays off.

A mask splits the rule. The kernel streams a 2D mask, and an SDPA-backend LLM prompt is causal through such a mask
(`IS_CAUSAL` 0, `WITH_ATTN_MASK` 1: llama-3.2-1b, 32 / 8 heads, a 4K prompt takes 7.9 ms per layer, against 1.2-1.5 ms
with `is_causal`). There every other tile measured was 1.06-1.21x the table (the 32 x 32 one 1.09x, the table itself at
256 GRF 1.18x; why the mask path prefers it was not analysed), and the 32 x 32 tile made that model's 4K TTFT 5.1% slower. The table with the loop changes
was 0.69-1.00x the table on every masked shape (1K-4K tokens, 12-32 heads, GQA, static and dynamic), so masked stages
take it.

| us per call (median of 3) | table (before) | this rule | `sdpa_micro` (master) |
|---|---|---|---|
| SD3.5 38 heads, q = k 1357, dynamic / static | 476.7 / 431.4 | 309.1 / 300.8 | 450.7 / 361.7 |
| Whisper encoder 20 heads, 1500, dynamic / static | 257.7 / 255.0 | 189.7 / 184.2 | 282.8 / 224.0 |
| CLIP-L/336 16 heads, 577 | 59.5 | 40.6 | 59.5 |
| 16 heads, 4K / 333 (38 heads) / 197 (12 heads) / 64 | 1,847 / 45.9 / 13.3 / 7.5 | 959.6 / 32.3 / 11.0 / 6.2 | 1,511 / 40.0 / 10.0 / 4.3 |
| head 40, 8 heads: self 4K / cross 4K x 77 | 835.2 / 66.1 | 618.9 / 48.2 | 1,038 / 39.4 |
| causal 32 / 8 heads: 128 / 2K / 4K prompt | 9.4 / 407.2 / 1,532 | 7.8 / 345.0 / 1,184 | 7.1 / 440.1 / 1,667 |
| causal 32 / 8 heads: 512 over 4K / 16 over 1K | 345.0 / 31.4 | 252.7 / 25.1 | 377.9 / 27.3 |
| causal 64 / 8 heads, 4K prompt | 3,004 | 2,349 | 3,273 |
| 2D mask, 32 / 8 heads: 1K / 2K / 4K | 281.0 / 1,243 / 7,921 | 251.8 / 916.6 / 7,920 | - |
| 2D mask: 16 heads 1K; 12 heads 4K; 16 heads 577 static | 147.7 / 2,049 / 126.2 | 131.1 / 1,422 / 126.4 | - |

All dynamic Q + K unless noted; the masked rows from a second sweep (same protocol); `sdpa_micro` has no masked numbers
(the master bench takes no mask). Against the table it is 0.52-1.00x; against `sdpa_micro` 0.60-0.83x from ~300 tokens
without a mask; the short and the cross-attention shapes stay slower than `sdpa_micro` (6.2 vs 4.3 us at 64 tokens), as
the table was. The output checksum of every case equals that of the swept configuration, and with `SDPA_OCL_PLAIN_D64=0`
that of the build before; the `d_max` 128 / 256 stages measured alongside are bit-identical to before. Max |out -
`sdpa_opt`| is the same as the table's in every case.

End to end (llm_bench on GPU.1 pinned at 1800 MHz, the daily commands; the same build with `SDPA_OCL_PLAIN_D64=0` /
default against the master install with `sdpa_micro`; sides interleaved per model, 2 rounds, median; ms, image models
per denoising step, LLMs and VLMs first-token latency):

| Model | master | before | this rule |
|---|---|---|---|
| stable-diffusion-3.5-large-turbo, 512 x 512, transformer step | 114.25 | 119.37 | 108.99 |
| stable-diffusion-v1-5, UNet step (head 40) | 47.89 | 46.14 | 43.62 |
| phi-3.5-vision, image + 802 / 1032 tokens (CLIP encoder) | 112.14 / 133.81 | 111.93 / 132.42 | 109.22 / 131.04 |
| llama-3.2-1b, SDPA backend, 1024 / 4093 tokens | 21.27 / 267.62 | 21.27 / 237.69 | 20.74 / 238.19 |
| gpt-oss-20b, SDPA backend, 1024 tokens | 143.11 | 143.02 | 140.96 |
| minicpm4-0.5b, SDPA backend, 1024 tokens | 15.80 | 16.35 | 15.82 |

The SD3.5 step repeated within 0.1% over 3 more rounds (114.2 / 119.3 / 108.8); in the model its SDPA takes 326.3 us
per call (cliloader), from 602.5. gemma-4-e2b (image, a masked vision encoder) and the 32-token prompts moved within
their run-to-run spread (up to 12%); their kernels did not get slower (minicpm4 at 32 tokens 11.3 -> 11.0 us). The 32 x
32 tile without the mask split had made llama-3.2-1b's 4K prompt 5.1% slower (249.66 ms).
Generated images differ from before, as from `sdpa_micro`: 20 denoising steps amplify any rounding change (PSNR 17-20 dB
between any two of the three), so accuracy was checked on one transformer call with identical inputs against the CPU
plugin in f32: relative L2 error 3.93e-2 with this rule, 3.88e-2 before and 3.91e-2 with `sdpa_micro` (t = 500; the
three f16 GPU outputs differ from each other by 1.9e-2; the GPU in f32 was 2.4e-2 from the CPU on a second input set). At
t = 1000 the three f16 outputs are bit-identical to each other and far from f32 (relative 1.0), on master too: an f16
issue outside SDPA, not investigated.

Tests: `sdpa_gpu_noncausal_tail` `*_d64` / `*head40*` (SD3.5's 1357 keys end 13 keys into a subgroup; CLIP 577; GQA
200 / 1100; head 40; a 2D mask) against the host reference, and `sdpa_gpu_causal_mask.*_d64` (static ones element-wise
against the explicit-mask reference, which now takes the masked table tile: max |diff| 6e-5-4.9e-4, bound 2e-3; a 16-query
step and a 512-query chunk) with Q in [-4, 4]. With the KQ branch's
subgroup bound 16 keys early, exactly the four non-causal rows without a mask fail (as do the `d_max` 128 rows that use the
branch or the causal skip), and the masked and causal `d_max` 64 rows, which take neither, pass.

### Plain-SDPA PREFILL tiling (`d_max` 512)

Plain (non-PA) multi-token SDPA with f16 Q / K / V at `d_max` 512 (`k` and `v` head 257-512) on Xe2
(`solve_plain_prefill_d512_tiling()`): 256 GRF with the PA PREFILL `d_max` 512 loop changes (`PREFETCH_V_L3`,
`WIDE_READS` 7, `SOFTMAX_BRANCHLESS`, `MAX_SPLIT_BARRIER`) and no causal tile skip (`causal_skip_off`). Without an
attention mask input it keeps the table tile (16 x 16 subgroup tiles, 8 x 2 subgroups, 128 keys x 32 queries); with
one (tensor or single-element) it takes the PA PREFILL `d_max` 512 workgroup (32 x 16 subgroup tiles, 8 x 4 subgroups,
256 keys x 64 queries). As at `d_max` 64 the choice reads the op's mask input only, never a count. Before, these stages
ran the table tile at 128 GRF, which spilled 1280 B (2432 B with a mask): a VAE decoder's attention (one head of 512,
4096 tokens at 512 x 512, non-causal, a dynamic model; stable-diffusion-3.5, flux and SD1.5 have it) took 865-874 us
against `sdpa_micro`'s 587-606 (in the model, cliloader, 1800 MHz). `SDPA_OCL_PLAIN_D512=0` keeps the bare table (A/B;
it reproduces the old output bits). bf16, compressed K / V, another `v` head and other devices keep the bare table.

Swept on B70 as the `d_max` 64 tiling (same protocol, 3 rounds, spread <= 1.2% on the kept sides) over VAE shapes (1K /
4K / 16K tokens, static and dynamic), an SDPA-backend LLM prompt with a 2D mask (8 / 1 heads at 1K and 4K, 16 / 2 at
1K) and causal GQA (a 16-query step over 1K keys, a 2K prompt). Against the table at 256 GRF with these loop changes the
PA workgroup was 1.03-1.50x without a mask (0.99x at 16K) but 0.77-0.89x with one; dropping the L3 prefetch made the
16K VAE 1.16x and the masked 4K prompt 1.07x; the KQ branch moved non-causal shapes -0.7% to +1.8% and the causal skip
made the 16-query step 1.15x (a 2K prompt 0.99x), so neither is taken. 32 x 16 tiles in a 256 x 32 workgroup were
0.99-1.36x the chosen tile (0.99x on one masked shape), in a 128 x 32 one 1.28-2.24x, and 16 x 16 tiles in a 64 x 32 one
1.50-2.84x.

| us per call (median of 3) | table (before) | this rule | `sdpa_micro` (master) |
|---|---|---|---|
| VAE, one head: 4096 dynamic / static | 747.2 / 757.2 | 420.0 / 418.5 | 552.6 / 424.1 |
| VAE, one head: 16384 / 1024 | 10,910 / 81.2 | 7,126 / 39.4 | 10,323 / 44.0 |
| 2D mask, 8 / 1 heads: 1K / 4K | 477.0 / 6,435 | 232.4 / 3,436 | - |
| 2D mask, 16 / 2 heads: 1K | 835.3 | 510.6 | - |
| causal 8 / 1 heads: 16 over 1K / 2K prompt | 87.3 / 857.5 | 40.7 / 562.5 | 42.8 / 677.7 |

All dynamic unless noted. Against the table it is 0.47-0.66x; against `sdpa_micro` 0.69-0.99x. Output checksums equal
those of the swept configurations, and with `SDPA_OCL_PLAIN_D512=0` those of the build before; the `d_max` 64 / 128 /
256 stages measured alongside are unchanged. Max |out - `sdpa_opt`| is the same as the table's in every case.

End to end on GPU.1 at 1800 MHz (same build with `SDPA_OCL_PLAIN_D512=0` / default, against the master install; ms):
a VAE decoder call (the decoder model alone, median of 30 after warm-ups, 3 rounds, spread <= 0.2 ms) at 512 x 512 /
1024 x 1024 went from 68.53 / 281.91 to 68.20 / 278.09 in stable-diffusion-3.5 (master 68.31 / 281.51), flux 68.53 /
282.09 -> 68.21 / 278.31, SD1.5 68.50 / 281.86 -> 68.16 / 278.05. gemma-4-e2b on the SDPA backend (its head-512 layers masked, 2 rounds): first token at 4212
tokens 387.47 -> 367.88 (master 372.24), at 1024 tokens with an image 103.72 -> 101.70, at 274 tokens 74.28 -> 74.50
(within the spread).

Tests: `sdpa_gpu_noncausal_tail` `*_d512` (one VAE-like head, 1100 keys ending 76 into a key tile; two static heads
333 / 777; a GQA 2D-mask prompt whose 300 queries end 44 into a query tile) against the host reference (max |diff|
1.3e-4, bound 4e-3), `sdpa_gpu_causal_mask.*_d512` (static lower-right 300 / 1100 against the explicit-mask reference,
which takes the masked workgroup, max |diff| 2.4e-4, bound 2e-3; a shape-agnostic 16-query step). With the paired V
read (`WIDE_READS` bit 2, which only these and PA `d_max` 512 use) one key row off, exactly these five fail and the other
77 plain-SDPA rows pass.

### Plain-SDPA causal tile skip

`CAUSAL_TILE_SKIP` ("Paged-attention PREFILL: short prompts") is on for a plain-SDPA PREFILL stage with `is_causal`
that takes the f16 `d_max` 128 loop changes above: the 256 x 128 tile and the table tile, static and shape-agnostic.
`add_prefill_pipeline_jit()` gates it on `config.is_causal`, which PA always sets, so PA keeps its default (non-causal
table-tile stages take its KQ branch for another reason, below).
The kernel already bounded each subgroup by the lower-right `causal_offset = max(0, k - q)` its causal predicate uses
(so with k < q the skip drops a subset of what the predicate masks). A skipped key's score is set to `-INFINITY` by the
predicate, a select that an attention mask added under `is_causal` does not get past, and its V block adds 0, so no
result changes with finite inputs: 75 bench cases (static and dynamic Q + K, q = k 512-8K at 32 / 32, 32 / 8 and 28 / 4
heads; the 16 chunks of an 8K prompt and their dynamic cases; k < q; a diagonal at offset 2000) gave bit-identical
outputs with the skip on and off.

Measured on B70 (GPU.1 pinned at 1800 MHz; one cliloader process per case and side, 100 warm-ups + 400 calls, min
device time, sides interleaved per case, 3 rounds, median, round-to-round spread <= 2.4%), lower-right causal, f16,
head 128, us per call:

| Case (32 / 32 heads unless noted) | skip off | skip on | `sdpa_micro` (master) |
|---|---|---|---|
| static q = k 512 | 69.5 | 54.5 | 67.4 |
| static q = k 1K | 211.1 | 180.7 | 200.1 |
| static q = k 2K | 713.9 | 638.9 | 642.5 |
| static q = k 4K | 2,363 | 2,218 | 2,260 |
| static q = k 8K | 8,447 | 8,184 | 8,410 |
| dynamic Q + K, q = k 2K | 715.0 | 664.9 | 693.0 |
| 16 chunks of 512 queries over 512..8192 keys, sum | 8,871 | 8,587 | 8,924 |
| the same, 28 / 4 heads | 8,253 | 7,996 | 8,295 |

On / off is 0.76-0.99 on every case, the gain shrinking with the share of diagonal tiles (q / k): 0.78 for 512 queries
over 512 keys, 0.98 over 8K. Against `sdpa_micro` every case is 0.75-0.99x except the 512-query chunk over 1.5K keys
(1.006-1.011x); static q = k 2K / 4K went from 1.11x / 1.05x to 0.99x / 0.98x. The table is the env A/B of one build
(`SDPA_OCL_CAUSAL_SKIP=0/1`); the default builds before / after this change repeated every case within 0.9%, with the
same outputs. The "Plain-SDPA PREFILL tiling" numbers were taken with the clock unpinned and are not comparable.

**Non-causal stages: the KQ branch alone.** A non-causal stage of the table tile without an attention mask also takes
`CAUSAL_TILE_SKIP` (`sdpa_ocl_config_t::noncausal_kq_branch`, set with the table tile's loop changes; the mask test is
the one `add_mask_jit()` uses: no const scalar, runtime scalar or tensor mask). There it skips only the subgroups whose
16 keys all lie past `k` (the last tile), yet it is faster: as a runtime branch, `if (key_base < min(k_tile_end,
kq_sg_causal_k))` puts the KQ depth loop in a basic block of its own, and IGC then issues the first two K block reads
ahead of the eight Q SLM reads instead of each right before its DPAS, so the DPAS no longer waits on the K latency.
Evidence (q = k 512, 32 / 32 heads, static): the k0 loop has the same 32 DPAS / 14 block reads / 33 SLM reads either way
and 55 more instructions with the branch (jumps, compares, `sync.allrd`), but the KQ phase reads `KK ssssssss DD ...`
instead of `sKsDD ...`; kernel variants injected through cliloader (`CLI_InjectProgramSource`, same JIT text) gave
skip off 92.7 us, KQ branch only 85.5, S*V skip only 93.6, both 86.5, and the KQ branch with a constant `if (1)` 92.8
(IGC folds it, no block split, old schedule). Why IGC's scheduler picks the late order in the joined block is internal to
IGC; the S*V skip stays causal-only in the kernel (`#elif CAUSAL_TILE_SKIP && IS_CAUSAL`).

| Non-causal, us per call (3 rounds, median) | before | KQ branch | KQ branch + S*V skip |
|---|---|---|---|
| static q = k 512, 32 / 32 heads, head 128 | 92.7 | 85.6 | 86.7 |
| static q 16 / k 1024 | 21.5 | 16.2 | 17.9 |
| static q 1024 / k 1500 (cross-attention shape) | 490.6 | 437.9 | 439.7 |
| shape-agnostic q = k 4K | 5,147 | 4,548 | 4,665 |
| shape-agnostic q = k 2K, head 96 / 112 | 1,169 / 1,318 | 1,108 / 1,278 | 1,110 / 1,270 |
| static q = k 729, 16 heads, head 72 (SigLIP) | 152.7 | 149.8 | 149.2 |
| shape-agnostic q = k 1K / 4K, 16 heads, head 80 (Qwen2-VL vision) | 192.0 / 2,537 | 185.5 / 2,504 | 197.9 / 2,535 |
| with a full 2D mask: static 512 / shape-agnostic 2K, head 128; 4K head 80 | 103.2 / 1,483 / 3,240 | 103.1 / 1,498 / 3,313 | - |

Over 26 non-causal cases without a mask (static and shape-agnostic, 16-8K queries, head 72 / 80 / 96 / 112 / 128, GQA)
the KQ branch was 0.76-0.97x at head 96-128 and 0.97-0.999x at head 72 / 80, never slower; with the S*V skip as well it
was -2% to +10% against the branch alone (and slower than before at head 72 / 80). With a mask it was 0.997-1.022x, and on the
256 x 128 tile 1.002-1.009x, so those keep the old code (their ISA is identical to before). Outputs were bit-identical
with and without it in every case. `sdpa_gpu_noncausal_tail` checks the table tile element-wise against the host
reference where `k` ends inside a key tile and inside a subgroup (max |diff| 1.2e-4-2.4e-4, bound 4e-3, Q in [-4, 4]);
with the non-causal subgroup bound 16 keys early the four rows without a mask fail (0.06-0.21) and the masked one,
which does not take the branch, passes unchanged.

A NaN or Inf in V at a key the causal mask hides no longer reaches the output from inside the diagonal tile (keys past
the workgroup's diagonal were never read). `sdpa_ref` multiplies every V row by its weight, 0 for a masked key, so a
NaN there makes its output NaN. Tests: `sdpa_gpu_causal_mask.prefill_static_*` (256 x 128 tile: upper-left q = k 2K,
lower-right 512 / 2560 and 300 / 2300; table tile: 28 / 4 heads 200 / 1100) and `prefill_32q_32kv_300q_2300kv`
(shape-agnostic) compare element-wise with the explicit-mask reference, Q in [-4, 4]: max |diff| 0 for the static rows
(same tiling on both sides), 1.2e-4 for the shape-agnostic one (bounds 2e-3 / 5e-3). With the skip bounds 16 keys
early all five fail (max |diff| 0.05-0.09; cosine 0.98 for the upper-left one), as do three older cosine-only rows.
Plain `d_max` 64 on Xe2 takes the KQ branch on non-causal stages without a mask but not the causal skip ("Plain-SDPA
PREFILL tiling (`d_max` 64)"); plain `d_max` 512 takes neither ("Plain-SDPA PREFILL tiling (`d_max` 512)"). Not
measured: plain `d_max` 256 (no loop changes there), head sizes below 128 within `d_max` 128, other devices (the ptl-h
build is checked with ocloc only).

### Plain-SDPA decode tiling

Plain (non-PA) SDPA with one query (the `SDPAOptImpl` single-token stage, kernel `sdpa_ocl_mixed_*`; a stateful LLM's
decode with `ATTENTION_BACKEND=SDPA`, optimum-intel, Whisper) with f16 Q / K / V on Xe2 (`solve_plain_decode_tiling()`):
32-key x 16-query subgroup tiles in a single row of subgroups (`kq_sg_per_wg_queries` 1), 16 subgroups = 512 keys per
workgroup at `d_max` 128 / 256 / 512 and 8 = 256 keys at `d_max` 64 (no S*V split of a 64-wide head over 16 subgroups
keeps value tiles of 16 columns and score tiles of 8 rows), 256 GRF, with `PREFETCH_V`, `WIDE_READS` 1 (K), `SOFTMAX_BRANCHLESS` and `MAX_SPLIT_BARRIER`.
`CAUSAL_TILE_SKIP` stays off (`add_prefill_pipeline_jit()`). `d_max` 32, bf16, compressed K / V and other devices keep
the table tile (not measured). The tiling reads no query or key count, so the static and the shape-agnostic stage get the
same one at compile time and dispatch, and a loaded blob re-derives it from the same params. The kernel is unchanged:
the config flag `plain_decode` only lets the PREFILL loop flags reach this stage. The stage used to take the table tile
(16 x 16 subgroup tiles, 8 x 2 subgroups = 128 keys x 32 queries, 64 at `d_max` 64; 128 GRF): with one query, 31 of its
32 query rows are padding, and its second row of subgroups (three of four at `d_max` 64) reads the same K and V tiles
again for padding rows. `sdpa_micro` puts every subgroup on keys (`xehpc_h64_2nd` 4 subgroups and `xehpc_h128_2nd` 8 at
256 GRF, `xehpc_h256_2nd` 16 at 128 GRF), one workgroup per query head, like this tiling.

**The bench data compresses.** The unit-test generator's default resolution (1/8: 17 distinct values in [-1, 1]) gives
data the B70 reads at ~900 GB/s, against 576-600 GB/s for random bits (a read kernel over 134 MB and 1 GB; no gain left at
resolution 1/64; `test/xe3_test/b70_repro/bw/bw.cpp`). Decode streams K and V once per call, so on that data it measured
faster than a real cache allows, and the gap to `sdpa_micro` looked larger than it was: 32 / 32 heads, 8K keys, head 128,
before this change, 240.1 against 192.0 us (1.25x) on the default data and 264.7 against 253.9 us (1.04x) at resolution
1/1024 (`SDPA_BENCH_RES`), where reading the 134 MB of K and V takes 244 us. The numbers below use resolution 1/1024.

Measured on B70 (GPU.1 pinned at 1800 MHz; one cliloader process per case and side, 100 warm-ups + 400 calls, mean device
time of the SDPA kernel, sides interleaved per case, 3 rounds, median, spread <= 0.6%), static Q / K / V, lower-right
causal, us per call:

| Case | before | this tiling | `sdpa_micro` (master) |
|---|---|---|---|
| head 64, 32 / 8 heads, 1K / 4K / 16K keys | 30.8 / 110.0 / 740.5 | 10.1 / 33.6 / 159.6 | 26.7 / 97.7 / 475.3 |
| head 64, 32 / 32 heads, 4K / 16K | 184.7 / 739.4 | 42.6 / 230.9 | 119.2 / 481.3 |
| head 128, 32 / 8 heads, 1K / 4K / 8K / 16K / 32K | 21.1 / 73.8 / 227.4 / 474.3 / 967.9 | 11.5 / 38.4 / 82.3 / 166.5 / 350.0 | 19.4 / 68.6 / 167.1 / 349.0 / 704.1 |
| head 128, 28 / 4 heads, 1K / 4K / 16K | 21.2 / 73.7 / 471.0 | 11.5 / 38.5 / 155.4 | 19.4 / 68.6 / 333.2 |
| head 128, 32 / 32 heads, 1K / 4K / 16K | 21.1 / 133.9 / 529.3 | 11.5 / 128.4 / 453.7 | 19.4 / 128.0 / 501.7 |
| head 128, 40 / 8 heads, 4K; head 96, 32 / 32, 4K | 107.7; 123.8 | 78.6; 89.1 | 85.5; 100.4 |
| head 128, 32 / 8, 4K: mask (non-causal); dynamic Q + K; dynamic + mask | 76.8; 74.9; 78.4 | 39.0; 39.4; 40.7 | 68.6; 71.3; 71.3 |
| head 256, 32 / 8 heads, 4K / 16K | 223.2 / 925.7 | 78.4 / 312.6 | 88.8 / 380.9 |
| head 256, 16 / 8 heads, 4K / 16K | 221.1 / 914.3 | 77.9 / 318.5 | 90.5 / 396.8 |
| head 256, 8 / 4 heads, 1K / 2K / 4K / 8K / 16K | 40.0 / 74.9 / 145.3 / 443.7 / 890.9 | 21.3 / 37.5 / 69.2 / 142.6 / 292.6 | 18.0 / 34.0 / 65.9 / 175.3 / 373.9 |
| head 256, 32 / 32 heads, 4K | 255.6 | 234.2 | 247.3 |
| head 512, 16 / 2 heads, 1K / 4K / 16K | 88.3 / 338.2 / 1,623 | 37.8 / 133.2 / 521.7 | 53.9 / 152.6 / 604.3 |

Over these 38 cases the tiling is 0.22-0.96x the table tile (never slower) and 0.34-1.19x `sdpa_micro` (geomean 0.65).
It trails `sdpa_micro` only with few heads at head 256 and up to 4K keys: 8 / 4 heads (8 workgroups on 32 Xe cores) at
1K / 2K / 4K keys, 1.19x (a 21 us kernel) / 1.10x / 1.05x; from 8K it is 0.78-0.81x. At 32 / 32 heads, head 128, 4K both
read at the DRAM rate (1.003x). On the original kit's compressible data (32 / 32 and 28 / 4 heads, 8K keys) it went from
242.0 / 144.3 to 154.2 / 74.0 us, `sdpa_micro` 198.1 / 134.2. With `SDPA_OCL_PLAIN_DECODE=0` every case gave the same
output bits as the build before this change. Against the tile it replaces the outputs differ (another key order per
workgroup), by the same margin as before: max |out - `sdpa_opt`| unchanged on the bench cases, against the host
reference 3.0e-5-2.5e-4 on the decode test rows (the table tile 3.2e-5-2.5e-4).

How it was chosen (tiling first with the `SDPA_OCL_KQ_*` / `SDPA_OCL_256GRF` overrides, then the loop flags with
`SDPA_OCL_*` on a build with the tiling; same protocol, 2 rounds, 10-20 shapes, heads 64-512):
- One query block and every subgroup on keys is what removes the gap; with it, 32-key subgroup tiles beat or matched
  16-key ones at 256 GRF, and at 128 GRF the 32-key tile was 1.3-1.8x slower at `d_max` 128 / 256 (equal at 64). 64-key
  tiles do not build (`kq_sg_tile_keys` is 16 or 32).
- `PREFETCH_V` is the largest of the loop changes: at head 256 / 32 / 8 heads the tiling alone was 1.51-1.56x
  `sdpa_micro`, with `PREFETCH_V` 1.16-1.22x, with the branchless softmax and the split barrier as well 1.02-1.03x, with
  16 subgroups and the wide K read 0.82-0.88x. It costs where the call is short: at head 512 and 4K keys 16 subgroups
  without any loop change were 0.67x `sdpa_micro` and with them 0.87x (but 1.11x against 0.86x at 16K), and the few-head
  head-256 cases above are the same trade.
- 16 subgroups beat 8 at `d_max` 128-512 (head 128, 32 / 8 heads, 4K: 0.56x against 0.67x `sdpa_micro`; head 256,
  8 / 4 heads, 4K: 1.05x against 1.07x).
- `CAUSAL_TILE_SKIP` made the tiling 1.40-1.75x slower. A prefetch of the next key tile's K ahead of the S*V
  (`sdpa_micro`'s `PREFETCH_K`, a kernel change) gave 1-5% at `d_max` 64 and cost 6-25% at 256 and 10-27% with 16
  subgroups at 128, so it was dropped and the kernel left as it was.

Tests: `sdpa_gpu_plain_decode` (key counts ending inside a workgroup tile and inside a subgroup, 17 keys, GQA ratios 4
and 7, heads 64 / 72 / 80 / 128 / 256 / 512, static and dynamic, with and without a mask, and one dynamic network running
a prompt and then decode steps) and four decode rows of `smoke_sdpa_gpu_test` (batch 2, per-head mask, also through a
saved and loaded blob), all against the host reference with Q in [-4, 4] (bound 4e-3). A build that drops the last
16-key S*V block of every decode workgroup tile failed exactly the 22 decode rows that have keys there (0.038-0.10), and
passed the 17-key row and `sdpa_gpu_causal_mask.decode_40q_10kv_444seq`, which have none.

### Kernel flow

1. Q is staged to SLM (`Q_slm`) cooperatively: the `q_blocks x DKS_ACTIVE` (query block, depth
   tile) tiles are dealt round-robin over all subgroups.
2. The key range is bounded: `causal_k` (causal upper bound), `window_k0_begin` (sliding-window
   lower bound), both extended for bidirectional image groups.
3. For every `kq_wg_tile_keys` key tile `k0`:
   - K tiles are loaded per depth tile `db` and multiplied with DPAS. The A operand is K itself
     (`lane == head dim`, 8 keys per row-block), B is Q from SLM, so the score tile lands
     `lane == query`.
   - Masks are applied and the running max is combined across subgroups with an SLM atomic max
     (`S_max_slm`).
   - The softmax exponentials go to `S_slm` in the input's 16-bit type, and the previous output is
     rescaled by alpha.
   - S*V: A is the score tile from SLM (`lane == key`), B is V in VNNI layout (`lane == value
     column`).
4. The output is divided by the summed denominators (`S_sum_slm`) and stored.

`MICRO_MATH=1` (PA PREFILL f16 without sink/alibi/sliding window/token_type_ids) reproduces
`sdpa_micro`'s rounding: raw maxima subtracted before scaling, keys summed in order, and each S*V
key tile accumulated from zero and added afterwards (`A_tile1`), as `ugemm_vs` does. The `d_max` 512 PREFILL
tile runs without it (see "Paged-attention PREFILL tiling").

The alpha rescale selects `alpha[alpha_qb]` with a select chain: `alpha_qb` comes from `sg_ij`, a
runtime value, and a runtime-indexed private array is moved to scratch. On the llama-3.2-1b MIXED
kernel that was `private memory size 128` (`kq_query_blocks` floats x 16 lanes x 4 bytes) plus two
scratch stores and two scratch loads inside the k0 loop.

### Key-range bounds

Without the causal bound the loop walks the full key range and every tile past the diagonal is
loaded, multiplied and masked away: at q = k = 1024 with `kq_wg_tile_queries = 32` that is 256
tile iterations instead of 144 (1.78x). `sdpa_micro` has had this bound
(`causal_k = min(k, wg_j0 + wg_tile_n)`) since the start.

The block-level causal skip (`causal_block_clear`) is the counterpart of `sdpa_micro`'s
`causal_k_end > causal_q_begin` guard. Measured at q = k = 4096 with the default tiling, 96.6% of
(key x query) blocks are entirely inside the causal region; without the skip every one of them
still paid 32 selects and 8 compares per (qb, k0) iteration, 16% of the loop body.

Within the diagonal tile, PA PREFILL (and plain-SDPA causal PREFILL at f16 `d_max` 128) also skips the KQ of a subgroup
whose keys all follow its queries and the S*V blocks after a subgroup's score rows (`CAUSAL_TILE_SKIP`, "Paged-attention
PREFILL: short prompts", "Plain-SDPA causal tile skip").

For `CAUSAL_MASK_LOWER_RIGHT` (plain SDPA) every query moves by `causal_offset = max(0, k - q)`,
the shift `sdpa_micro` / reference / `sdpa_opt` apply. Without it a single new token (q == 1,
k == 512) would only see the first tile of keys. The sliding-window lower bound (`window_k0_begin`)
takes the same shift: without it a chunk or a single token on a longer past never skips a tile (the
result is the same, every element below the window is masked one by one).

### Paged-attention MIXED: current tokens from Kc/Vc

With `PA_CUR_KV_F16` the MIXED kernel reads keys `[0, past_len)` from the paged cache and the new
keys `[past_len, k)` from the raw f16 K/V inputs (`Kc`/`Vc`). A compressed cache is a lossy round
trip, so reading the new rows back from it is not equivalent. An f16 cache is not, and reads its new rows from the page
by default ("Paged-attention MIXED: i8 and f16 caches"). A key tile that crosses `past_len` is
shortened (`k_chunk`) so that every iteration reads one source, which is `sdpa_micro`'s boundary
handling. For u4 two optimisations ride on this split: whole S*V blocks past `k_chunk` are skipped,
and the Vc tiles are prefetched before the softmax.

The u4 cache forces a permuted depth axis (below), which a 16b block read of Kc cannot produce.
The u4 Kc read therefore views Kc as a DWORD surface: channels `(win + 2L, win + 2L + 1)` are one
dword, and the parity is a half-select. It reads each window twice (once per parity) — 2x L1
traffic on a tile that is already cached, and still about 4x fewer instructions than the page
path's nibble extract, zero-point subtract and scale multiply. It needs the head's first channel
at an even half offset from the (64B-aligned) surface origin. That depends on runtime padding, so
the kernel tests it (`kc_dword_ok`) and falls back to a per-lane gather from Kc when it fails.

### Paged-attention MIXED tiling (chunked prefill)

With GenAI's continuous-batching scheduler a prompt is cut into chunks of `max_num_batched_tokens` (256 by default); only
the first chunk runs PREFILL, every later one runs MIXED over the cache of the chunks before it. The daily benchmark with
`max_num_batched_tokens` 256 had every model with head 256 or 512 slower than master, by more with the prompt length:
gemma-2-9b 1.17x at 4K, gemma-3-4b 1.25x and gemma-4-26b 1.25x at 32K, gemma-4-e2b 1.35x at 32K, qwen3.5-9b / 3.6-35b /
3.8-27b 1.22x / 1.14x / 1.26x at 32K, gemma-4-26b and qwen3.6-35b 1.38x / 1.35x at 128K; head 64 / 128 models were level
or faster. The same build with `TEST_USE_SDPA_OCL=0` matched master (qwen3.5-9b, 8K). MIXED ran the table tiling: 16-key x 16-query subgroup
tiles at 128 GRF, which spilled 4,480 B at `d_max` 256 and 12,992 B at 512, while `sdpa_micro`'s MIXED had moved to
`xe2_h256_pa_mixed` (16 x 32 subgroup tiles, 8 x 2 subgroups, 256 GRF) for head 256. On qwen3.5-9b (16/4 heads, u4
cache, 8K prompt, cliloader, B70 at 1800 MHz) `sdpa_ocl_mixed` took 1,431 us per call against `sdpa_micro`'s 862.

A subgroup dequantizes its K and V cache tiles once for the queries of its tile, so the subgroup query tile sets the
dequant cost per query. `solve_pa_mixed_tiling()` (Xe2, MIXED only) gives MIXED its own tiles:

| `d_max` | KQ subgroup tile, subgroups | Workgroup, keys x queries | 256 GRF |
|---|---|---|---|
| 128, 256 | 16 x 32, 8 x 2 | 128 x 64 | yes |
| 256 with few heads, 512 | 16 x 32, 16 x 1 | 256 x 32 | yes |

64 and 32 keep the table (at 64 no tile measured was faster on llama-3.2-1b at 8K: table 627 us, 128 x 32 at 128 GRF 627,
the 256 GRF tiles 705-1,598). At 512 the 8 x 2 tile spilled 14 KB. "Few heads" is the number of heads up to which a 256-token chunk
in 256 x 32 workgroups is one wave of 256 GRF threads, `heads * 8 * 16 <= EUs * 4`, i.e. 8 heads on the B70 (256 EUs):
8 x 2 leaves half of those threads idle there. Per call at 8K, 8 x 2 against 16 x 1: gemma-3-4b (8 heads) 793 / 564 us,
gemma-4-e2b's head 256 layers (8 heads, 512-key window) 100 / 92 us, but qwen3.5-9b (16 heads) 827 / 1,144 us and
gemma-2-9b (16 heads, 4K) 434 / 592 us. The rule is measured on the B70 only. The S*V split follows from
`solve_sv_split()`: 32 values x 32 scores for 8 x 2, 16 x 32 for 16 x 1.

Three changes to the u4 loop body went with it, found in the ISA (IGC ShaderDump) of the new tile:

- **u4 dequant without a conversion.** The nibble used to be extracted from a dword, converted (`mov` to a strided
  half), repacked and then `(q - zp) * scale`: 6-7 instructions per element. `U4_TO_HALF` builds the half directly,
  `as_half(0x6400 | n) - 1024` with 16-bit integer ops on a word of the read (`bfn`, `shr`), which IGC keeps packed;
  `1024 + n` is exact and so is the subtraction, so the value is the old one bit for bit. About 5 instructions per element
  remain: the `- 1024` cannot be folded into the zero point, which is not an integer.
- **V page indices once per key tile** (`pa_v_pages()`): the S*V loop looked each cp page up twice per block, the comp
  through per-lane branches, a dependent load chain inside the loop. One lane-parallel load per tile now, and the comp
  loads take a select.
- **Prefetch of this tile's cache pages** at the start of each key-tile iteration (`PA_MIXED_PREFETCH_V`,
  `PA_MIXED_PREFETCH_K`): every V block column the subgroup reads in the S*V loop (`8b_32r16x1c`, the read's own surface),
  and the subgroup's whole K page rows (`8b_16r16x4c` per 64 B). The loop issued each read right before consuming it.
  The K prefetch is on with 8 x 2 only (two query subgroups read each K page); with 16 x 1 it was +0.5%.

Per call at 8K, qwen3.5-9b `d_max` 256 (`sdpa_micro` 862 us): table 1,431, the 8 x 2 tile 934, + dequant and page indices
827, + V prefetch 743, + K prefetch 740. Skipping the dequant altogether (wrong values, timing only) gave 697 before the
prefetches. VTune at 827 us: XMX instructions equal to `sdpa_micro`'s (22.5 G), I-cache miss 2.8% against 0.2%.

| Kernel, 8K prompt, us per call | `sdpa_micro` | before | after |
|---|---|---|---|
| qwen3.5-9b, `d_max` 256 (16 heads) | 862 | 1,431 | 740 |
| gemma-3-4b, `d_max` 256 full / 1024-key window (8 heads) | 819 / 177 | 793 / 189 (a) | 500 / 138 |
| gemma-4-e2b, `d_max` 512 / 256 with 512-key window (8 heads) | 1,675 / 98 | 2,746 / 113 (a) | 922 / 81 |
| gemma-4-26b, `d_max` 512 / 256 with 1024-key window (16 heads) | 3,284 / 211 | | 1,942 / 201 (b) |
| llama-3.1-8b, `d_max` 128 | 1,507 | 1,511 | 906 |
| llama-3.2-1b, `d_max` 64 (table) | 753 | 675 | 627 |

(a) the `d_max` 256 kernel already with the 8 x 2 tile. (b) before the prefetches. At 32K (before the prefetches) qwen3.5-9b
was 3,257 against 3,424 us and gemma-3-4b's full-attention layers 2,216 against 3,270 us.

First-token latency against master end to end (`test/run_daily_bench.py` with `max_num_batched_tokens` 256, master and this
build run back to back per model and size, alternating which goes first, B70 at 1800 MHz; 32 / 1K / 2K / 4K / 8K / 16K /
32K tokens): gemma-3-4b 1.014 / 0.927 / 0.906 / 0.899 / 0.882 / 0.853 / 0.803, gemma-4-e2b - / 0.911 / 0.927 / 0.898 /
0.850 / 0.779 / 0.697, gemma-4-26b - / 0.925 / 0.966 / 0.951 / 0.924 / 0.875 / 0.800 and 0.641 at 128K, gemma-2-9b 0.951 /
0.966 / 0.948 / 0.927 (to 4K), qwen3.5-9b - / 0.995 / 0.991 / 0.980 / 0.974 / 0.960 / 0.935, qwen3.6-35b - / 0.994 / 0.997 /
0.991 / 0.985 / 0.971 / 0.951, qwen3.8-27b - / 0.994 / 0.991 / 0.987 / 0.975 / 0.958 / 0.934, qwen3-8b 0.987 / 0.938 / 0.879
/ 0.810 / 0.737 / 0.679 / 0.640, llama-3.2-1b 0.924 / 0.962 / 0.942 / 0.904 / 0.880 / 0.863 / 0.847, gpt-oss-20b 1.001 /
0.995 / 0.992 / 0.986 / 0.972 / 0.952 / 0.926. Of the 128 (model, length) pairs of the run none was above 1.02; the two above
1.0 are prompts that run no MIXED (gemma-3-4b at 32 tokens, phi-3.5-vision, which does not take the PA backend). The
qwen3.5 / 3.6 / 3.8 gain is smaller because a quarter or less of their layers is attention. The saved results of earlier
runs could not serve as the baseline: the build before this change measured again was about 1.2x slower on every model,
the second token included, so they were taken under other clock conditions.

The other caches took the tiling and the page indices here; per call at 8K against `sdpa_micro`: i8 qwen3.5-9b 1,232 to
894 (797), llama-3.1-8b 1,536 to 1,264 (1,299), gemma-3-4b 1,070 to 593 (729); f16, whose K page stays d-major and was read
by the per-key gather, qwen3.5-9b 1,679 to 1,465 (522), llama-3.1-8b 1,850 to 1,658 (950), gemma-3-4b 1,445 to 799 (454).
Their own changes are in the next section.

Measured and not used: reading each u4 K window once for its tile pair and one window ahead (double buffer), +17% (one read
per window without the look-ahead +15%; the window re-read per depth tile is an L1 hit); a prefetch of the next tile's K
and V pages instead of this tile's, +8%; `SOFTMAX_BRANCHLESS` and `MAX_SPLIT_BARRIER` in MIXED, 0%; the u4 dequant through
a byte convert or with dword magic constants (more moves than the word form).

Value checks: with 8 x 2 the key tile and the subgroup key split are the table's (128 keys, 8 subgroups), so the online
softmax rescales at the same keys and sums in the same order: of the 173 `basic` PA rows 169 are bit-identical to the
build before, and the four that differ are the sliding-window rows at head 128 (`basic/112`, `116`, `131`; the window's
first key tile is rounded down from the workgroup's first query, which moves with the query tile) and the head 512 row
`basic/114` (256-key tile). The PA, `*sdpa*:*rope*:*kv_cache*:*update_shape_test.paged*` and
`*large_head*:*kv_cache_rotation_content*:*shared_kv_cache*` suites fail the same 13 / 60 / 0 tests as before.
`basic/173`-`179` are u4 MIXED rows for the new tiles (head 256 with 8 and 16 heads, 512, 128; cliloader shows the
intended workgroups); swapping the V nibble or the K channel parity on purpose fails the sharp-softmax ones (174, 176,
178, 179) and none of the plain ones, whose near-uniform softmax averages the error away. The upstream
`paged_attention_h256_mixed_qtile64` rows assert that `sdpa_micro` runs and fail on this branch for that reason alone.

### Paged-attention MIXED: i8 and f16 caches

**i8** (the default cache of models without 4-bit weights: BY_CHANNEL, token-major page) after the MIXED tiling above:

- **Exact widen.** `I8_TO_HALF` forms the K and V page bytes as `as_half(0x6480 ^ b) - 1152` on 16-bit words of the
  read, as `U4_TO_HALF` does for u4. `0x6480 ^ b` is `0x6400 | (b ^ 0x80)` and `b ^ 0x80` is the signed byte plus 128, so
  the value is `(half)(char)b` bit for bit; the convert it replaces landed strided and took a repack move per value. The
  `- 1152` stays separate: the zero point is not an integer. qwen3.5-9b's hot loop went from 7,690 to 5,842 instructions
  (moves 5,147 to 2,003).
- **V page prefetch** (`PA_MIXED_PREFETCH_V`) on every tiling, the `d_max` 64 table tiling included. The K prefetch stays
  off for i8: from 0 to +1.6% on the models below.

| i8 cache, 8K prompt, us per call | `sdpa_micro` | before | exact widen | + V prefetch |
|---|---|---|---|---|
| qwen3.5-9b, `d_max` 256 (16 heads) | 797 | 894 | 799 | 709 |
| llama-3.1-8b, `d_max` 128 | 1,299 | 1,266 | 1,167 | 972 |
| gemma-3-4b, `d_max` 256 full / 1024-key window | 751 / 168 | 609 / 157 | 563 / 146 | 490 / 129 |
| gemma-4-e2b, `d_max` 512 / 256 with 512-key window | 1,012 / 86 | 1,010 / 99 | 955 / 94 | 902 / 82 |
| llama-3.2-1b, `d_max` 64 (table) | 709 | 630 | 630 | 527 |
| gpt-oss-20b, `d_max` 64 full / 128-key window | 1,345 / 55 | 1,197 / 67 | 1,171 / 66 | 961 / 61 |

The 180 `basic` PA rows are bit-identical to the build before. An unsigned widen put in on purpose changes exactly the 18
i8 MIXED rows (`basic/77`, `78`, `80`, `104`, `107`, `109`, `111`-`117`, `127`, `129`-`132`; 17 of them fail), so those rows
run the new widen.

**f16** (`OV_KV_CACHE_PRECISION=f16`, never a default): the K page was d-major, `[head_size, 16]`, a 32-byte row per
channel under the block2d minimum, so MIXED gathered K one key per message (qwen3.5-9b: 1,465 us against `sdpa_micro`'s
522). A token-major page took 441 us, but changed the layout for every reader of the page ("Paged-attention cache
layouts"); it has been the default on the `sdpa_ocl` lane since. `pa_k_tile_dt()` (`USE_2D_BLOCK_IO_K_PA_DT`) read the
d-major page in place instead. It served `OV_GPU_PA_K_TOKEN_MAJOR=0` and the models the layout rule turned down until
2026-11-02, when MIXED became token-major only (a d-major page runs `pa_multi_token`); kept here as a record:

- Viewed as a 64-byte-pitch surface, row `r` holds channels `2r` and `2r + 1`, 16 keys each, as 8 dwords of key pairs.
- A 32-bit transpose read gives lane `L` surface row `y + L` with element `i` at dword column `x + i` (probed on the B70:
  `test/slower/probe/probe_t32.cpp`). At `x = 8 * (db & 1)`, `y = 16 * (db >> 1)` lane `L` holds channel
  `32 * (db >> 1) + 2L + (db & 1)`, which is the u4 permuted depth (`PA_K_U4_CHANNEL`, "u4 K: the permuted depth axis"):
  the u4 Q staging and Kc reads serve it unchanged (`PA_K_PERMUTED`).
- The `k_raw` rows are the word halves of those dwords. The instruction that extracts a row also ANDs it with its key's
  lane of a mask that is all ones below the page's valid rows, which keeps never-written slots (a NaN would survive the
  masked score) out without a branch; a branch there cost 16 phi moves per depth tile (565 against 508 us).
- The 32-row read (`PA_K_DT_ROWS` 32) interleaves rows `y + L` and `y + 16 + L` dword by dword, i.e. depth tiles `db` and
  `db + 2`, in one message: 460 against 508 us. It keeps 16 dwords per parity live across the depth loop, which spilled
  4 KB at 128 GRF (the `d_max` 64 table tiling: 497 against 413 us), so kernels compiled with 128 GRF take the 16-row read.
- Gate: f16/bf16 d-major page, 16-token pages, `K_HEAD_SIZE % 32 == 0` (whole 32-channel windows; head 80 keeps the gather).

An f16 cache also reads the new tokens from its pages rather than from Kc/Vc (`PA_CUR_KV_F16` 0): the page holds the same
bits, and leaving the Kc branch out of the key loop was 5-14% faster per call.

| f16 cache, 8K prompt, us per call | `sdpa_micro` | before (gather) | transpose read | + new tokens from the page | token-major page (Kc on) |
|---|---|---|---|---|---|
| qwen3.5-9b, `d_max` 256 | 522 | 1,464 | 459 | 407 | 441 |
| llama-3.1-8b, `d_max` 128 | 949 | 1,657 | 652 | 562 | 605 |
| gemma-3-4b, `d_max` 256 full / 1024-key window | 468 / 110 | 821 / 205 | 312 / 90 | 297 / 85 | 296 / 84 |
| gemma-4-e2b, `d_max` 512 / 256 with 512-key window | 602 / 61 | 1,440 / 124 | 470 / 67 | 403 / 63 | 403 / 62 |
| phi-3.5-mini, head 96 (`d_max` 128) | 882 | 1,389 | 661 | 588 | 604 |
| llama-3.2-1b, `d_max` 64 (table, 16-row read) | 526 | 671 | 413 | 392 | 401 |
| gpt-oss-20b, `d_max` 64 full / 128-key window | 972 / 44 | 1,202 / 67 | 732 / 54 | 684 / 52 | 694 / 48 |

The permuted depth reorders the KQ sum, so an f16 MIXED output may change in rounding; the three f16 MIXED rows of the
earlier set (`basic/37`, `38`, `103`, head 64) came out bit-identical anyway. `basic/180`-`185` are sharp-softmax f16
MIXED rows (head 256 with 8 and 16 heads, 512, 128, 64 and 96, past lengths and totals that end inside a page), and
reading the other parity on purpose fails all nine. The PA / sdpa / misc suites fail the same 13 / 60 / 0 tests as
before.

First-token latency of this branch against master end to end (`test/run_daily_bench.py` with `max_num_batched_tokens`
256, master and this build back to back per model and size, alternating which goes first; B70 at 1800 MHz, CPU load
logged with no other job running): i8 cache at 8K / 32K, gemma-3-4b 0.913 / 0.846, llama-3.1-8b 0.862 / 0.779, qwen3.5-9b
0.984 / 0.955; f16 cache at 2K / 8K / 32K, gemma-3-4b 0.964 / 0.947 / 0.882, llama-3.1-8b 0.937 / 0.809 / 0.670, qwen3.5-9b
0.997 / 0.980 / 0.931. Second-token latency 0.91-1.00 with i8 and 0.99-1.005 with f16 (GENERATE kernels unchanged). These
compare the whole branch with master; the kernel tables above isolate the changes of this section.

Still slower than `sdpa_micro` per call: the small sliding-window layers, gpt-oss-20b's 128-key window at `d_max` 64 with
every cache (f16 52 against 44 us, i8 61 against 55, u4 69 against 57) and gemma-4-e2b's 512-key window at `d_max` 256
with f16 (63 against 61).

### Token-major f16 page

The token-major f16 page is the `[key, head]` geometry of the K input, read by `pa_k_tile_b2d16` one 16-key x 16-channel
block per depth tile. Against the d-major page's 32-row transpose that is twice the messages, and the reads of a key
group were serialised: IGC put them all in one register range, so each waited for the DPAS of the previous one. On a
small GQA MIXED step (4-16 new tokens, 32/8 heads, head 128; few workgroups, so the load latency is exposed) `sdpa_ocl`
took 1.08-1.19x the d-major page's time. `PA_K_WIDE_READS` (`pa_k_tile_b2d16_pair`) reads two depth tiles per
`16b_16r16x2c` message and issues the reads of `PA_K_WIDE_TILES` (4) depth tiles together, two messages in flight per key
group as the 32-row transpose has, into 32 GRF of `k_wide_pa`. Per call (B70 pinned at 1800 MHz, `test/ktm/run_smallq.py`,
16 new tokens at 1K / 4K / 8K): d-major 45.3 / 145 / 440 us, token-major before 52.2 / 172 / 475, one wide read in flight
50.3 / 166 / 462, two 47.0 / 154 / 440, four (`PA_K_WIDE_TILES` 8, 64 GRF at head 128) 46.6 / 153 / 436. On a 256-token
chunk at 8K (`test/ktm/arms_f16w.list`, us per call, d-major / before / after): llama-3.1-8b 566 / 556 / 544, qwen3-8b
565 / 558 / 545, llama-3.2-1b 392 / 379 / 367, phi-3.5-mini 588 / 569 / 567, gpt-oss-20b 687 / 649 / 633 and 128-key
window 52.9 / 48.5 / 52.1, gemma-4-26b `d_max` 512 813 / 832 / 726 and 1024-key window 120 / 115 / 113; no kernel spills.
Swapping the two reads of a group on purpose fails the f16 MIXED rows (`basic/37`, `38`, `180`-`185`) and the feature-pad
and runtime-scale suites only with the token-major page.

`paged_attention_opt.cl` read the uncompressed and i8 BY_TOKEN token-major pages until 2026-11-01, with a per-lane
`vload16` and, on Xe2, a 32-bit transpose of the page (`USE_2D_BLOCK_IO_K_TM`) that brought its GQA multi-token fallback
to its d-major time (16 new tokens at 1K / 4K / 8K: 78.6 / 240 / 535 against 78.8 / 245 / 540 us). Both were removed then
and came back on 2026-11-06 together with BY_CHANNEL readers, when the page became token-major for every op
("Paged-attention cache layouts").

## Paged-attention cache layouts

A K cache page is either d-major (`[head_size, block_size]`, the upstream layout) or token-major
(`[block_size, head_size]`, the same as V). The layout is decided once per model in `transformations_pipeline.cpp`, by one
static rule for every kind of page, `paged_attention::k_token_major_rule()`. It looks at the device and the config, never
at the ops: the page is token-major where

- `k_token_major_device()`: Xe2+ with XMX and a oneDNN build, the devices that have `sdpa_ocl` (MIXED) and
  `sdpa_ocl_decode` (GENERATE). The device only: `TEST_USE_SDPA_OCL_DECODE=0` and `OV_GPU_ENABLE_SDPA_MICRO=0` keep the
  token-major page. Other devices (no XMX, xe_hpg, xe_hpc) keep the upstream d-major page and kernels;
- the inference precision is f16 (`sdpa_ocl_decode` reads an f16 query only, and the compressed pages' comp is f16 only
  there); f32 and bf16 inference keep the d-major page;
- the page is a kind every reader reads token-major (`k_token_major_kind()`, given the configured kv-cache precision):
  f16, i8 BY_TOKEN, i8 or u4 BY_CHANNEL; not u8 (`sdpa_ocl_decode` widens the byte as signed), i4, f32 or bf16.

Every kernel that can serve a paged-attention stage reads the token-major page, so a gate that refuses an op changes
only which kernel reads it. `paged_attention_opt.cl` (`pa_single_token`, `pa_gqa_single_token`, `pa_multi_token` and its
GQA multi-token fallback) reads every kind: f16 with a per-lane `vload16` or, where the block2d pitch rule allows, a
32-bit transpose of the page (`USE_2D_BLOCK_IO_K_TM`); i8 BY_TOKEN and BY_CHANNEL with `vload16` rows; u4 BY_CHANNEL K
by row bytes, and the relaid-out u4 V page (split nibbles, trailing per-token comp). It serves what `sdpa_ocl` and
`sdpa_ocl_decode` refuse: an op with alibi, a scores output (cache eviction) or adaptive R-KV, a head size either
rejects, k != v without an `sdpa_ocl` tiling, and every MIXED step under `OV_GPU_ENABLE_SDPA_MICRO=0` or GENERATE step
under `TEST_USE_SDPA_OCL_DECODE=0`. The adaptive R-KV diversity kernel reads every kind too. Until 2026-11-06 the rule
was per op (`k_token_major_readable()`): one op that `sdpa_ocl` or `sdpa_ocl_decode` would refuse kept the whole model
d-major, and `PagedAttentionOptImpl::update_rt_params()` threw when a token-major page met another reader. What the
static rule costs such an op is the opt kernels' token-major read in place of their d-major one ("Token-major K page by
a static rule (2026-11-06)" in "Known issues"). `smoke_paged_attention_by_channel_tm_any_reader` and
`smoke_paged_attention_tm_opt_reader_scores` force a token-major page onto ops the DPAS kernels do not serve (scores
output, `OV_GPU_ENABLE_SDPA_MICRO=0` / `TEST_USE_SDPA_OCL_DECODE=0`); `OV_GPU_PA_K_TOKEN_MAJOR=0` runs them on the
d-major page as the control.

Where `sdpa_ocl` serves MIXED on the token-major page, a small GQA MIXED step (at most 16 new tokens per subsequence)
goes to the GQA fallback, `pa_small_q_opt.cl` or `sdpa_ocl`, chosen in `select_gqa_multi_tokens_kernel()` by cache kind,
head size, new tokens and context ("Small GQA MIXED step on the token-major page (2026-11-08)" in "Known issues"). An op
`sdpa_ocl` refuses, and a d-major page, keep upstream's choice: the fallback while new tokens^2 x context is at most 64K
(`prefer_gqa_multi_tokens_fallback_kernel()`), `pa_small_q_opt.cl` above it.

`OV_GPU_PA_K_TOKEN_MAJOR=0` makes every page d-major (the control for the token-major readers); unset is the rule. `1`
(the f16 / BY_TOKEN page token-major on every device, read by the opt kernels) and `OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR` are
retired and rejected. Every consumer -- the writer, rotate, `pa_kv_reorder`, the
attention kernels, `graph/paged_attention.cpp` and `ops/paged_attention.cpp` -- reads the layout back off the physical
cache shape with `paged_attention::k_cache_token_major()` (the block dimension sits at dim 2 in a token-major page and at
dim 3 in a d-major one; where the shape cannot tell, a dynamic dim or a square page, the page is d-major), and every OCL
kernel that reads a token-major page gets the result as one jit flag, `IS_KEY_TOKEN_MAJOR`, which it splits by the quant
mode (`KEY_BY_TOKEN_TM` /
`KEY_BY_CHANNEL_TM`; `sdpa_ocl` gets no layout flag, only `IS_PA_K_BY_CHANNEL`). The `sdpa_ocl` MIXED stage reads the
token-major page only, with block reads, or the per-key gathers (`PA_K_TOKEN_STRIDE` / `PA_K_HIDDEN_STRIDE`) where the
block2d rule rejects the pitch, and `can_use_sdpa_ocl_for()` sends MIXED on a d-major page to `pa_multi_token`. Until
2026-11-02 it also read a d-major f16 page with a 32-bit transpose read ("Paged-attention MIXED: i8 and f16 caches") and
a d-major i8 BY_TOKEN page through a per-key gather. On a d-major page the MIXED stage still compiles, because the
constructor adds it with PREFILL; it is compiled as the token-major reader and never dispatched.

A token-major page has eight readers: the cache writer, rotate, `pa_kv_reorder`, `sdpa_ocl` (MIXED),
`pa_small_q_opt.cl` (MIXED with at most 16 new tokens per subsequence, GQA), `sdpa_ocl_decode` (GENERATE),
`paged_attention_opt.cl` (MIXED and GENERATE) and the adaptive R-KV diversity kernel.

qq_bias models (EAGLE3) get the token-major page too: EAGLE3 reorders the cache from a separate kv-update model through
`pa_kv_reorder`, which has no paged-attention op to learn the layout from. Its impl reads the layout off the cache shape:
a static shape picks the d-major or token-major kernel at construction; a dynamic one compiles both where a token-major
page can exist (`k_token_major_device()`) and picks per dispatch, forcing the kernel
arguments on the first run of each. The token-major BY_CHANNEL arms mirror the d-major ones expression for expression; a
u4 lane owns one byte column (both of its channels), so no byte is written by two lanes, and the relaid-out u4 V page
is copied as rows of `PACKED_V_HEAD_SIZE` plus the per-token comp arrays. `smoke_pa_kv_reorder_layout` reorders the same
random cache in both layouts (f16, i8 BY_TOKEN, i8 and u4 BY_CHANNEL; static and dynamic index inputs) and compares codes
and comp bit for bit; breaking the i8 or u4 token-major arm fails only those cases, and pinning the dynamic impl to the
d-major kernel fails only the dynamic ones. `smoke_qq_bias_d_major` keeps the `pa_single_token` / `pa_multi_token`
qq_bias paths covered on a forced d-major page.

A page is a data region followed by a comp region holding the dequantisation parameters. The
comp region grows whichever of the two page factors it is indexed by, so the host jits
`ADJUSTED_K_HEAD_SIZE` and `ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE` and every layout is their product
(`h` = head size, block size 16):

| Cache | Data row pitch (bytes) | Comp region | K page stride (bytes) |
|---|---|---|---|
| f16 | `2h` | none | `32h` |
| i8 BY_TOKEN | `h` | per-token f16 scale at `[t]`, zp at `[16 + t]` | `16 * (h + 4)` |
| i8 BY_CHANNEL (token-major) | `h` | per-channel interleaved (scale, zp) f16 pairs, one dword at `[c]` | `20h` |
| u4 BY_CHANNEL K (token-major) | `h / 2` | as i8 BY_CHANNEL | `12h` |
| u4 BY_TOKEN V | `Align(h / 2, 16)` | as i8 BY_TOKEN | `16 * (Align(h / 2, 16) + 4)` |

The comp region always starts right after the data region, at `PA_K_ROW_ELEMS * block_size` /
`PA_V_ROW_ELEMS * block_size` (`PA_K_COMP_OFF` / `PA_V_COMP_OFF`), the same offsets
`pa_kv_cache_update_ref.cl` writes. The u4 K row is `h / 2` exactly and deliberately
not aligned up: `16 * (h / 2)` data bytes plus `4 * h` comp bytes is `12h`, which is what makes the
token-major page a byte-exact fit into the allocation the upstream d-major INT4 page already has.
Aligning would overflow it whenever `h % 32 != 0`. The u4 V row can be aligned for free, because
the trailing comp slack absorbs it (`16 * PV + 64 == 16 * (PV + 4)`).

How the host classifies the cache MIXED reads (`classify_pa_cache()`):

- `IS_PA_KV_COMPRESSED` follows the cache data type (i8/u8), not the quantization mode. Paged
  attention adds the PREFILL and MIXED stages together, so a MIXED kernel that fails to compile
  takes a PREFILL-only case down with it. The kernel reads the token-major page only (f16, i8
  BY_TOKEN, i8 and u4 BY_CHANNEL); every d-major page compiles as that reader (the BY_CHANNEL
  flags stay off there) and is kept off dispatch by `can_use_sdpa_ocl_for()`.
- u4 is recognised from the configured kv-cache precision: the tensor is u8 (an i4 cache is i8),
  so its layout type cannot tell. u4 only, not i4: the int4 quantizer clamps to [0, 15], so u4
  nibbles are unsigned, while i4 would need a signed widen that nothing implements.
- Whether BY_CHANNEL was relayed token-major is read from the physical cache shape (the adjusted
  block size sits at dim 2), the single decision made in `transformations_pipeline.cpp`.
- BY_CHANNEL's per-channel scale and zero point fold as a per-lane scalar. The KQ A operand is K,
  so the lane index is the head dim, which is what BY_CHANNEL indexes; BY_TOKEN's are per key,
  i.e. per element within a lane, and need a broadcast each. This is the mirror of
  `sdpa_ocl_decode`, where Q is the A operand and the fold lands on Q.
- No layout flag is jitted: `PA_K_TOKEN_STRIDE` / `PA_K_HIDDEN_STRIDE`, the addressing of the gather
  fallbacks, are the token-major strides (`IS_PA_K_TOKEN_MAJOR` selected the d-major ones until 2026-11-02).

### u4 K: the permuted depth axis

The KQ A operand is K itself, so `lane == head dim` (the DPAS depth index) — the mirror of
`sdpa_ocl_decode.cl`, where K is the B operand and `lane == token`. The K page uses the upstream
adjacent nibble order (byte `b` holds channel `2b` in its low nibble and `2b + 1` in its high one).
The writer needs that order: `NUM_K_HEAD_SIZE_PARTITIONS` splits the channel range across
workgroups, and a split-at-`h/2` convention would put a byte's two channels in different
workgroups.

A byte column is therefore a channel pair, and the 8-bit VNNI-transform read (whose lane is the
byte column) hands lane `L` channels `(2L, 2L + 1)` of a window, never the contiguous `base + L`
a DPAS tile wants. No lane-local rearrangement can fix that. Depth is a contraction axis, though:
permuting it identically in A and B leaves `sum_d K[key][d] * Q[query][d]` unchanged. So both
operands adopt this labelling, in which tile `db` of the pair covering the 32-channel window
`win = (db >> 1) * 32` owns the even channels for even `db` and the odd ones for odd `db`:

```
PA_K_U4_CHANNEL(db, L) = win + 2L + (db & 1)        byte = win/2 + L,  nibble = db & 1
```

Every consumer falls out cheaply:

- the 2D read at byte column `win / 2` lands exactly this, one read per tile pair;
- the per-lane fallback addresses byte `win / 2 + L`, lane-contiguous (better coalesced than the
  natural order), with a lane-uniform nibble select;
- the per-channel comp is one `vload2` per window, serving both tiles of the pair;
- Q pays the whole cost once per workgroup in the SLM staging: a chunk spans 32 consecutive
  channels, read as two windows, and each output dword takes one half of two input dwords
  (three operations per dword). Q's staging traffic doubles, which is negligible next to K/V.

Until 2026-11-02 the f16 d-major page read landed the same labelling from a 32-bit transpose read ("Paged-attention
MIXED: i8 and f16 caches") and shared the Q staging and the Kc reads; hence the name `PA_K_PERMUTED`, now `IS_PA_K_U4`.

V keeps the split convention instead (byte `b` holds dims `b` and `b + PA_V_ROW_ELEMS`), because V
is the S*V B operand where `lane == head dim` as well: with adjacent packing a lane would own two
different dims, which no DPAS N axis can express. A read at the folded byte column hands lane `c`
head dim `base + c` from either nibble, so the S*V tile loop is unchanged except for the column
fold and a nibble select.

### u4 1D page read

A u4 row is `h / 2` (K) or `Align(h / 2, 16)` (V) bytes, 32 at head 64, below the 64-byte block2d
minimum for both tensors. Without another path every element was fetched by a per-lane byte gather:
on the gpt-oss-20b MIXED kernel that was 64 K and 128 V scattered messages per k0 iteration against
16 DPAS, plus 5952 bytes of spill from the address arithmetic.

The page's data region is one contiguous run of `16 * ROW` bytes, and
`intel_sub_group_block_read_uc16` lands component `i` of lane `L` on byte `SUBGROUP_SIZE * i + L`.
Both consumers want byte (token `t`, 16-byte column group `c`) `+ L`, with `COLS = ROW / 16`:

```
SUBGROUP_SIZE * (16 * r + i) + L  ==  t * ROW + SUBGROUP_SIZE * c + L
                              <=>  16 * r + i == t * COLS + c
```

so `PA_PAGE_R` / `PA_PAGE_I` place any `(t, c)` with no shuffle and no per-lane address, and
`COLS` reads cover a whole column group. `r` is a compile-time constant whenever `t` is, even if
`c` is not, because `c < COLS` and `COLS` divides 16 (the host requires a power of two). `i` is
not, so a caller with a runtime `c` (V, whose column group comes from `sg_j0_sv`) biases the base
by `SUBGROUP_SIZE * c` and passes `c = 0`, which keeps the `uchar16` component select a register
subscript. K's `c` is `db >> 1`, a constant, so K reads at the plain page base and hoists the reads
out of the depth loop.

The last byte touched is `SUBGROUP_SIZE * c + COLS * 256 - 1`: 527 bytes for the head-64 u4 V page
(`COLS = 2`, `c <= 1`) against a 576-byte page, and 511 bytes for K against 768. The overhang past
the data region only lands in the page's own comp arrays, and only unused components read it.

The host enables it only where block2d is off (in practice u4 head 32 and 64, plus V alone at head
48, whose `Align(24, 16) = 32`-byte row qualifies while the 24-byte K row does not), and it was
measured at 3.92x on gpt-oss-20b.

### Block2d rules

The 2D block builtins need a surface width of at least 64 bytes and a multiple of 4, a pitch of at
least 64 bytes and a multiple of 16, and a 64-byte aligned base.

- Plain SDPA and the PA current-token inputs: `block2d_layout_ok()` (`row_bytes >= 64 &&
  row_bytes % 64 == 0`) needs no base repair; `block2d_layout_fixup_ok()` (`row_bytes >= 64 &&
  row_bytes % 16 == 0`) relies on `BLOCK2D_KV_BASE_FIXUP` / `BLOCK2D_KV_CUR_BASE_FIXUP`. The fixup tier is
  what lets a Q/K/V that is a crop view of a fused QKV tensor (phi-4-multimodal's vision tower) use
  block IO; without it `sdpa_ocl` fell back to the scalar gather and ran 7.5x slower than
  `sdpa_micro` there. Both tiers look at the padding of the layout:
  - A rank-4 tensor keeps its pitch and base offsets a whole number of rows as long as the innermost
    axis (X) is unpadded, so that is the test. Any other padded rank is refused (X does not exist at
    rank 3). The strict tier also assumes a 64 B aligned buffer, which holds for engine allocations and
    for in-place crop views of them.
  - A paged-attention Q/K/V is a rank-2 token matrix `[tokens, heads * head_size]`. The head dimension
    lives inside FEATURE, so feature padding moves the base (by the padding before) and the token
    stride of every head by an arbitrary amount, and the X test is vacuously true. With static padding
    the strict tier needs the padding before and the token stride to be multiples of 64 B, the fixup
    tier multiples of 16 B. A dynamic padding cannot be proven: the strict tier refuses it (Q and A,
    which have no base repair, fall back to the scalar path; K/V, Kc/Vc take the fixup tier with
    `BLOCK2D_KV_BASE_FIXUP`), and the fixup tier takes it on trust. That assumes the token stride stays a
    multiple of 16 B and the start of the first head a multiple of 4 B (a crop view of a fused QKV
    tensor at head-size offsets does); a dynamic K/V padding that breaks it reads wrong values.
  - Measured on Arc Pro B70 with `paged_attention_feature_pad_test`: a token stride that is not a
    multiple of 16 B (260 B, 264 B) and a base that is 2 B off give wrong results; a base off by 4, 16,
    32 or 48 B with a 16 B-multiple stride reads correctly even without the repair. The 64 B base
    rule is therefore the documented one, not what this device enforces, and only the host test
    `sdpa_block2d_gate` guards it. With one KV head the widened fixup surface (row plus up to 48 B)
    can exceed the pitch, which the spec leaves undefined; the B70 reads it correctly
    (`paged_attention_feature_pad_test`, 4 heads / 1 KV head).
- Cache pages: `block2d_page_ok()` (`row_bytes >= 64 && row_bytes % 16 == 0`). A page base is
  always a whole number of pages, and in each layout above the pitch rule already makes the page
  stride a multiple of 64 (f16 `32h` with `h % 8`; i8 BY_TOKEN `256n + 64`; i8 BY_CHANNEL `320n`;
  u4 K `384n`; u4 V `256k + 64`).
- u4 stays on the strict `% 64` rule on purpose: the 1D page read is gated on block2d being off,
  and relaxing the rule would displace it on head sizes that have no model to measure.
- Q additionally needs a subgroup of 16, the transpose read's 16-row geometry. Head sizes that are
  not a multiple of `DPAS_K`, and the `D_MAX > d` tail, are guarded in the kernel, so there is no
  head-size whitelist.
- The fixup flags come from the strict test, evaluated after the `SDPA_OCL_*_2D` override: a
  surface that is not provably aligned gets the fixup whenever the block path is on. That is a
  no-op on an aligned base, and it keeps a forced `SDPA_OCL_KV_2D=1` correct; before it, forcing the
  path at head 72 gave the right timing and wrong results for 12 of 16 heads.
- `BLOCK2D_KV_CUR_BASE_FIXUP` is forced on for u4 even on the aligned tier: the u4 Kc dword read
  needs a 64-byte aligned origin and a meaningful `KcD_x0` for its parity test, and the fixup is a
  no-op on an aligned base.
- The 8-bit VNNI-transform read has a 32-row minimum on Xe2 (there is no 16-row variant), while a
  page holds 16 tokens, so the page reads clamp the surface height to the page and consume only the
  first 16 rows. Unlike the plain-SDPA i8 path they cannot pair two key groups into one read:
  consecutive key groups live in different, non-adjacent pages.
- `sdpa_ocl_decode` checks its pages with the strict `% 64` rule (`block2d_surface_ok()`), which
  narrows by itself with the cache: f16 needs `h % 32 == 0`, i8 `h % 64`, u4 K `h % 128`.

`BLOCK2D_KV_BASE_FIXUP` repairs the base the way `sdpa_micro`'s `block2d_load` does: round it down
to 64 bytes, shift x and widen the surface by the same number of bytes. The per-head offset is a
multiple of the row width but not of 64 (a 144-byte row at head 72 lands 16, 32 or 48 bytes past a
boundary for `head % 4 != 0`). The widening extends backwards, so the surface still ends at the
head's last element: columns past the head dim stay out of bounds and are zero-filled by the
hardware, which is what the scalar path's `head < d` guard did. The shift always divides exactly:
the offset is a whole number of elements, so `prem` is a multiple of the element size (all the 16b
builtins need); `prem % 16 == 0`, which a 32-bit transposed read would need, follows from
`base = m * row_bytes` and the host's `row_bytes % 16 == 0` gate.

### The u4 token-major K writer (IGC 2.41)

With IGC 2.41.5 the u4 BY_CHANNEL token-major writer in `pa_kv_cache_update_ref.cl` reported `TPM=576` at every
head size and its PREFILL dispatch became 1.3-1.6x master's d-major writer. IGC 2.41 no longer promotes a private array
indexed by a runtime value, even a subgroup-uniform one, to registers: `token_vals[2][9]` (requantize helper) and
`input_data[2][9]` (prefill helper) went to scratch (2 x 9 halves x 16 lanes = 576 B). 256 GRF did not change it.
Both helpers now index those arrays only with unrolled constants: the loops run over all 8 token pairs under a uniform
`it < n` guard, the PREFILL odd tail has its own variable, and a value whose slot is known only at run time (the
leftover old token, a new token, the requantize odd tail) is written or read through a select over every slot.
Storing the slots as a `half16` with a runtime subscript also avoids scratch, but its indirect register access made
GENERATE 3-15% slower than before (standalone probe).

In GENERATE the head-256 shape (16 partitions, each owning 8 bytes of the same 128-byte token rows) then queued the
partitions' stores on the same lines (the old code's stores were spaced out by its scratch reloads). Under
`NUM_HEAD_SIZE_GROUPS == NUM_K_HEAD_SIZE_PARTITIONS && BC_TOKEN_STRIDE >= 128` each partition starts its stores at
pair `partition % 8`, reading the rotated byte from a `uchar8`. It does not pay at heads 128 / 512 (+3-8% / +5% in
the standalone probe), so they keep the plain loop. Cache bytes are unchanged: the 186 basic paged-attention rows
dump identical output, K cache and V cache bytes before and after (the gtest tolerance alone cannot show this, see
"Test coverage gaps").

Per call (us, B70 at 1800 MHz, cliloader; prefill `-ic 1`, generate 2K prompt + 256 tokens), master (d-major) / before
/ after:

| | PREFILL 8K | PREFILL 32K | GENERATE |
|---|---|---|---|
| qwen3-8b (head 128, 8 kv heads) | 224 / 322 / 172 | 778 / 1219 / 570 | 8.3 / 8.9 / 8.1 |
| llama-3.2-1b (head 64, 8 kv heads) | 121 / 156 / 82 | 437 / 621 / 318 | 7.7 / 8.4 / 7.6 |
| gemma-4-26b head 256, 8 kv heads | 400 / 592 / 274 | 1483 / 2392 / 1032 | 9.2 / 10.1 / 9.5 |
| gemma-4-26b head 512, 2 kv heads | 270 / 354 / 133 | 932 / 1279 / 574 | 11.9 / 11.9 / 11.9 |

First-token latency after / before (llm_bench, 2 interleaved rounds, median of 6): qwen3-8b 0.995 / 0.997 at 8K / 32K
(-5 / -23 ms), llama-3.2-1b 1.000 / 0.997, gemma-4-26b 0.993 / 0.992 (-9 / -38 ms).

## Dequantisation

Paged-attention caches: `(q - zp) * scale`, in half. The cache writer stores `1 / scale`, so the
value read back is already the multiplier. The block-read and scalar paths use exactly the same
arithmetic, which makes the `*_PA_I8_2D` and `*_PA_1D` switches clean bisection toggles. The u4 block
reads form `q` as `as_half(0x6400 | n) - 1024` (`U4_TO_HALF`), which equals `(half)n` exactly
("Paged-attention MIXED tiling"), and the i8 block reads as `as_half(0x6480 ^ b) - 1152` (`I8_TO_HALF`), which equals
`(half)(char)b` exactly ("Paged-attention MIXED: i8 and f16 caches").

Plain-SDPA i8 (the bias trick): `as_half(0x6480 ^ byte) == signed_byte + 1152` exactly, since
`0x6480` is `1152.0h` and XOR-ing the byte into its mantissa maps the two's-complement range onto
consecutive halves. With `zp + 1152` folded into the zero point, a byte dequantises as
`(as_half(0x6480 ^ byte) - (zp + 1152)) * scale` with no convert: a microbenchmark measured 69%
fewer moves than the `convert_float` widen for K and 66% fewer than `convert_half4` for V. Extracting
bytes with shift and mask also avoids the `:b` region deinterleave that `as_char4` costs. The fold
has a price: f16 has a 1.0 ulp at 1152, so a non-integer zp is rounded. The paged-attention paths
deliberately do not use the trick, since the cache writer's zp is not an integer.

The per-key scale and zero point are hoisted out of the depth and key loops. Read from the
innermost position they are lane-uniform, and IGC emits one SIMD-1 load each: about 128 per k0
iteration for the plain i8 path, and 256 per k0 tile for the paged i8 cache at head 128 (the
MIXED kernel's load count was dominated by scale/zp: 272 `d16u32` messages against 128 `d8u32`
data loads). The page lookup (`block_indices[]`) is hoisted the same way, from
`DKS * kq_key_blocks * DPAS_ROWS` loads (64 at head 64) to `kq_key_blocks`.

The plain-SDPA scales have the query's type (`kv_cache_compression.cpp`), so a bf16 query brings bf16
scales, jitted as `ushort` (their bits). `KEY_SCALES_IS_BF16` / `VAL_SCALES_IS_BF16` make every scale
read widen them by a 16-bit shift (`K_COMP_SCALE_TO_HALF`, `V_COMP_SCALE_TO_HALF` / `_TO_FLOAT` in
`sdpa_ocl_config.cl`); before that the integer was converted and every bf16 compressed output was
garbage, on SG8 and SG16 alike (upstream #38532 made the path reachable and fixed `sdpa_opt` /
`sdpa_ref` only). The per-key scale lanes stay half, which holds a bf16 scale exactly for
6.1e-5 <= |scale| <= 65504; a smaller scale is rounded to a half subnormal, a larger one overflows.
`smoke_sdpa_kv_compressed_bf16` covers it.

## Masks

- `MASK_KIND` is the host's compile-time proof of the mask shape: 2 = full 2D, 1 = per key,
  0 = scalar/broadcast, -1 = decide at runtime from `MSK_D2`/`MSK_D3`. For a dynamic mask the host
  infers the kind from the stage, so a `[B, H, 1, K]` per-key mask can be compiled as kind 2. The
  full-2D path therefore clamps the query row to 0 when `MSK_D2 == 1`, and the key column when
  `MSK_D3 == 1`. Without the clamp the read walks past the single row, which on Xe2 surfaces as
  `CL_OUT_OF_RESOURCES` or a NaN mask value.
- `HAS_SCALAR_ATTN_MASK` / `STATIC_SCALAR_ATTN_MASK_VALUE`: a single-element mask broadcast to
  every logit.
- `HAS_QQ_BIAS` (MIXED only): the speculative-decoding tree mask over the new keys.

### Bidirectional image groups

For models with `token_type_ids` (gemma-4 and friends) a maximal run of `token_type_ids[t] == 1`
is an image group whose members attend to each other in both directions, on top of the causal and
sliding-window region (`openvino/reference/paged_attention.hpp`).

- `token_type_ids` covers the new tokens only, so it is indexed in LOCAL coordinates
  (subsequence-relative, `[0, q)`), while keys and `causal_k` / `window_k_begin` are KEY coordinates
  (`key = query_position_offset + local`). PREFILL is the `past_len == 0` case of the same code.
- The causal bound and the window start are extended to cover the groups that straddle them.
  Groups never leave the new-token region, and reading a future key from the cache relies on
  `pa_kv_cache_update` running before this stage, as the reference does.
- The mask loop needs only per-query group bounds `[begin, end)`, computed once per workgroup
  (the removed `sdpa_micro` read per-token bounds from a separate `pa_tt_bounds` stage; its kernel,
  `TT_BOUNDS_STAGE` in the upstream `paged_attention_opt.cl`, is no longer compiled).
- A small GQA MIXED step (at most 16 new tokens per subsequence, e.g. a speculative draft block)
  takes the GQA multi-token path as without token_type_ids. `pa_small_q_opt.cl` has no bidirectional
  mask, so that is the `paged_attention_opt.cl` fallback, which has one; on a token-major page where
  `sdpa_ocl` serves MIXED, an f16 cache above 12 new tokens goes to `sdpa_ocl` instead ("Small GQA MIXED
  step on the token-major page (2026-11-08)"). With token_type_ids on that page the fallback took
  0.11-0.97x `sdpa_ocl`'s time with the u4 and i8 BY_CHANNEL caches at 2-16 new tokens and 512-16K
  context, and 0.15-1.02x with f16 and i8 BY_TOKEN up to 12 new tokens (32/8 heads, head 128; f16 at
  16 new tokens up to 1.34x). Giving `pa_small_q_opt.cl` the mask was measured
  through its time without token_type_ids (the mask costs the fallback under 2%) and not done
  (2026-10-30): with the default BY_CHANNEL caches it would take 0.27-0.51x `sdpa_ocl`'s time from 4K
  but 1.13-1.74x at 1K, and with f16 or i8 BY_TOKEN 1.3-4.8x the fallback's at 4 new tokens.
- A runtime-empty `token_type_ids` (`[B_token | 0]`) is legal while `HAS_TOKEN_TYPE_IDS` is decided
  at compile time from a possibly dynamic shape, so every read is gated on the runtime count. The
  paged-attention impl is compiled once from dynamic params (`shape_types::any`) and afterwards only
  refreshes its dispatch data, so the count reaches the kernel as a scalar. A non-empty buffer
  shorter than `B_token` is a contract violation rather than "no image tokens" and is refused, as
  `intel_cpu` does.
- GENERATE never needs any of this: one new token per subsequence makes every group the query
  itself.

### Attention sink

The sink is an extra per-head logit whose value vector is zero, so it only joins the softmax max
and denominator. `sdpa_ocl` seeds the online-softmax state with it (running max = sink, running sum
= 1 on one subgroup, output = 0), which leaves the key loop untouched. `sdpa_micro` instead injects
it per k0 tile from the subgroup owning the last key and has to pre-scale the score tile.
`sdpa_ocl_decode` adds it in partition 0 only, because the finalization rescales and sums the
per-partition denominators.

## sdpa_ocl_decode

Decode has one query per sequence, which flips the DPAS operand roles: A = Q (M = `Q_PER_WG`
heads), B = K, so the scores land one per lane with `lane == key` and the softmax is a pair of
subgroup reductions instead of an SLM tile plus an alpha rescale. Almost nothing is shared with
`sdpa_ocl.cl`.

- GQA: `Q_PER_WG` q-heads of one kv group share every K/V tile load. `pa_gqa_single_token` does the
  same amortisation with scalar MADs, but its candidates are {4, 3, 2}; DPAS carries M in the repeat
  count, so M = 8 costs the same MACs per cycle as M = 1. The host caps M by a live-register
  estimate: on gemma-4's head-512 layers M = 8 spilled 34432 bytes and ran 3.14x slower than
  `pa_sdpa_opt`, while M = 1 spilled nothing. The response was monotone in M at every `SG_PER_WG`
  tested, and "the largest M that does not spill" reproduced the measured optimum. M rounds the kv
  group up to a power of two (kv group 3 -> M = 4): the surplus row is masked like a leftover and never
  stored, so a group that is not a power of two takes one workgroup per kv head instead of two and
  reads each K/V page once. phi-4-mini (24/8) at 4K, against M = 2 with the same `SG_PER_WG`: i8 0.90x,
  u4 0.70x of the GENERATE device time, with identical generated tokens.
- `live_grf_estimate()` counts the arrays that stay live across the KQ loop. It was calibrated on
  gemma-4 (u4 BY_CHANNEL, M in {1, 2, 4, 8} x `SG_PER_WG` in {8, 16}, head 512 and 256) and tracks
  spill volume monotonically (head 512 at `SG_PER_WG` 8 scores 136/164/220/332 against 0/6400/15936/
  34432 bytes), but it cannot resolve differences under about 15 GRF: (8, M = 1) scores 136 and does
  not spill, (16, M = 2) scores 126 and spills 640. So it is a coarse gate, and the budget of 112
  brackets the measured optimum instead of sitting at 128: head 512 at `SG_PER_WG` 16 wants M = 1
  (100) and rejects M = 2 (126, 34% slower), head 256 wants M = 2 (94), and llama's head 128 at
  M = 4 scores 104 with an i8 cache. A u4 cache reads four K tiles at a time and scores 124 there, so 8
  subgroups drop to M = 2 although M = 4 did not spill on B70 (16 subgroups score 106). Changing
  `SG_PER_WG` therefore changes M too; read both back from the dispatch (`LWS` dim 1 is `SG_PER_WG`,
  `GWS` dim 1 is `SG_PER_WG` x head groups) before attributing a timing to either. `SDPA_OCL_DECODE_M`
  bypasses the cap but not the local-memory clamp (half the arena), which guards the build.
- When that cap holds M below the (rounded-up) kv group, the host switches to a low-register layout and
  keeps the whole group. With M below the group, every K/V page is read and dequantized group / M times per
  kv head, and decode is bandwidth bound, so the cost of the cap grew with the context: gemma-4-26b's
  head-512 layers ran M = 1 at kv group 8, and its 2nd-token latency fell behind `pa_gqa_single_token`
  (`HEADS_PER_WI` 4) from 4K on, by 8% at 32K and 12% at 128K. The layout only moves data, so the output is
  bit-identical to the plain one at any M (each head is its own DPAS row):
  - `Q_IN_SLM`: the scaled query sits in local memory, one `Q_PER_WG`-wide vector per (tile, lane), and
    each A operand is read back where it is used; `q_reg[M][K_TILES]` alone was the whole 128-GRF file at
    head 512 and M = 8.
  - `K_COMP_PER_READ` (BY_CHANNEL): the scale/zp pairs of one K read's tiles are loaded with that read,
    and `k_corr` accumulates inside the KQ loop in the same tile order as the up-front pass.
  - `V_COMP_IN_SLM` (compressed): the subgroup that scored a page loads that page's V scale/zp once,
    folds the scale into the probabilities it writes to `slm_p` (the same f16 multiply) and stages the zp
    there. Before, every S*V subgroup loaded every page's pair inside the loop and waited for it (the V
    prefetch covers only the data rows).
  - `V_TWIN_PAIR` (u4, V block reads, both twins in one subgroup): tiles `cd` and `cd + V_READS` share
    one V read; one nibble each.

  `low_reg_grf_estimate()` has no `K_TILES` term left, and its budget (160) brackets the measured
  boundary: 148 (u4 head 128, 8 subgroups, M = 8) and 138 (head 512, 16, M = 8, twins) spill nothing,
  184 (4 subgroups) spills 896 bytes. It is not used at head 64 once the plain M is 4: there a page is
  read at most twice, and halving the cheap workgroups' count lost up to 2K keys (i8 gpt-oss and
  minicpm4-0.5b, 1.06-1.15x). Kernel time against the plain layout (B70 at 1800 MHz, standalone probe
  with the L2 flushed, 300 to 33K keys): gemma-4 head 512 0.23-0.85x (M 1 -> 8), gemma-4-e2b SWA head 256
  0.92x (2 -> 8), minicpm4-8b 0.63-0.97x (4 -> 8), qwen3.5-9b 0.60-0.82x (2 -> 4), qwen3.6-35b-a3b
  0.46-0.91x and qwen3.8-27b 0.55-0.90x (2 -> 8), and with an i8 cache 0.58-0.94x. On gemma-4-26b at
  32K the head-512 call went from 712.8 to 161.5 us (`pa_gqa_single_token`: 441.9 us) and the decode
  device time per token from 13.02 to 10.20 ms (master: 11.81 ms; cliloader); its 2nd-token latency against
  master went from 0.99-1.13x, rising with the prompt from 278 to 133K tokens, to 0.84-0.99x (llm_bench with
  the daily prompts, 1800 MHz). Shapes the plain layout already holds at the full group (llama, qwen3-8b,
  phi, mistral, gpt-oss with u4, gemma-2/3, the gemma-4-26b SWA layers) compile to the identical ISA.
- `SG_PER_WG` is 16 when there are at least 16 V head-dim tiles, and for a u4 cache at head 64 or 128
  when 16 subgroups still get M >= 4; 8 otherwise. One subgroup is one thread, and at 8 the kernel had a
  quarter of `pa_sdpa_opt`'s threads per Xe core: on gemma-4 at M = 1, 16 was 2.07x faster at head 512
  and 1.26x at head 256, and 4 was 2.2x slower. With fewer V tiles the surplus subgroups split the key
  axis instead, which adds the `slm_out` staging and a barrier, and 16 doubles the threads a partition
  needs, which costs a wave once the GPU is full. So 16 pays only where a workgroup is slow (the u4
  dequant; u4 is the default cache of a paged-attention model with 4-bit weights) and only at M >= 4,
  where 16 subgroups need no more threads per partition than 8 subgroups at M = 2 (the u4 head-128
  default before, as its register estimate rejects M = 4 at 8). GENERATE device time on B70 against the
  previous choice: u4 head 64 (lfm2.5, llama-3.2-1b, minicpm4) 0.45-0.51x at 1-6 partitions and 0.83x at
  17-18; u4 head 128 llama-3.1-8b 0.82x/0.79x at 1K/4K, qwen3-8b 0.77x at 8K, phi-4-mini 0.82x at 1K and
  4K (with the rounded-up M). i8 loses with 16 once the GPU is full (head 64 1.26x and head 128 1.29x at
  4K, against 0.85-0.89x at 1K), so they keep 8. Head 96 MHA (phi-3.5) takes 16 since 2026-11-06: six
  tiles split unevenly over 8 subgroups, so S*V reduces over key slots at 8 as well, and a one-head
  workgroup is the cheapest there is. With the per-lane page reads of "Compressed pages block2d does not
  take", 16 measured 0.85-0.98x the time of 8 from 700 to 13K keys for u4, i8 and f16 (probe), 1.06-1.09x
  at 1.1K (5 partitions: 2560 threads, just over one wave of B70's 2048) and parity past 18K; with the
  old byte gather it had measured mixed. `SG_PER_WG` must divide `SEQ_LEN_PARTITION_SIZE /
  SUBGROUP_SIZE` and not exceed the subgroup size.
- The cache is f16, i8 or u4. i8 only among the 8-bit types: the dequant widens the byte as signed,
  as `kv_cache_update` wrote it, so a u8 cache would decode with the wrong sign. i4 is rejected. qq_bias
  is accepted: one new token per sequence makes the tree mask the 1 x 1 identity, which plain causal
  masking already gives.
- `SDPA_OCL_DECODE_256GRF=1` is for the larger M values: at M = 8 and head 128 the live set is about
  110 GRF (the S*V accumulator alone is `V_TILES` x float8 = 64). Read SPILL= from a runtime
  cliloader line; ocloc mispredicts it.
- The key axis is split into partitions of `SEQ_LEN_PARTITION_SIZE`, and the kernel writes
  `pa_sdpa_finalization_stage`'s intermediates unchanged (see the contract at the top of the file).
- S*V splits the head dim, not the keys (except `V_PAGE_1D`, below): a subgroup that owns a head-dim
  tile reduces over every key and produces a final result. The earlier key split left every subgroup holding a partial over
  all dims and cost a `SG_PER_WG * Q_PER_WG * V_HEAD_SIZE` SLM round trip — 16 KB at head 128, which
  capped Xe-core occupancy at 7 workgroups instead of 8. The DPAS and V-load counts are the same
  either way (16 per subgroup at head 128).
- The probabilities go through SLM indexed by key with the heads as the vector element. Storing
  them head-major instead cost 64 separate 32-byte reads per subgroup (SLM loads 44 -> 72,
  instCount +12%).
- The partition's page table is read once, one chunk per lane, and broadcast where needed; a
  per-use lookup cost 18 separate scalar loads.

Compressed caches factor the affine dequant off the B operands:

```
BY_TOKEN K:    S[key] = sc[key] * (sum_d Q[d] * q[key][d] - zp[key] * sum_d Q[d])
BY_CHANNEL K:  S[key] = sum_d (Q[d] * sc[d]) * q[key][d] - sum_d (Q[d] * sc[d]) * zp[d]
V:             O[d]   = sum_key (P[key] * sc[key]) * (q[key][d] - zp[key])
```

The int8 B operand is widened with `as_half(0x6480 ^ b) == b + 1152` as pure dword arithmetic.
Xe2 has no direct byte-to-half convert, so `convert_half16(as_char16(...))` costs three moves per
element (a stride-4 byte gather, b->w, w->hf): instCount 4431 / 2852 mov against 4045 / 2022 for
the dword widen at head 128, M = 4. The bias is removed with the zero point in the float score
correction, so zp stays exact. u4 uses `as_half(0x6400 | n) == 1024 + n`.

The BY_CHANNEL correction `k_corr` must multiply by the f16-rounded `Q * sc` the DPAS sees, not
the float product: the bias cancels exactly only against the same half. Computing it in float
raised the worst-case error over the BY_CHANNEL cases from 7.21e-04 to 1.68e-03 (2.3x, consistent
with the bias-to-signal ratio times a half ulp). That is below a 6e-3 accuracy threshold, so a
passing test does not protect it.

V is prefetched `PREFETCH_DIST` chunks ahead with the 2D block prefetch. The limit was
memory-level parallelism: doubling K/V traffic cost 23%, while cutting instructions 6.8% and SLM
traffic 86% bought only 1.2%, and every occupancy knob (`SG_PER_WG` 2/4/16, 256 GRF) was neutral
or worse. Each prefetch costs about 8 instructions of address and descriptor setup. On
llama-3.1-8b (head 128, M = 4) prefetching V was 2.1% faster, because S*V walks 16 different pages
on one accumulator chain. Prefetching K was 3.6% slower, because the KQ loop already runs
`KEY_GROUPS` independent DPAS chains over one page, so K is not prefetched. The distance only
moves where the prefetches are issued, never how many: `min(PREFETCH_DIST, chunks)` go in the
pre-S*V barrier window and the rest one iteration group ahead. Against 1.1194e9 ns with prefetch
off, distance 1, 2 and 4 measured 1.1031e9, 1.0984e9 and 1.0958e9 ns, so the barrier window is the
better place for them; the prefetches issued in the barrier wait account for 0.7 points of the
2.1% win at distance 4.

### Compressed pages block2d does not take (2026-11-06)

A u4 row is `h / 2` (K) or `Align(h / 2, 16)` (V) bytes: 32 at head 64 and 48 at head 96, below the
64-byte block2d minimum, and the i8 row of head 96 (96 bytes) fails the strict `% 64` rule. Those pages
took the per-lane fallback, which is cheap for K (IGC merged the byte `vload8` into dword loads) but
not for V: one byte gather per (token, lane), 16 messages per (tile, chunk), 128 `load.ugm.d8u32` per
tile in the S*V loop of phi-3.5-mini (u4 and i8 head 96) against 8 block reads with block2d. This is
what made `paged_attention_opt.cl` faster there ("Performance opportunities", the record-only
comparison).

- `V_PAGE_1D`: S*V splits the keys only (`SV_DIM_SGS` 1), so every subgroup owns every head-dim tile
  of its chunks and reads the whole data region of each page with `V_PAGE_COLS = ROW / 16`
  `intel_sub_group_block_read_uc16` reads -- the mapping of "u4 1D page read", with `t` and `c` both
  compile-time, so nothing is read twice and both nibbles of a u4 byte feed their two tiles. The V
  scale and zero point are loaded, folded and broadcast once per chunk instead of once per (tile,
  chunk); that is why the 1D read also beat the V block read on i8 head 96 (2D K + 1D V 0.76-0.90x the
  time of 2D K + 2D V with the head-dim split, probe). The partials of the `SG_PER_WG` key slots are summed per (tile, head
  row), so all subgroups take part when `V_TILES * Q_PER_WG >= SG_PER_WG` (head 64 has 4 tiles and 16
  subgroups); same order and multiply as the per-tile loop, so bit-identical to it. The key split
  changes the summation order against the head-dim split (output bits differ, the probe's error
  against the reference did not move on any of the 35 changed cases). The host takes it only where
  its partial-output staging (`SG_PER_WG * Q_PER_WG * V_HEAD_SIZE` floats) does not cost M through
  the local-memory clamp.
- `K_LANE_TILES`: the per-lane K fallback of a compressed cache loads one dword vector per lane for
  4 (u4) or 2 (i8) tiles, a divisor of `K_TILES`, the 32 bytes a transposed block read hands each
  lane. Bit-identical (408 of 408 probe cases). On i8 head 96 it measured the same as the block read
  the relaxed `% 16` page rule would allow (52.2 / 86.7 / 249.9 against 52.3 / 86.6 / 249.7 us at
  2.3K / 4.5K / 18K keys), so `sdpa_ocl_decode` keeps the strict rule.
- Head 96 MHA takes 16 subgroups (the `SG_PER_WG` bullet above).

Probe (B70 1800 MHz, L2 flushed, median of 60 launches), HEAD -> this change: phi-3.5-mini (32 / 32,
head 96) u4 58.2 / 93.8 / 257 -> 30.6 / 52.5 / 157 us and i8 BY_CHANNEL 74.2 / 117.5 / 316 -> 47.7 /
82.0 / 247 us at 2.3K / 4.5K / 18K keys; llama-3.2-1b (32 / 8, head 64, u4) 40.8 / 62.6 / 117.6 ->
25.2 / 38.5 / 69.5 us at 8.2K / 16K / 32.6K; gpt-oss-20b (64 / 8, head 64, u4) 51.5 / 146 -> 36.6 /
102 us at 8.4K / 33.4K. Over every per-lane shape the host generates (u4 / i8 head 16..192, kv group
1..8, 600 keys) no shape got slower (0.12-1.00x); spill fell where the old layout spilled (head 160 /
192 at M = 8: 18.5-20 KB -> 0.4-3.5 KB) and appeared only on u4 head 192 at M = 4 (1.25 KB, 0.29x the
time). Block2d shapes compile to the identical kernel (ISA compared: llama-3.1-8b u4 / i8 / f16, the
gemma-4 head-512 low-register layout).

On the models (cliloader device time per decode token, 128-token minus 1-token runs, B70 pinned at 1800
MHz, two runs in reversed order agreeing within 1%), attention kernel time HEAD -> this change:
phi-3.5-mini u4 0.60 / 0.60 / 0.68x at 2K / 4K / 16K, i8 BY_CHANNEL 0.66 / 0.68 / 0.74x, i8 BY_TOKEN
0.63 / 0.67 / 0.75x, f16 0.98 / 0.99 / 1.00x (16 subgroups only: f16 is bandwidth-bound there, the probe's
0.85x did not carry over); llama-3.2-1b u4 0.73 / 0.72 / 0.72 / 0.68x at 2K / 8K / 16K / 32K; gpt-oss-20b
u4 0.93 / 0.84x at 8K / 32K; llama-3.1-8b u4 (block2d) 1.000x. Device time per token: phi-3.5-mini u4
0.90 / 0.86 / 0.80x, llama-3.2-1b u4 0.96 / 0.92 / 0.89 / 0.82x, gpt-oss-20b 0.99 / 0.96x.

End to end (`test/run_daily_bench.py` with the daily prompts, one (model, size) at a time, HEAD / this change / the
opt kernel back to back in rotating order, B70 pinned at 1800 MHz, load <= 3.7 on 32 cores; 4-bit models, so a u4
cache), second-token latency against HEAD: phi-3.5-mini 0.95 / 0.97 / 0.89 / 0.86 / 0.85 / 0.82 / 0.78 at 32 / 1K /
2.2K / 4.5K / 9.2K / 17.9K / 36.6K tokens, phi-3.5-vision 0.88 / 0.82 / 0.75 at 2.2K / 9.2K / 36.6K, llama-3.2-1b
0.97 / 0.97 / 0.97 / 0.94 / 0.93 / 0.90 / 0.84 at 32 / 1K / 2K / 4K / 8K / 16K / 32K, minicpm4-0.5b 0.97 / 0.97 /
0.94 / 0.92 (32 / 1K / 4.4K / 17K), gpt-oss-20b 1.00 / 0.99 / 0.99 / 0.96 (32 / 1K / 8K / 32K); llama-3.1-8b (block2d,
the control) 0.993-0.997 with identical output MD5s. The opt kernel against this change: phi-3.5 1.00-1.15, llama-3.2-1b
1.00-1.09, minicpm4-0.5b 1.02-1.14, gpt-oss-20b 1.02-1.34. First-token latency within 1% except 32-token prompts
(0.98-1.03). phi-3.5's outputs differ between the iterations of one prompt in every build, so MD5 is no oracle there.

### xe_hpg reader (SG8)

On xe_hpg (`TEST_USE_SDPA_OCL_HPG=1`) `SDPAOclDecodeGenerator` compiles `sdpa_ocl_hpg_decode.cl`, which
`sdpa_ocl_decode.cl` includes in place of its own kernel when `SUBGROUP_SIZE` is 8 (the SG16 text is unchanged). The
contract, the partition and window arithmetic, the work split and the dequant identities are the SG16 kernel's; the
operand forms are DG2's SG8 DPAS (N = 8, A one int = two depth halves per lane, B int8, C float), and there is no 2D
block IO, so every read is a per-lane one:

- KQ: A = Q, lane l holding head dims (2l, 2l + 1) of each tile; B = K, lane = key, one 32 B (f16) / 16 B (i8) / 8 B (u4)
  read of the token-major row. A 16-token page is two DPAS per tile (keys `lane` and `lane + 8`), so a lane holds two
  scores per page. The BY_CHANNEL (scale, zp) pairs of a tile are two dwords per lane (channels 2l, 2l + 1), loaded
  per page and tile; their scale rides into the A operand and the zero-point correction accumulates per lane in the
  KQ loop (the SG16 kernel's `K_COMP_PER_READ` form).
- S*V: A = P, lane l holding keys (2l, 2l + 1) of the page read back from `slm_p`; B = V in VNNI order, dword j = keys
  (2j, 2j + 1) of the lane's head dim, read down the page rows. The V scale of keys 2l, 2l + 1 multiplies the
  probabilities; the zp of every key of the page is read by every lane (one uniform address per key).
- The page of a chunk is one load of its block-table entry at a uniform address, as `pa_v_pages` does for the SG8
  MIXED reader (above): the SG16 form, a per-lane load of page `lane` and `sub_group_broadcast`, gave wrong pages on
  DG2 under some code shapes there, and a partition's 16 pages do not fit one lane each anyway.
- `SG_PER_WG` is 8 (the cross-subgroup combine reduces one value per subgroup over the 8 lanes); M comes from
  `sg8_grf_estimate()`, the plain layout's count with the doubled Q and score shares, against the same budget of
  112; the low-register layout, block IO and the V prefetch are not written for SG8. u4 adds 8 for the nibble
  unpacks: u4 head 128 at M = 4 estimated 106 and spilled 224 bytes, so it runs M = 2. The S*V page loop is unrolled
  by 4 (`SV_CHUNK_UNROLL`), not whole: whole, i8 BY_CHANNEL head 128 at M = 4 (also estimated 106) spilled 0, 1184
  or 12160 bytes depending only on how the page index was read, because IGC hoisted the V reads of many pages at
  once. Every variant of `smoke_paged_attention_decode_tm` compiles without spill.
- The u4 V row pitch stays the page writer's (`Align(v_head / 2, 16)`), not the reader's subgroup.

The layout rule does not make xe_hpg pages token-major yet (`k_token_major_device()` keeps an Xe2+ term), because
`sdpa_ocl`'s MIXED does not read every kind of token-major page on SG8; until then the reader runs only on a page a
test forces token-major (`smoke_paged_attention_decode_tm`: f16, i8 BY_TOKEN, i8 BY_CHANNEL and u4 BY_CHANNEL, heads
64-512, k != v, kv groups 1 / 2 / 3 / 4 / 8, page and partition edges, window, sink, runtime scale, dynamic padding,
each also with a sharp softmax). `SDPA_OCL_NEG_SG8=30..35` break it on purpose and fail 26 / 34 / 20 / 2 / 4 / 5 of
the 51 cases. The compressed tolerances (25e-3 for i8, 75e-3 for u4) hide some of them: 32 fails only f16 cases, 33 only
the sharp i8 BY_CHANNEL heads 64 / 128 (u4 reads its pairs with the same line), 34 the four sharp i8 BY_TOKEN cases.
Its speed is not measured yet.

## Debug and bisection switches

All are read on the host when the kernel is compiled.

| Variable | Effect |
|---|---|
| `TEST_USE_SDPA_OCL_DECODE` | `0` selects the opt GENERATE path instead, on the same K page layout (until 2026-11-06 it also made the page d-major; `TEST_USE_SDPA_OCL`, which selected `sdpa_micro`, was retired with it; `OV_GPU_ENABLE_SDPA_MICRO=0` turns the DPAS kernel off) |
| `SDPA_OCL_KQ_TILE_KEYS`, `_TILE_QUERIES`, `_PER_WG_KEYS`, `_PER_WG_QUERIES` | Override the KQ tiling (`kq_sg_tile_keys` must be 16 or 32) |
| `SDPA_OCL_TRACE_CONFIG`, `SDPA_OCL_TRACE_STAGE` | Print the chosen tiling / PA stage |
| `SDPA_OCL_256GRF`, `SDPA_OCL_DECODE_256GRF` | Large-GRF compile (`SDPAOclGenerator` always on xe_hpg; `SDPA_OCL_256GRF=0` turns it off for the PA PREFILL tiling) |
| `SDPA_OCL_V_PREFETCH`, `_SM_BRANCHLESS`, `_MAX_SPLIT`, `_MASK_LAST` | PA PREFILL loop changes ("Paged-attention PREFILL tiling"): `0`/`1` forces each off/on |
| `SDPA_OCL_V_PREFETCH_L3` | PA PREFILL V tile L3 prefetch (on at `d_max` 512): `0`/`1` forces it off/on; on replaces `PREFETCH_V` |
| `SDPA_OCL_PLAIN_MIRROR`, `SDPA_OCL_PLAIN_TABLE_FLAGS` | Plain-SDPA f16 `d_max` 128 PREFILL ("Plain-SDPA PREFILL tiling"): `0`/`1` forces the table / the 256 x 128 tile; `0` drops the table tile's loop changes |
| `SDPA_OCL_PLAIN_D64` | Plain-SDPA f16 `d_max` 64 PREFILL on Xe2 ("Plain-SDPA PREFILL tiling (`d_max` 64)": the 32 x 32 tile, or the table with loop changes under a mask): `0` keeps the bare table tile (the jit before that change) |
| `SDPA_OCL_PLAIN_D512` | Plain-SDPA f16 `d_max` 512 PREFILL on Xe2 ("Plain-SDPA PREFILL tiling (`d_max` 512)"): `0` keeps the bare table tile at 128 GRF (the jit before that change) |
| `SDPA_OCL_PLAIN_DECODE` | Plain-SDPA decode ("Plain-SDPA decode tiling"): `0` keeps the table tile (the jit before that change) |
| `SDPA_OCL_WIDE_READS` | PA PREFILL wide reads (`d_max` 64: `5`, `0` with a sliding window; 128 / 256: `1`; 512: `7`; plain `d_max` 64: `1`, `0` with a sliding window): bit mask, `1` K pair, `2` V pair, `4` Q_slm as `ulong4`; `0` = narrow |
| `TEST_USE_SDPA_OCL_HPG` | `1`: xe_hpg may take the `sdpa_ocl` lane (default off; see "xe_hpg bring-up") |
| `SDPA_OCL_HPG_TIERS` | `all` or a comma list of `HpgTier` names: pretend those xe_hpg tiers are ready (dump only) |
| `SDPA_OCL_HPG_PA_PREFILL` | `0`: xe_hpg PA PREFILL at heads 64 / 96 / 128 / 256 / 512 takes the generic kernel instead of `sdpa_ocl_hpg_pa_prefill.cl` |
| `SDPA_OCL_HPG_PLAIN_PREFILL` | `0`: xe_hpg plain SDPA (no causality) at heads 64 / 72 / 96 / 128 / 256 / 512 takes the generic kernel instead of `sdpa_ocl_hpg_pa_prefill.cl` |
| `SDPA_OCL_HPG_PA_V_LEAD` | key blocks of V the xe_hpg PA PREFILL kernel loads ahead (1..key blocks of the tile, default 6; 3 at head 128 is miscompiled with a dynamic K / V padding, see "xe_hpg follow-ups") |
| `SDPA_OCL_HPG_PA_TILE_KEYS` / `_PER_WG_KEYS` / `_PER_WG_QUERIES` | xe_hpg PA PREFILL kernel tiling (16 / 32 keys per subgroup tile, subgroups along keys / queries); a tiling the kernel is not written for throws |
| `SDPA_OCL_HPG_PA_KQ_UNROLL` | xe_hpg PA PREFILL kernel: unroll of the KQ depth loop (default whole up to head 128, 2 at 256; `0` = whole) |
| `SDPA_OCL_SG8_KQ_UNROLL` / `SDPA_OCL_SG8_SV_UNROLL` | xe_hpg generic kernel: unroll of the KQ depth loop / S*V key loop (default 1 = not unrolled, `0` = unrolled whole, which spills) |
| `SDPA_OCL_NEG_SG8` | xe_hpg (SG8) only: `1` swaps the K pair order, `2` reads the S*V A operand transposed, `3` swaps the S_slm pair order (each must fail the sharp-softmax test); `4` forces the unaligned-K fallback, and the unaligned-Q staging of the xe_hpg PA PREFILL kernel (must pass, but for the IGC hazard cases in "xe_hpg follow-ups"); `5` / `6` read the full / per-key mask with the wrong lane width / one lane off (must fail the mask tests); `7` / `8` / `9` count the sink in every subgroup / read qq_bias transposed / end the sliding window one 16-key block short (must fail the PA feature tests); `10` reads plain-SDPA K / V of batch 0, KV head 0 in the PREFILL kernel (must fail `smoke_sdpa_hpg_rows`); `11` / `12` / `13` break the mask of the xe_hpg PREFILL kernel, `18` its head-72 output columns (above); on an i8 BY_TOKEN page in the generic kernel's MIXED `19` reads the K comp of tokens 0..7 for both 8-key groups of a page, `20` swaps the V scales of a lane's two keys, `21` takes the V zps of the next key pair, `22` swaps the two K channels of a dword (each must fail the i8 BY_TOKEN cases 0-16 of `smoke_paged_attention_comp_tm` but its PREFILL-only case 13: 15 / 16 / 16 / 16 of 17, 19 also passing the V-only case 5; 20 / 21 / 22 also fail the 13 BY_CHANNEL cases that read a page); on an i8 BY_CHANNEL page `23` swaps the (scale, zp) pairs of a lane's two channels, `24` scale and zp of a pair (each must fail the BY_CHANNEL cases 17-30 but the PREFILL-only case 27: 13 of 14); for the new tokens of a compressed MIXED `27` / `28` read the next token's Kc / Vc row (each must fail `smoke_paged_attention_comp_tm` but its PREFILL-only and tolerance-hidden cases: 23 / 21 of 33); `30` reads key `lane` for both halves of a page, `31` swaps the S*V key pair, `32` reads the page of chunk % 8 (pages 8-15 of a partition wrong), `33` swaps the BY_CHANNEL pairs, `34` drops the BY_TOKEN K zero point, `35` gives both keys of a pair the even key's V scale in the SG8 decode reader (each must fail `smoke_paged_attention_decode_tm`); `36` swaps the two keys of a K pair, `37` the V pages of adjacent blocks, `38` reads the odd V row of a pair past k in the generic kernel's token-major MIXED readers (each must fail `smoke_paged_attention_mixed_tm`: 36 / 37 fail 100 of 105, 38 exactly the 15 NaN-poisoned cases) |
| `OV_GPU_ARCH_OVERRIDE` | Debug caps only: report another arch (`xe_hpg`, `xe2`, ...) so its host code runs on this device (dump only) |
| `SDPA_OCL_Q_2D`, `_KV_2D`, `_A_2D`, `_K_I8_2D`, `_V_I8_2D` | Plain-SDPA block IO paths |
| `SDPA_OCL_K_PA_2D`, `_V_PA_2D`, `_K_PA_I8_2D`, `_V_PA_I8_2D`, `_K_PA_1D`, `_V_PA_1D` | Cache page read paths (`0` = scalar gather, same dequant) |
| `SDPA_OCL_PA_K_WIDE`, `SDPA_OCL_PA_K_WIDE_TILES` | MIXED f16 token-major K page: `0` reads one depth tile per message (default 1, two per message); depth tiles issued together, an even number (default 4) |
| `SDPA_OCL_PA_CUR_F16` | MIXED current tokens from Kc/Vc (default for compressed caches only; `0` = from the cache) |
| `SDPA_OCL_MIXED_PREFETCH_V`, `_K` | u4 / i8 MIXED page prefetches ("Paged-attention MIXED tiling", "... i8 and f16 caches"): `0`/`1` forces each off/on where the 8-bit page read is on |
| `SDPA_OCL_CAUSAL_SKIP` | PREFILL causal tile skip ("Paged-attention PREFILL: short prompts", "Plain-SDPA causal tile skip"): `0`/`1` forces it off/on (default on for PA at `d_max` 128 / 256 / 512, for plain `is_causal` f16 `d_max` 128, and as the KQ branch alone for plain non-causal table-tile stages without a mask) |
| `SDPA_OCL_MICRO_MATH` | `0` / `1` forces `sdpa_micro` rounding off / on for PA prefill (default on except the `d_max` 512 tile and on xe_hpg) |
| `SDPA_OCL_BIDIR`, `SDPA_OCL_BIDIR_GATE` | Negative controls for the image-group mask and its empty-buffer gate |
| `SDPA_OCL_DECODE_M`, `_SG_PER_WG`, `_PREFETCH`, `_K_2D`, `_V_2D` | Decode tuning and block IO paths (on xe_hpg only `_M` and `_SG_PER_WG` up to 8) |
| `SDPA_OCL_DECODE_LOW_REG` | Decode low-register layout: `0`/`1` forces it off/on (bit-identical either way) |
| `SDPA_OCL_DECODE_V_PAGE_1D`, `_K_LANE_TILES` | Decode per-lane pages: `0` restores the V byte gather and the head-dim split; K tiles per per-lane load, 1 / 2 / 4 dividing the tile count (bit-identical) |
| `OV_GPU_PA_K_TOKEN_MAJOR` | K cache layout ("Paged-attention cache layouts"): `0` every page d-major (MIXED then runs `pa_multi_token`, GENERATE the opt kernels), unset the rule. `1` and `OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR` are retired and rejected |
| `OV_GPU_PA_K_TM_BREAK=<consumer>` | Forces one consumer back to the d-major read, to test whether a suite observes it |

The negative controls must fail the token-type suites when disabled; a pass means the suite does
not observe the feature.

## Known issues

Found in the 2026-09 review of these files and deliberately left out of the refactor, because each
fix changes behaviour or lies outside them. Everything here comes from code reading unless a test
result is quoted.

### sdpa_micro removal (2026-11-02)

What deleting `sdpa_micro` (`sdpa_gen_micro.cpp`, `sdpa_micro.cl`, the PA `pa_tt_bounds` stage) changed. Xe2+ results
are unchanged (it never ran there by default); everything below is about the other devices, none of which is available
here, so none of it was measured on them.

- xe_hpg by default (DG2/Arc-A, ARL-H) and xe_hpc (PVC) have no DPAS SDPA kernel any more: plain SDPA and PA
  PREFILL/MIXED run the opt kernels, so attention there is slower by the old `sdpa_micro` margin (not measured).
  xe_hpg with `TEST_USE_SDPA_OCL_HPG=1` runs `sdpa_ocl` where its tiers are ready and the opt kernels elsewhere (the
  temporary routing of refused ops to `sdpa_micro` is gone). xe_hpc has no `sdpa_ocl` arm (`sdpa_ocl_arch_ok()`).
- Asymmetric planar KV-cache compression (what `kv_cache_compression.cpp` builds on every XMX device): an XMX device
  off the lane now runs it on the `sdpa_opt` stages and their planar zero-point path (`SDPAOpt::validate_impl()`),
  instead of `sdpa_micro`; without that it would have dropped to `SDPARef`. On the lane `validate_impl()` keeps its
  old decision (knob + `dpas_sdpa_op_supported()`). The opt planar path already ran on B70 for int4, a bf16 zero point
  and beam search, but not on those devices.
- The SDPA decomposition callback keeps SDPA fused for head sizes up to 256 on the `sdpa_ocl` lane only; other XMX
  devices follow the generic rules (heads below 64, odd heads and unaligned compressed heads are decomposed).
- Per-channel compressed KV (the GQA decomposition's quantized K/V, #37157) had only `sdpa_micro` as a reader:
  `sdpa_ocl` (`k_comp_per_key`), `sdpa_opt` and `sdpa_ref` read one scale per token. It was already broken on Xe2 by
  default: on B70 at 1d50b4272a all 24 per-channel cases of `smoke_sdpa_gpu_compressed_kv_per_channel` /
  `smoke_sdpa_gpu_gqa_decomp` failed or faulted the GPU (`CL_OUT_OF_RESOURCES`), on `sdpa_ocl` and with
  `OV_GPU_ENABLE_SDPA_MICRO=0` alike, while 18 passed with `TEST_USE_SDPA_OCL=0`. Now the GPU pipeline runs
  `GroupQueryAttentionDecomposition` with `fuse_compressed_kv = false` on every device (K/V are dequantized before SDPA,
  the path non-XMX devices always took), `SDPAOpt::validate_impl()` refuses per-channel compressed KV, and those two
  suites are `DISABLED_` reproducers. A per-channel reader in `sdpa_ocl` would let the fusion come back. Done on
  2026-11-02, see "Per-channel compressed KV".
- Opt-kernel accuracy gaps that become the default path off the lane (reproduce on B70 with
  `OV_GPU_ENABLE_SDPA_MICRO=0`): `pa_sdpa_opt` outputs zeros for Q/K/V with a 2 B feature padding
  (`smoke_paged_attention_prefill_h128` /15 and /33; those cases skip off the lane), `feature_pad/46` misses its 0.002
  tolerance (0.0034). Not fixed here. (This list also said the `sdpa_opt.cl` stages ignore `sliding_window`; they apply it
  with `IS_CAUSAL` (#38167). The 0.511 of `sdpa_sliding_window_test.opt_stage_kernels_apply_sliding_window` came from
  `sdpa_ocl`, which takes that test's single-element mask too; see "Plain SDPA sliding window".)
- Plain-SDPA cases only `sdpa_micro` got right, found when the removal widened the unit tests beyond the kit groups
  (B70, results identical before and after the removal; "micro" = `TEST_USE_SDPA_OCL=0` on 1d50b4272a):
  - transposed V (`input_v_transpose_order` ending `3, 2`, which `GroupQueryAttentionDecomposition` fused on XMX
    devices): `sdpa_gpu_micro.transposed_v_matches_non_transposed_v` / `_multi_tile` gave cosine similarity -0.008 /
    0.012 on `sdpa_ocl` and 0.084 / 0.028 on `sdpa_opt` (knob off), and passed on micro (#38309 added the layout to it).
    Fixed on 2026-11-02, see "GQA with a transposed V cache";
  - `sliding_window` on plain SDPA: 6 of the 9 `smoke/sdpa_sliding_window_test.basic` cases are below 0.95 (0.47 to
    0.89) on `sdpa_ocl` and passed on micro (and on `sdpa_opt`). Fixed on 2026-11-02, see "Plain SDPA sliding window";
  - K/V batch broadcast (`sdpa_broadcast_test`, e.g. Q batch 2 over K/V batch 1, NaN-guarded buffers): 16 of 18 cases
    fail on `sdpa_ocl` (`/0`: a non-finite output, as from a read past the K/V batch); `/0` passes on micro, and all 18
    pass on `sdpa_opt` (knob off). Fixed on 2026-11-02, see "Plain SDPA batch broadcast".
  On Xe2 these were wrong before the removal too; off the lane they move from micro to the opt kernels.
  Two suites also expect a kernel that no longer runs: `sdpa_opt_key_leftover_test` expects `sdpa_opt` but `sdpa_ocl`
  takes its single-element mask (fixed on 2026-11-02, see "Plain SDPA key leftover test"), and the old micro-pinned
  suites (`sdpa_gpu_micro.*`, `sdpa_micro_key_broadcast_mask_test`, `sdpa_micro_prefetch_k_test`,
  `sdpa_sliding_window_test`) now pin `sdpa_ocl` and skip off the lane (`sdpa_sliding_window_test.basic` also runs the
  opt stage kernels since 2026-11-02).
- Builds without oneDNN still have no DPAS SDPA: the `ENABLE_ONEDNN_FOR_GPU` guards around `sdpa_ocl` stay, although
  nothing in it needs oneDNN now (only the include-order note in `sdpa_gen_ocl.hpp` mentions `micro_utils.hpp`).
- `TT_BOUNDS_STAGE` in `paged_attention_opt.cl` is dead code, kept so the file stays upstream's.
- Kept names: `enable_sdpa_micro` / `OV_GPU_ENABLE_SDPA_MICRO` (now the switch for `sdpa_ocl`), the runtime flag
  `use_micro_sdpa` (means "the DPAS kernel serves this stage"; its gate `can_use_micro_sdpa_for()` became
  `can_use_sdpa_ocl_for()` on 2026-11-02), and test suites named after `sdpa_micro` that now cover `sdpa_ocl`.

### GQA with a transposed V cache (2026-11-02)

An ONNX `com.microsoft` GroupQueryAttention model whose V cache inputs and outputs are `[B, kv_heads, head_size, S]`
(`past_value -> Transpose{0,1,3,2} -> GQA -> Transpose{0,1,3,2} -> present_value`) was wrong or failed on every GPU.
`GroupQueryAttentionDecomposition` (GPU plugin, #38578) keeps that cache layout in `StatelessKV` (append along axis 3).
On XMX devices with the SDPA path it passed V to SDPA as value order `{0,1,3,2}`; `sdpa_ocl` (`ldv = VAL_S2`, `VAL_S3`
never read) and `sdpa_opt` (`VALUE_BLOCK_READ` along the head dim) both read the head dim of V as contiguous. On every
other path (softcap, `qk_output`, quantized KV, no XMX) it kept `StatelessKV -> Transpose -> Result`, which
`realloc_outputs_for_stateless_kv()` rejects ("stateless_kv should directly connect to an output"). Only that I/O
pattern was affected; a plain `[B, kv_heads, S, head_size]` cache never got the order. No real model with it was at
hand: the B70 results come from a synthetic ONNX model run through the frontend (cosine -0.007 / -0.055 / 0.0005 for
prefill 17, decode 1 and a 40-token chunk on `sdpa_ocl`, 0.04 / 0.08 / -0.02 on `sdpa_opt`, 1.000000 on CPU and with
the plain layout).

- The decomposition now keeps the cache layout on every path (`StatelessKV` axis 3 feeds `present_value` directly) and
  gives the attention V back as `[B, kv_heads, S, head_size]` through an explicit `Transpose{0,1,3,2}`, so SDPA always
  gets the default V order and the explicit chain and the KV dequantization get the layout they expect.
- Guard: `dpas_sdpa_op_supported()` no longer accepts V order `{..,3,2}` (a `sdpa_micro` leftover) and
  `SDPAOpt::validate_impl()` refuses any V order that moves the head dim, so such an op goes to `SDPARef`, which
  indexes V through `INPUT2_DIMS_ORDER`. Nothing in the pipeline makes that order any more; the two
  `sdpa_gpu_micro.transposed_v_*` tests now check `sdpa_ref` against the plain layout on `sdpa_ocl`.
- Test: `smoke_GQATransposedVCache` (`ov_gpu_func_tests`) runs GQA end to end against the TEMPLATE plugin: both cache
  layouts x SDPA / softcap / `qk_output` / i8 per-channel KV x head 64 / 128, steps prefill 17, decode 1, chunk 40 on
  a past of 18. On the B70 all 16 pass; with the plugin from before the change the 8 transposed-V cases fail (SDPA:
  wrong output, max error 1.86 and 1.98; the other paths: the `StatelessKV` exception) and the 8 plain-layout cases
  pass.
- Cost: one V-cache-sized permute per layer and step (`permute_ref`), plus a V-cache-sized intermediate buffer (its
  size follows the cache; not measured). Device time of one GQA layer at decode on the B70 (H 32, KV heads 8, head
  128, f16, perf counters, median of 50): plain layout 146 / 291 / 759 us at 1K / 4K / 16K context, transposed layout
  190 / 371 / 988 us (the Transpose is 31 / 74 / 223 us; SDPA is the same as for the plain layout). The old fused path
  was 417 / 844 / 2237 us, its SDPA 12 to 14x slower than the plain one besides being wrong. Reading `{0,1,3,2}` V in
  `sdpa_ocl` directly would save the permute; not done.

### Plain SDPA sliding window (2026-11-02)

A plain SDPA with `sliding_window` and `is_causal` attended over the whole causal range on `sdpa_ocl`:
`add_mask_jit()` emitted `SLIDING_WINDOW_SIZE` for paged attention only, so for plain SDPA every
`#if SLIDING_WINDOW_SIZE` block of the kernel compiled out (`sdpa_micro` emitted it for both). The kernel's window code
itself (the per-element predicate, the block-level skip) was right; its key-loop lower bound `window_k0_begin` lacked
the `CAUSAL_MASK_LOWER_RIGHT` shift that `sdpa_micro` had. `sdpa_opt.cl` applies the window (#38167); `sdpa_ref.cl` has
no window code at all, but it only gets an op `SDPAOpt` refuses, which the path below never produces (not fixed).

- Reach: the only producer of a plain-SDPA window is the GPU `GroupQueryAttentionDecomposition` (`local_window_size`).
  It drops the explicit mask and gives SDPA `is_causal` + the window for a causal GQA with `sliding_window_cache` and
  no attention bias, scale or sink; every other GQA window keeps the explicit causal + window mask with
  `is_causal = false`, where all kernels leave the window attribute alone. A synthetic ONNX model through the frontend
  (window 16, capacity 24, prefill 40, two decodes, a 20-token chunk; kit `test/xe2fix/x2`) gave cosine 0.933 /
  0.812 / 0.811 / 0.689 on `sdpa_ocl` (head 64; head 128 alike) and 1.000000 on CPU, with `OV_GPU_ENABLE_SDPA_MICRO=0`
  and with the explicit-mask control. No real model with it was at hand.
- Fix: plain SDPA jits `SLIDING_WINDOW_SIZE` too, and `window_k_begin` takes the `causal_offset` shift (as `causal_k`
  does). The shift changes no result (bitwise identical outputs on every case measured below; it skips only tiles
  every query of the workgroup masks).
- Tests: `smoke/sdpa_sliding_window_test.basic` runs every case on `sdpa_ocl` (`_dpas`, skips off the lane or with the
  knob off) and on the opt stage kernels (`_opt`, `enable_sdpa_micro(false)`), and checks which kernel compiled;
  `opt_stage_kernels_apply_sliding_window` pins the opt stage kernels the same way (its single-element mask went to
  `sdpa_ocl`); all three check against the host reference with the explicit causal + window mask ("Test coverage gaps"); `smoke_GQASlidingWindow` (`ov_gpu_func_tests`) runs the GQA end to end against the
  TEMPLATE plugin (native window, windowed cache with an explicit scale, full-length cache; head 64 / 128). On the B70:
  all pass; without the jit line the 6 `_dpas` cases whose window cuts a key fail (cosine 0.57 to 0.89) and the rest
  pass; with the bound one key tile too high 6 `_dpas` cases and `causal_true_with_explicit_mask` fail; the func test
  with the plugin from before fails its 2 native-window cases at every step and passes the 4 explicit-mask ones.
- Cost (SDPA device time on the B70, one GQA layer, 32 / 8 heads, head 128, f16, capacity 16384, perf counters, median of
  30, one process per case, 2 rounds within 1%, unpinned clock):

  | case | master (`sdpa_micro`) | jit only | jit + shifted bound |
  |---|---|---|---|
  | window 1024, decode on 4000 / 8000 / 16000 keys | 31 / 32 / 33 us | 116 / 241 / 490 us | 34 / 36 / 37 us |
  | window 1024, 512-token chunk on 8000 | 190 us | 1,392 us | 208 us |
  | window 1024, 4096-token prefill | 1,307 us | 1,479 us | 1,479 us |
  | window 4096, decode on 16000 | 102 us | 491 us | 130 us |
  | window 4096, 512-token chunk on 12000 | 644 us | 2,049 us | 694 us |
  | window 4096, 4096-token prefill (window never cuts) | 2,633 us | 2,930 us | 2,929 us |

  Without the shift a decode or a chunk walks every key, as it did without the window. With it `sdpa_ocl` is 8 to 27%
  slower than `sdpa_micro`, the same order as without a window (the last row; not investigated).
- Measurement trap, not investigated: within one process the SDPA time depends on the first shape the model ran. After
  a 4096-token prefill a decode on 4000 keys takes 818 us on master and 926 us here, against 31 / 35 us when it runs
  first, with or without a window; so each case above ran in its own process.

### Plain SDPA batch broadcast (2026-11-02)

The batch dimension of Q, K and V broadcasts numpy-style: each is 1 or the output batch (#38467 made the shape
inference take the broadcast and fixed `sdpa_opt` and `sdpa_micro`). `sdpa_ocl` read all three with the output batch
index `b1`, so a batch-1 input was read past its end: NaN where the NaN-guarded test buffers follow, rows of other
buffers, or a GPU page fault. The grid already took the output batch and the mask already wraps `b1` by `MSK_D0`.

- Reach: a core `ScaledDotProductAttention` with such batches stays fused on the lane (4D, head size up to 256), as
  in Whisper beam-search cross-attention (Q batch = beams, K/V batch = 1; ticket of #38467). One op through the plugin
  (kit `test/xe2fix/x3`): Q batch 3 over K/V batch 1 on 1500 keys faulted the GPU (`CL_OUT_OF_RESOURCES`) at 1 and at
  64 queries; Q batch 1 over K/V 2, K only and V only gave batch 1 off by 0.03 to 0.56 (batch 0 right); equal batches,
  CPU and `OV_GPU_ENABLE_SDPA_MICRO=0` were right.
- Fix: plain SDPA jits `QRY/KEY/VAL_D0` (the batch sizes) and `BROADCAST_{Q,K,V}_BATCH` by `sdpa_opt`'s rule (1 if
  that batch is not statically the output's, or any batch is dynamic; `get_broadcast_batch()` also rejects static
  batches that cannot broadcast). The kernel reads Q/K/V, the compressed scale / zero-point bases and the per-key V
  scales at batch `D0 == 1 ? 0 : b1`. That equals `b1 % D0` since a batch is 1 or the output's, without the emulated
  integer division: `%` added 213 instructions to the shape-agnostic kernels and moved their time by -2 to +4%, the
  select adds 8 and stays within 0.5% of before on every case below. A static equal batch compiles to the same ISA as before
  (plain prefill / single token, i8 compressed, a spilling head-72 kernel; `isa_ab.sh`).
- Tests: `sdpa_broadcast_test` runs every case on `sdpa_ocl` (`_dpas`, skips off the lane or with the knob off) and on
  the opt stage kernels (`_opt`), checks which kernel executed (it accepted either before, so a fallback would have
  passed), and adds 4 mask cases (mask batch 1 or the output's) and 3 causal ones (25 cases, 50 tests).
  `smoke_SDPABatchBroadcast` (`ov_gpu_func_tests`) runs one op with dynamic batch and lengths end to end (Whisper decode
  and chunk, Q / K / V alone broadcast, equal batches; plain, mask, causal; head 64 / 128) against an fp32 reference of
  the same f16 inputs. The TEMPLATE plugin is not the oracle there: its f16 result is up to 0.025 off on 1500 keys, and
  it throws "Tile with non-positive repeats" on some of these shapes (not investigated). On the B70 all pass; with the
  kernel and jit from before, the 23 broadcast `_dpas` cases fail and the 2 equal-batch `_dpas` and the 25 `_opt` cases
  pass; the func test with the plugin from before fails all 6 (batches 1 and 2 wrong at head 64, a GPU fault at 128).
- Cost (SDPA device time on the B70, one op, 32 heads, head 128, f16, K/V device-resident, perf counters, median of 30,
  one process per case, mean of 2 rounds within 1%, unpinned clock; both modes ran the shape-agnostic kernel).
  Before / after:

  | case | new key length every infer | same shape |
  |---|---|---|
  | decode on 4096 / 16384 keys | 172.5 / 605 us, 173 / 605 us | 165 / 596.5 us, 165 / 596 us |
  | batch 4 decode on 4096 / batch 8 on 256 | 536.5 / 91 us, 536 / 91 us | 564 / 74.5 us, 563 / 74 us |
  | causal prefill 2048 / 512-token chunk on 8192 | 856 / 141 us, 860.5 / 141.5 us | 863 / 140.5 us, 864.5 / 140.5 us |

  With the fix, the Whisper shape (5 beams, K/V batch 1, 1500 keys, 20 heads, head 64) takes 73 us per decode and
  92 us per 64-query chunk on `sdpa_ocl` (610 us on `sdpa_opt` for the chunk).
- Measurement trap: SDPA reading the K/V *input* tensors reads host memory over PCIe (4.2 ms instead of 167 us for a
  decode on 4096 keys, before and after alike), so `sdpa_cost.py` puts an `Add` in front. `sdpa_opt`'s perf counter
  shows 3 us for the Whisper decode and 17 us for the 4096-key one, below what reading K/V takes, so it covers only
  part of that stage (not investigated).

### Plain SDPA key leftover test (2026-11-02)

`sdpa_opt_key_leftover_test` (head sizes 40 / 72 / 76 / 136, key lengths that leave a partial 16-row block, f16 and
bf16, `[B,H,L,E]` and `[B,L,H,E]`, lower-right causal with a single-element runtime mask) was written for the
`sdpa_opt` multi-token kernel's leftover key read and asserted `sdpa_opt__multi_reg` ran. `sdpa_ocl` takes that mask,
so on the lane all 8 cases failed on that assert, which stopped them before the accuracy compare. A test change only.

- `sdpa_ocl` accuracy on these shapes (B70, the assert taken out): all 8 within tolerance, max error 1.5e-4 for f16
  and 1.1e-3 for bf16 (tolerance 5e-3 / 1e-2), no non-finite output; `sdpa_opt` (knob off) 4.7e-4 to 9.7e-4. With Q
  scaled by 8 (a peaky softmax) 2.8e-3 on `sdpa_ocl`, 2.6e-3 on `sdpa_opt`. Dropping the last key from the reference
  moves the output by 0.011 to 0.049, 1.6 to 10 times the tolerance, so the data catches a lost leftover row.
- Test: every case runs on `sdpa_ocl` (`_dpas`, skips off the lane or with the knob off) and on the opt stage kernels
  (`_opt`, `enable_sdpa_micro(false)`); each arm checks the kernel it executed, non-fatally, so the compare still runs.
  With the two arms' configs swapped on purpose, all 16 fail on the kernel check and none on the values.

### Per-channel compressed KV (2026-11-02)

The GPU `GroupQueryAttentionDecomposition` (#37157) can hand SDPA the quantized cache of an ONNX `com.microsoft`
GroupQueryAttention with `kv_cache_bit_width` 8 / 4 and `PER_CHANNEL` k/v scales: i8 symmetric (no zero point), or 4-bit
stored as u8 bytes of two nibbles (even channel low) described as u4 with a fixed zero point 8, plus `[1, kv_heads, 1,
head_size]` f16 scales. Since the `sdpa_micro` removal no kernel read that, so the fusion was off and the K/V were
dequantized before SDPA (a Convert + Multiply, for 4-bit a 16-node unpack chain, all `generic_eltwise_ref`).

- `sdpa_ocl` reads it (`KV_COMP_PER_CHANNEL`; `PLAIN_KV_U4[_SIGNED]` for packed 4-bit, u4 or i4), symmetric or
  asymmetric, on Xe2 with f16 queries (`per_channel_kv_supported()`). The scale / zero-point tensors are `[B or 1, KV
  heads or 1, 1, head size]`, read at their own batch and head.
  - K is folded, not dequantized per element: Q is staged as q times the channel's K scale (`q_fold_k_scale()`, one
    half rounding where the dequantization rounds k times scale), the KQ DPAS reads the stored code widened to half
    plus a constant (`I8_WIDE`: code + 1152, u4 + 1024, i4 + 1032), and sum_d q'_d (zp_d + bias) is subtracted from
    every score of the query (`k_comp_query_offset()`, in float; exact algebra). Per-element zp and scale had cost
    about 8% of a prefill.
  - V subtracts the column's zero point per element (exact for an integer zp) and multiplies the output column by its
    scale in the epilogue (in float).
  - 4-bit: K takes the u4 page's permuted depth (`PA_K_U4_CHANNEL`, Q via `q_chunk_u4`); V permutes the value columns of
    each 32-column window (`V_COMP_COLUMN`: two value blocks of one subgroup, or of two neighbouring subgroups at d_max
    <= 128, where a subgroup has one block), so lane L of the block with parity h takes nibble h of byte L and no lane
    shuffle is needed; the epilogue stores per column. The first version shuffled the bytes to the natural order: that
    was about 45% of a 4-bit prefill (6.3 ms instead of 3.4 ms for 4096 tokens).
  - Block reads (8-bit transform of the packed rows, `USE_2D_BLOCK_IO_{K,V}_I8` / `_U4`) where the row pitch allows
    (a 64 B row: i8 head 64+, 4-bit head 128+), per-element gathers otherwise (`k_tile_gather_ch`, `v_tile_gather_ch`).
- `SDPAOpt::validate_impl()` admits per-channel compressed KV only where `sdpa_ocl` serves it (lane, knob,
  `dpas_sdpa_op_supported()`, `SDPAOclGenerator::supported()`), and `execute()` refuses to run it on any other stage;
  everything else goes to `SDPARef`, which now reads it too (`COMPRESSED_PER_CHANNEL`, the nibble unpack, the default
  scale from the logical head size). `sdpa_opt` still reads one scale per token.
- The pipeline fuses again where `sdpa_ocl` serves it: `sdpa_ocl_selected(device, false)` (Xe2+ XMX, oneDNN build),
  `enable_sdpa_micro` on and f16 inference precision; the pass also requires a static head size up to 256 and an i8
  8-bit cache. Head 512 stays unfused although `sdpa_ocl` reads it right: its per-channel kernels spill (11136 / 8192
  bytes int8 / int4, the f16 one 1408) and one layer (16 heads, 4 KV heads) took 3410 -> 4846 us (+42%) for an int8
  prefill of 2048 tokens and 4000 -> 4306 us (+8%) int4, while decode on 4K keys gained 26% / 52%. Only the batch-1 caches the GPU decomposition builds take it (the full `StatelessKV` cache and the
  windowed `sliding_window_cache` one); a batch other than a static 1 goes through the common per-batch cache, which
  dequantizes before SDPA on every device.
- Tests: `smoke_sdpa_gpu_compressed_kv_per_channel` and `smoke_sdpa_gpu_gqa_decomp` (24, no longer `DISABLED_`) run
  each case on `sdpa_ocl` (kernel check) and on `sdpa_ref` (knob off, kernel check). Their decode cases are not causal
  for per-channel KV and Q is in [-8, 8]: a causal (upper-left) single query attends key 0 only, and a near-uniform
  softmax over 512 keys hides K as well; with the old data every decode case passed with a wrong K read.
  `smoke_GQAQuantizedKVCache` (`ov_gpu_func_tests`, 22) runs GQA end to end against the same model in f32 on TEMPLATE
  (TEMPLATE runs the f16 model in f16 and was up to 0.06 off on every GPU path), int8 / int4 x head 64 / 80 / 128 x full,
  windowed and batch-2 caches, plus heads 256 / 512; present caches bit-exact. Negative controls (a temporary build
  with `SDPA_OCL_NEG_X5`, since `SDPA_OCL_NO_GETENV` compiles the switches out): the K scale / zp of the neighbouring
  channel (1) or the V one (2) fail all 24 unit and all 16 fused func cases, swapped K (3) or V (4) nibbles every
  4-bit one, and only the batch-2 (unfused) func cases pass; `sdpa_ref` with per-token indexing fails all 24 (12 GPU
  faults).
- Cost, one GQA layer on the B70 (synthetic ONNX model, 32 heads, 8 KV heads, head 128, f16, perf counters, median of
  30, one process per case, mean of 2 rounds within 9%, unpinned clock; kit `test/xe2fix/x5`), dequantize-then-SDPA before /
  fused after, total device time of the layer (it also holds a full-cache `StatelessKV` copy per step):

  | case | int8 | int4 |
  |---|---|---|
  | decode on 1K / 4K / 16K keys | 180.5 / 580.5 / 2319 -> 125 / 360 / 1403 us (-31 / -38 / -39%) | 260 / 764 / 2999 -> 129.5 / 283 / 911 us (-50 / -63 / -70%) |
  | 512-token chunk on 4000 | 1379 -> 1283 us (-7%) | 1647.5 -> 1280 us (-22%) |
  | prefill 1024 / 4096 | 715.5 / 4609 -> 717.5 / 4993.5 us (0 / +8%) | 908 / 5182 -> 852 / 5359 us (-6 / +3%) |

  Head 256 (16 heads, 4 KV heads, no spill either way) behaves alike: int8 decode on 4K / 16K -26 / -28%, prefill 2048
  / 4096 +5 / +10%; int4 -45 / -52%, +1 / +4%.
  The fused SDPA itself is slower than the f16 one in prefill (int8 3382.5 against 2781 us at 4096 tokens): every query
  tile widens the codes again. The removed dequantization (52 / 200.5 / 965.5 us int8, 124.5 / 451 / 1991 us int4 at decode)
  pays for that elsewhere. Master's `sdpa_micro` reads int8 per channel faster (2485 us at prefill 4096, 274.5 us at
  decode on 16K); its fused 4-bit path does not compile ("No layout format available for sdpa": `validate_impl()`
  refused the u8 cache). Not done: folding V the same way (zp + bias through the exact row sum of the f16 P) measured
  about 7 points of prefill, but cancels a 1152-sized constant summed over every key.

### Token-major-only MIXED (2026-11-02)

The `sdpa_ocl` MIXED stage reads the token-major K page only: the d-major reads (`pa_k_tile_dt()`, the d-major arm of
the SG8 `pa_k_tile_gather`, the d-major `PA_K_TOKEN_STRIDE` / `PA_K_HIDDEN_STRIDE`, `IS_PA_K_TOKEN_MAJOR`) are gone, and
`can_use_sdpa_ocl_for()` sends MIXED on any d-major page to `pa_multi_token`. The refusal is in that dispatch gate only:
`choose_dpas_backend()` still adds both stages, so PREFILL (which reads the K input) keeps `sdpa_ocl`, and the MIXED
stage still compiles on a d-major page, as the token-major reader, without being dispatched.

- Xe2 defaults do not change: the default caches (u4 and i8 BY_CHANNEL) were already refused on a d-major page, and
  every page the rule makes token-major reads as before. What moved to `pa_multi_token` is MIXED on a d-major f16, bf16,
  i8 / u8 BY_TOKEN page: `OV_GPU_PA_K_TOKEN_MAJOR=0`, `TEST_USE_SDPA_OCL_DECODE=0`, and models the layout rule turns
  down while `sdpa_ocl` MIXED would take them (a head size that is not a multiple of 16, f16 head 16, an op of the
  model with alibi / scores / adaptive R-KV, non-f16 inference). Those MIXED steps now run the upstream kernel; how much
  slower that is was not measured. Since 2026-11-06 the page no longer depends on the ops or on
  `TEST_USE_SDPA_OCL_DECODE`, so of these only `OV_GPU_PA_K_TOKEN_MAJOR=0`, a u8 or bf16 cache, f16 head 16 (a square
  page) and non-f16 inference still give a d-major page ("Token-major K page by a static rule (2026-11-06)").
- xe_hpg with `TEST_USE_SDPA_OCL_HPG=1`: its pages are always d-major, so MIXED runs `pa_multi_token` there; PREFILL
  keeps `sdpa_ocl`. `PA_MIXED_F16` stays in `kHpgTiersReady` (PREFILL and MIXED compile together); the SG8 MIXED stage
  now compiles the token-major arm of `pa_k_tile_gather`; it runs on DG2 only on a forced token-major page
  (`smoke_paged_attention_mixed_tm`, "xe_hpg bring-up").
- Checked on the B70: the kit groups U1, SEL, RULE and F2 are identical to the build before; U3
  (`OV_GPU_PA_K_TOKEN_MAJOR=0`) has no failure, and 45 token-type tests written for `sdpa_ocl` MIXED now skip there. With
  the refusal taken out on purpose, U3 fails 85 tests (wrong values and NaN from the token-major read of a d-major
  page), so the refusal is what keeps them right. Under `=0` the PA tests built 111 distinct `sdpa_ocl` MIXED kernels
  (cliloader build log: 1173 builds, all `CL_BUILD_SUCCESS`) and enqueued none of them. The 50 SG8 f16 MIXED configs of
  the S5 `OV_GPU_ARCH_OVERRIDE=xe_hpg` corpus compile for `dg2` with the new body (`sdpa_ocl_ab.py hpg`, all with DPAS);
  that is a compile check with jit from an older build, not a run. WWB (llama-3.2-1b, deterministic): the default, f16
  and i8 BY_TOKEN arms give the same answers byte for byte; f16 with `OV_GPU_PA_K_TOKEN_MAJOR=0`, whose MIXED steps
  moved to `pa_multi_token`, went from 0.8685 to 0.8712 against the CPU reference (14 of 26 answers differ, the same on
  a rerun).

### Token-major K page by a static rule (2026-11-06)

The K page layout is `k_token_major_rule()` (device, f16 inference, page kind) instead of the per-op
`k_token_major_readable()` ("Paged-attention cache layouts"). `paged_attention_opt.cl` and the adaptive R-KV diversity
kernel got token-major readers for every kind, so an op the DPAS kernels refuse runs the opt kernels on the
token-major page. Moved from the d-major to the token-major page on Xe2: models with an op that has a scores output
(GenAI cache eviction), adaptive R-KV or alibi, a head size `sdpa_ocl` / `sdpa_ocl_decode` reject (above 512, not a
multiple of 16), k != v without an `sdpa_ocl` tiling, and the `OV_GPU_ENABLE_SDPA_MICRO=0` / `TEST_USE_SDPA_OCL_DECODE=0`
runs. Models every gate serves were already token-major and run the same kernels. The GQA multi-token fallback stays
off only where `sdpa_ocl` serves MIXED (as before on a token-major page); an op it refuses keeps the fallback. (Since
2026-11-08 it serves small steps there too: "Small GQA MIXED step on the token-major page (2026-11-08)".)

- Checked on the B70 (kit, `build_tests` Release): U1 672 PASS / 4 SKIP and F1 2199 PASS, the same tests as the build
  before; RULE (its truth tables rewritten for the new rule), U3, SEL and KVC without a failure. The new groups U1_OPT
  / U3_OPT (`TEST_USE_SDPA_OCL_DECODE=0 OV_GPU_ENABLE_SDPA_MICRO=0`, so the opt kernels serve MIXED and GENERATE, on the
  token-major / d-major page) fail the same 23 tests, as the build before did: PREFILL tests that expect `sdpa_ocl`
  under `OV_GPU_ENABLE_SDPA_MICRO=0`. Two
  broken builds were the negative controls: a neighbour's K row read in every token-major K reader of
  `paged_attention_opt.cl` (both 2D transpose rows, too) and in the R-KV BY_CHANNEL reader, and the two u4 V nibbles
  swapped. Both fail only on the token-major page (every K kind, R-KV BY_CHANNEL, the u4 V dual-nibble read and the GQA
  fallback) and nothing on the d-major one. The harness data observes a misread K row only with f16, so the new suites
  have `sharp_softmax` copies ("Test coverage gaps"). `smoke_adaptive_rkv` (kit groups RKV / RKV_DM) passes its 24 rows
  on both pages; with the broken R-KV reader its 4 BY_CHANNEL rows fail on the token-major page only.
- Cost, kernel level: GENERATE on `paged_attention_opt.cl` (`TEST_USE_SDPA_OCL_DECODE=0`), the token-major page (default)
  against the d-major one (`OV_GPU_PA_K_TOKEN_MAJOR=0`), same build. Device time per decode token from cliloader
  (128-token runs minus 1-token runs), B70 pinned at 1800 MHz, two runs in reversed order within 1.5 points of each
  other. Attention kernel, token-major against d-major, at 2K / 4K / 16K context:

  | Model | u4 | i8 BY_CHANNEL | i8 BY_TOKEN | f16 |
  |---|---|---|---|---|
  | llama-3.1-8b (32/8 heads, head 128; `pa_gqa_single_token` from 4K) | -23 / -17 / -16% | +2 / +4.5 / +4.0% | 0 / +10.8 / +10.2% | -10 / +4.9 / +5.8% |
  | phi-3.5-mini (32 heads, head 96; `pa_single_token`) | -21 / -21 / -25% | +7.5 / +2.7 / -0.3% | +8.6 / +4.2 / +3.2% | +0.6 / +4.3 / -1.2% |

  The whole decode token (every kernel) is at worst 2.8% slower (llama-3.1-8b, 16K, i8 BY_TOKEN), and 1.5-14.9% faster
  with u4, the default cache of the 4-bit models. The i8 BY_TOKEN read is a per-lane `vload16`; a 32-bit transpose like
  the f16 one might recover the GQA kernel's 10% (not tried).
- Cost, end to end (llm_bench, llama-3.1-8b 4-bit, so a u4 cache unless noted; 3 rounds x 3 iterations, arms rotated,
  CPU load at most 1.8; "before" = the build before the change): cache eviction (`start_size` 32, `recent_size` 128,
  `max_cache_size` 672, 4K prompt) 13.86 ms per token against 14.05 d-major and 14.06 before; 16K prompt (`recent_size`
  512, `max_cache_size` 4096) 15.02 against 15.14; first token the same. With i8 BY_CHANNEL, i8 BY_TOKEN and f16 the
  4K eviction decode differs by +0.1, -0.1 and -0.6% (2 rounds). A 4K prompt in 512-token chunks with
  `OV_GPU_ENABLE_SDPA_MICRO=0` (MIXED on `pa_multi_token`): first token 3575 against 5103 ms d-major and 5104 before; i8
  BY_CHANNEL 3452 / 3618, i8 BY_TOKEN 3456 / 3602, f16 2237 / 2307 (2 rounds). Decode with `TEST_USE_SDPA_OCL_DECODE=0`:
  -0.2% at 4K, -3.9% at 16K. A model every gate serves: second token 12.96 against 12.98 ms before, first token 531.8
  against 531.9 (657.2 / 657.3 in 512-token chunks). The generated text was identical (MD5) across token-major, d-major
  and before in every set.
- Not measured: alibi (no model), adaptive R-KV end to end, a head size the DPAS kernels reject.

### Padded key in the i8 BY_CHANNEL requantize (2026-11-06)

`pa_kv_cache_update_ref.cl`, `quantize_and_save_by_channel_block_with_requantize` (i8), stepped to the next new token
with the unpadded row `K_HEAD_SIZE * KV_HEADS_NUM` and ignored its `in_data_pitch` argument, so a key input with feature
padding was misread for the second and later new tokens of a partially filled i8 BY_CHANNEL block (a MIXED step that
continues a prompt at `past_len % 16 != 0`), on either page layout. Its three callers pass `KEY_IN_STRIDE`, the padded
row; the u4 twin and the prefill helper already stepped by it. GENERATE stores one token, so only MIXED changes. Found
while testing the block2d gate, which is why the compressed `paged_attention_feature_pad_test` case padded Q and V only;
it pads K too now.

The output of that step does not show it: MIXED reads the new tokens from the key input, not from the cache, and the
feature-pad case (and its sharp-softmax copy) passes with the bug in it; a later step reading those tokens would see
them. `smoke_paged_attention_key_pad_cache` dequantizes the K
cache back and compares every stored token with its input (B70, both page layouts): before the fix the three i8
BY_CHANNEL cases with a padded key (dynamic 16 B, static 16 B, static 4 B pads) fail at the second new token of the
`{10, 34}` subsequence (stored -0.0001 for -0.0767, tolerance 0.01); the unpadded, u4 BY_CHANNEL and i8 BY_TOKEN controls
pass. After it all six pass.

### Small GQA MIXED step on the token-major page (2026-11-08)

A small GQA MIXED step (at most 16 new tokens per subsequence: a speculative-decoding validation step, a draft model
catching up, a short prompt continuation) never took the GQA fallback where `sdpa_ocl` serves MIXED on the token-major
page, a leftover of the per-op layout rule, when the fallback could not read every token-major page: it ran `sdpa_ocl`
while new tokens^2 x context was at most 64K and wherever `pa_small_q_opt.cl` cannot run (token_type_ids, a kv group
that does not divide 8), `pa_small_q_opt.cl` above. Since the fallback reads every token-major page ("Token-major K page
by a static rule (2026-11-06)"), `select_gqa_multi_tokens_kernel()` picks among the three there
(`token_major_gqa_*` helpers in `paged_attention_opt.cpp`):

- the fallback while new tokens^2 x context is at most 320K with an f16 or u4 cache and 192K with an i8 one;
- above that, `pa_small_q_opt.cl` from a 1.5K context;
- otherwise, and where `pa_small_q_opt.cl` cannot run, the fallback, except `sdpa_ocl` for an f16 cache above 12 new
  tokens;
- both thresholds scale with min(1, head size / 128): measured at head size 64, 128 and 256, the sizes in between
  interpolate.

An op `sdpa_ocl` refuses, a d-major page and every other device keep upstream's choice. `OV_GPU_PA_GQA_MULTI_TOKENS_KERNEL=2`
now forces the fallback on the token-major page too (it fell through to `sdpa_ocl` there).

Why each wins: the fallback reads the KV once per new token but spreads the work over tokens x heads x 256-key
partitions; `pa_small_q_opt.cl` reads it once per partition with DPAS, but a work-group of ceil(tokens / TOKENS_PER_SG)
subgroups dequantizes every K/V tile into SLM, which costs the same for any token count and leaves the GPU mostly idle
at a short context (32 work-groups of 2 subgroups at 4 new tokens, 1K context, 32/8 heads); `sdpa_ocl` has no split over
keys, so its time does not depend on the token count and grows with the context.

- Measured on the B70 at 1800 MHz (all busy clock samples 1800, no throttle, CPU load at most 2.8): unit-test perf rows
  inserted locally (`DISABLED_gqa_perf`, inputs copied to `usm_device`), kernel plus finalization from the cliloader
  chrome timeline, median of 100 calls, the KV cache pushed out of L2 before every call (a 128 MB copy on the same queue,
  as the other layers of a model do; without it the times are 10-50 us lower, `pa_small_q_opt.cl` most, and the
  ranking was the same in the cases compared), harness data at 1/1024 resolution (no difference from the default 1/8
  here). Forced arms: `sdpa_ocl` `OV_GPU_PA_GQA_MULTI_TOKENS_MAX=0`, `pa_small_q_opt.cl`
  `OV_GPU_PA_GQA_MULTI_TOKENS_KERNEL=1`, the fallback on the same token-major page `OV_GPU_ENABLE_SDPA_MICRO=0` plus
  `OV_GPU_PA_GQA_MULTI_TOKENS_KERNEL=2`. 736 cases: 32/8 heads at head size 128, 2-16 new tokens x 512-16K context, f16,
  u4 BY_CHANNEL, i8 BY_CHANNEL and i8 BY_TOKEN, each with and without token_type_ids (two rounds within 3.1%), and 32/8
  heads at 64, 16/8 at 256, 24/8 and 28/4 at 128, 2-16 x 1K-16K, the four caches (one round).
- The previous choice took 1.98x the fastest kernel's time (geometric mean over the 736; up to 20x: head 64, u4, 2 new
  tokens, 8K, where the u4 `sdpa_ocl_mixed` tile at `d_max` 64 spills). The new default, measured with the change
  (no environment): the kernel the rule names ran in all 736, 1.007x the fastest (up to 1.69x: 16/8 heads, head 256,
  f16, 16 new tokens at 1K, where `sdpa_ocl` would take 91 against 153 us), 0.51x the previous default's time (geometric
  mean); slower than the previous default by more than 2% in 3 cases, at most 7.3% (24/8 heads, f16, 12 new tokens at
  1K: 77.6 against 72.3 us). Previous / new default with the u4 cache (default for 4-bit weights), us per call:

  | Heads, head size | Context | 2 / 4 / 8 / 16 new tokens |
  |---|---|---|
  | 32/8, 128 | 1K | 109/30, 109/32, 166/52, 142/87 |
  | | 4K | 391/56, 215/85, 181/149, 193/193 |
  | | 16K | 370/174, 351/292, 388/388, 633/633 |
  | 32/8, 128, token_type_ids | 1K | 114/31, 114/32, 113/52, 114/85 |
  | | 16K | 1579/174, 1578/294, 1579/535, 1580/1041 |
  | 32/8, 64 | 1K | 150/30, 150/30, 72/32, 73/73 |
  | | 4K | 568/36, 92/52, 78/78, 85/85 |
  | 24/8, 128 | 4K | 392/55, 394/84, 395/146, 396/273 |

- End to end (llm_bench, speculative decoding: llama-3.1-8b-instruct with llama-3.2-1b-instruct as the draft, both
  4-bit, so u4 caches, `num_assistant_tokens` 5, 256 new tokens, fixed prompt, 5 iterations x 3 runs in alternating
  order; `build_release` install of HEAD against the same install with the new plugin): at an 8K prompt the generated
  text (MD5) and the number of speculative steps (45) were the same in both arms and the second-token latency went from
  10.36 to 7.83 ms (-24.4%), first token unchanged. At 2K and 4K the validation step's kernel changes the numerics, so
  the generated text and the step count differ (74 -> 70, 74 -> 75) and the latencies (10.35 -> 8.80, 11.26 -> 10.17 ms)
  are not the kernel's alone. Per call under cliloader (warm-up plus one iteration): the main model's validation step
  (6 new tokens) 179 / 186 / 210 us on `pa_small_q_opt.cl` against 67 / 119 / 208 us on the fallback at 2K / 4K / 8K;
  the draft model's MIXED steps (head 64) 310 / 593 / 1148 us on `sdpa_ocl_mixed` (the spilling `d_max` 64 u4 tile)
  against 26 / 30 / 51 us on the fallback; device time of the whole run 5239 / 6278 / 7183 ms against 4347 / 5589 /
  5756 ms.
- Numerics: the fallback multiplies without DPAS and dequantizes in f16 where `pa_small_q_opt.cl` and `sdpa_ocl` use
  DPAS (and `sdpa_ocl` the exact u4 dequant); it is upstream's kernel for these steps on every other device. The PA
  suites pass at their tolerances, and the 2K / 4K texts above differ because a greedy token flipped.
- Tests (B70, `build_tests` Release): `smoke_paged_attention_gqa_multi_tokens` mirrors the choice in its kernel oracle
  and has 14 new rows (basic/48-61): a sharp-softmax row per `pa_small_q_opt.cl` branch (f16, i8 BY_CHANNEL, i8
  BY_TOKEN at head 128, u4 at head 64), two f16 rows above 12 new tokens below the 1.5K context (`sdpa_ocl`), and eight
  token_type_ids rows (sharp softmax, short context; the test asserts that the bidirectional reference differs from the
  causal one by more than the tolerance, which two u4 rows at 8 new tokens failed, so they have 16). Negative controls:
  the previous choice with the new tests fails 52 of the 62 rows on the kernel; the fallback with its token_type_ids
  extension removed fails exactly the 7 token_type_ids rows it serves. Kit groups against the build before: U1 710 PASS
  / 4 SKIP and U3 573 / 60 (the 14 new rows added, nothing else changed), U1_OPT the same 25 failures as the build
  before (tests that expect `sdpa_ocl` under `OV_GPU_ENABLE_SDPA_MICRO=0`, `feature_pad`/46, `prefill_h128`/15, /33), SEL, RULE, KVC, RKV and SDPAX unchanged.
- Not measured: devices other than the B70 (the thresholds are its), several subsequences of different lengths (the
  choice takes the longest subsequence and the largest context), head sizes other than 64 / 128 / 256, the i8 and f16
  caches and a token_type_ids draft end to end, an EAGLE3 tree (qq_bias) end to end.

### Correctness

- Plain SDPA with `is_causal` and an attention mask: the spec ignores the mask when `causal` is true, and so do the
  CPU reference, `sdpa_ref.cl` and `sdpa_opt.cl` (`#elif !IS_CAUSAL && HAS_ATTN_MASK_INPUT`), but `sdpa_ocl.cl` adds
  it on top of the causal mask (`STATIC_SCALAR_ATTN_MASK_VALUE`, `HAS_SCALAR_ATTN_MASK` and `WITH_ATTN_MASK` in the
  logit loop); the plugin passes both (`ops/scaled_dot_product_attention.cpp`). A finite scalar shifts every logit
  alike and changes nothing. Measured on the B70 with the `sdpa_opt_key_leftover_test` shapes and a single-element
  runtime mask of -inf: `sdpa_ocl` outputs all zeros (no NaN), `sdpa_opt` (knob off) the right result; with 0 or 5
  both are right. A full mask with causal is not measured. Found in the 2026-11-02 X4 work, not fixed.
- Compressed BY_CHANNEL with f32 or ACCURACY (dynamic) inference precision:
  `transformations_pipeline.cpp` sizes the per-channel comp by the inference precision (f32: i8
  page row 24, u4 16; dynamic has size 0: 16 and 8), while `graph/paged_attention.cpp` and
  `ops/paged_attention.cpp` assume f16 comp (20 / 12), so the block-size assert should fire at run
  time on either layout. Master has the same code. From code reading only.
- `pa_kv_cache_update_ref.cl` loads the new K / V tokens with `BLOCK_READN` (a subgroup block read, which needs a
  4 B aligned address), so a K / V input whose rows start 2 B off a dword (an odd f16 padding before the head, or
  an odd row pitch) is written to the cache wrong, on every arch. `smoke_paged_attention_comp_tm` cases 31 / 32
  pass because a compressed MIXED reads its new tokens from Kc / Vc, but fail with `SDPA_OCL_PA_CUR_F16=0`
  (0.044 / 0.055 against 0.025); a later step reading those tokens from the cache gets the wrong values. The two
  controls (the same case without the pads passes; the same padded case passes on Kc / Vc) put it in the writer,
  the alignment cause is from code reading. Found in the xe_hpg port (P7e), not fixed.
- `sdpa_opt.cl` types a paged-attention runtime scale as `INPUT3_TYPE` (`SCALE_TYPE` follows
  `HAS_ATTN_MASK_INPUT`, which the paged-attention generator never sets, and INPUT3 is a 32-bit index
  input there), so `pa_sdpa_opt` PREFILL and MIXED read the 16-bit scale as an int32. Real models give a
  constant scale and never reach it. `paged_attention_runtime_scale_test` is skipped below Xe2 because of it.
- `SDPA_OCL_KQ_TILE_KEYS=32` gave wrong results when `kq_sg_per_wg_keys >= 4` and a subgroup had
  two query blocks (`kq_sg_tile_queries = 32`): deterministically, the first query block was right
  and the second was not (last reproduced 2026-09-10 on `paged_attention_test.basic/31`). It no
  longer reproduces (2026-10-24): basic/31 and a sharp-softmax copy pass with the failing tilings
  (32 x 32 subgroup tiles, 4 or 8 key subgroups, 1 or 2 query subgroups) with `MICRO_MATH` 0 and 1,
  and the outputs differ by tiling only in rounding. The harness data predates the failure, so a
  change between 2026-09-10 and 2026-09-25 fixed it; which one was not found. The `d_max` 512
  PREFILL tile has used 32-key tiles since, checked bit for bit against `sdpa_micro` ("Paged-attention
  PREFILL tiling").

### Latent

- `SDPAOclGenerator::supported()` admits bf16 for paged attention, but every compressed-cache path
  assumes f16 (`as_half8(pA)` in `pa_v_comp_fold()` and the like). Only
  `PagedAttentionOpt::validate_impl()`, which takes f32/f16 queries, keeps bf16 away.
- The plain-SDPA i8 bias trick ("Dequantisation") is exact only for an integer zero point. Today it
  is one: KV compression stores an i8 zp on XMX devices (`kv_cache_compression.cpp`), and
  `sdpa_ocl` needs XMX. The int4 configuration keeps its zp in the query type, though, so
  dispatching plain int4 would round the zp to the 1.0 ulp at 1152, and the 0.1 tolerance of the
  kv_cache_sdpa tests would not notice.
- Plain-SDPA int4 KV compiles an `sdpa_ocl` stage that `execute()` never dispatches.
- A plain-SDPA int4 KV decode with an unaligned head runs the opt single-token kernel, which does
  not support it: `execute()` skips `regular_multi_tokens` whenever the `sdpa_ocl` single-token
  stage is staged, then skips that stage for int4. Master sent every unaligned decode to the
  multi-tokens kernel. `unaligned_head_size()` reads the packed K/V layouts, so logical heads 80
  or 112 take the same route harmlessly; a logical head of 72 would be wrong.
- `unaligned_head_size()` gives the rank-3 descriptor orders of a 3D SDPA to layouts canonicalized
  to 4D, so it reads the sequence length instead of the head size. A 3D decode with an unaligned
  head and a key length divisible by 16 runs the opt single-token kernel (as on master).
- `tiling_fits_device()` allows 1,024 work-items per group whatever the GRF mode, but a 256 GRF kernel
  cannot dispatch that many: `SDPA_OCL_KQ_PER_WG_KEYS=16 SDPA_OCL_KQ_PER_WG_QUERIES=4` (64 subgroups) on a
  PA PREFILL tiling compiled with 256 GRF fails at enqueue with `CL_INVALID_WORK_GROUP_SIZE` (B70,
  gemma-4-e2b). The shipped tilings stay at 32 subgroups or fewer, so only an override reaches it.

### xe_hpg follow-ups

What the xe_hpg integration leaves open. Nothing here falls back to `sdpa_micro`; the timing items wait for a DG2 that
is not shared (the harness numbers are from the 2026-10 study on an A770, driver 26.35). Every speed against `sdpa_micro`
below was measured before this branch removed it (real models on the same A770, 2026-10-30 to 11-04); the xe_hpg port
re-measures against the opt kernels.

- PA PREFILL prompts up to about 2K tokens are slower than `sdpa_micro` with the 256-key x 32-query tile: 65 and 128
  tokens about 1.2x, 384-768 tokens 1.03-1.16x with gaps between calls, 2048 tokens 1.03x. Two study kernels cover parts
  of it (`h128_2_2_active_q8_pfix`, 8 queries, best up to 512 tokens; `h128_2_2_one_e1p6`, 16 queries x 256 keys, best
  around 1-2K); a kernel choice by the longest subsequence is not written. 384 and 768 tokens with gaps had no
  passing candidate.
- With gaps between calls the short and medium prompts pay a fixed cost `sdpa_micro` does not (768 tokens: +82 us
  against +59 us from queued to gapped), not explained (not the instruction cache: 40 KB of OCL code against 71 KB of
  `sdpa_micro`).
- The xe_hpg PA PREFILL kernel serves heads 64, 96, 128, 256 and 512 (plain SDPA also 72); other heads take the generic SG8
  kernel. It serves a sliding window, a sink and token_type_ids; alibi or a runtime scale takes
  the generic SG8 kernel. At head 256 it is 1.13x `sdpa_micro` at 1K tokens (gemma-2-9b; gemma-4-e2b's window layers
  1.02-1.05x), at head 128 1.06x at 0.6-1.4K tokens and 1.11-1.13x at 4.9K (32 / 8 heads, qwen3-vl-4b, real-model
  device time per call), at head 96 1.083x at 1K tokens and 1.10x at 32 (phi-3.5-mini, 473 / 67.6 us against 437 /
  61.4; 3940 us on the generic kernel) and 1.07x in phi-3.5-vision, at head 512 0.90-0.91x at 1K tokens (gemma-4-e2b)
  and 0.94-0.96x on flux.1-schnell's VAE.
- The head-128 gap at 4.9K tokens (qwen3-vl-4b, 7337 us against 6518) is reproduced to 0.2% by a standalone harness
  with the product's jit header and `sdpa_micro`'s final-linked binary, once V is the product's dynamically padded view of
  the fused QKV output (row pitch 6144): 6982 us with a static V padding of 0, 7178 with a dynamic one of 0, 7348 at pitch
  6144, while `sdpa_micro` stays at 6513-6531. Bounds by removing one part (wrong output, timing only, pitch 6144): V loads
  of the S*V loop hoisted 5565 us (-24%), no score transpose 6847 (-6.8%), Q read from SLM once 6945 (-5.6%), P read from
  SLM once 7148 (-2.8%), K loads hoisted 7292. No knob closes it (`SDPA_OCL_HPG_PA_*`: 128 x 32 7858, 256 x 64 7595, V
  lead 3 / 4 / 8 7287 / 7360 / 8545, KQ unroll 2 7450), nor did P as qwords (one 256 B SLM read per operand: 7920), 32-bit
  or strength-reduced V row addresses (7338 / 7803), keys (l, l + 8) per lane for consecutive V rows (7266) or IGC's bank
  conflict reduction (`IGC_EnableBCR=1`: 7300, and it hits the hazard below). Left: a softmax in the KQ output layout
  (lane = key), which drops the transpose.
- IGC can compile the S*V loop of the xe_hpg PA PREFILL kernel so that one 8 x 8 accumulator tile of a random subgroup and
  head comes out as garbage (inf, NaN, values in the thousands), differently from run to run. `SDPA_OCL_HPG_PA_V_LEAD=3`
  at head 128 does it with a dynamic K / V padding: `smoke_paged_attention_prefill_h128/16` and `/40` fail every run (the default lead
  6 passes), as do the qwen3-vl shape and 64 tokens at 32 / 8 heads in the harness; so does the kernel with the first V
  blocks loaded after the barrier wait, and the default one with `IGC_EnableBCR=1`. With the same source,
  `IGC_EnableForceDebugSWSB=1`, `IGC_EnableVISANoSchedule=1` or `IGC_VISAOptions=-DPASTokenReduction` (the same
  instruction stream, other SWSB tokens) give correct output in every run, and IGC 2.38.5 compiles the lead-3 kernel
  correctly but not the `IGC_EnableBCR=1` one; IGC 2.41.5 is the driver's (26.35). A static scan of the listings found no
  missing SWSB wait under in-order DPAS completion, so the hazard is not pinned down. At head 512 (512 x 16) a V lead of 4
  does it on IGC 2.41.5 (`prefill_heads/116`, 2-3 runs in 15), and IGC 2.38.5 does it with the default lead 6 (`/116` and
  `/143` in every run); only the driver's IGC is a gate. The forced fallback build of `SDPA_OCL_NEG_SG8=4` (a compile-time
  variant the product never builds: `q_dword_ok` and `mask_dword_ok` are constants there, while the product kernel computes
  them per call) hits it on IGC 2.41.5 as
  well, so these are known failures of that control, not of the product kernel: `prefill_heads/116` and `/143` (head
  512, dynamic K / V padding, NaN) in every run, `smoke_sdpa_hpg_rows/*head512_q1100_kv1100*` in one run of three, and
  `prefill_heads/196` (head 96, dynamic padding) once in four; all four pass with `IGC_EnableForceDebugSWSB=1` and on IGC
  2.38.5. A change to that loop has to pass the dynamic-padding cases several times (uncached,
  `NEO_CACHE_PERSISTENT=0`).
- The product kernel hits the same hazard with plain SDPA: `smoke_ScaledAttnDynamic4D_GPU` f16 at head 128, 5 / 5 heads,
  batch 2, 100 and 387 queries, only V transposed (order {0, 2, 1, 3}: V rows 1280 B apart, the pitch a literal), no
  runtime scale (10 cases) fails in every run on the xe_hpg lane, several 8 x 8 accumulator tiles per run. They pass with
  `IGC_EnableForceDebugSWSB=1`, with `IGC_VISAOptions=-DPASTokenReduction` and with a V lead of 2-5 or 8, and fail on
  IGC 2.38.5 as well. The same kernel logic passed them before the broadcast-batch selects changed the code layout. Open
  until a fix (a V lead, a vISA option for these kernels, or the hazard pinned down) passes them several times uncached.
- The generic SG8 kernel (heads the PA PREFILL kernel does not serve, alibi / a runtime scale, PA MIXED, plain SDPA with causality or at other heads) was 3.6-14x `sdpa_micro` per call in real models (A770, f16 KV,
  2026-10-30/31): PA PREFILL 6.5-14x at 1K tokens (1.2-2.1x at 32), PA MIXED after a long prompt 4.3-8.3x against
  `sdpa_micro__generate` (measured at head 128), plain SDPA 3.6-4.2x (phi-4-multimodal's vision encoder at head 72,
  flux.1-schnell's VAE at head 512); the PREFILL kernel serves those plain SDPA heads now.
- Not ported to SG8 yet: the u4 cache (`PA_U4`).
- Every qq_bias feature case is MIXED, which runs `pa_multi_token` on xe_hpg while its pages are d-major, so
  `SDPA_OCL_NEG_SG8=8` has nothing to detect there yet. Where `sdpa_ocl` ran them (an earlier d-major SG8 MIXED reader)
  the control failed them only just past the tolerance (first error about 0.003); a case with a short past, where the
  speculative keys weigh more, would make it sharper.
- On DG2, `SDPAWithKVCacheTest` bf16 `batch=2 causal=1 compressed=1` head 64 fails in a group run on both lanes and
  passes alone (order or state dependent, not investigated), and f32 PA (`smoke_PagedAttentionTokenType` `Prc=f32`,
  4 heads, 500 tokens) ends in `CL_OUT_OF_RESOURCES` on every lane.
- `paged_attention_test.basic/182` (MIXED f16, head 512, 8 / 2 heads, token-major K page, sharp softmax) returned `inf`
  once in a group run on the xe_hpg lane (2026-10-29, U2 with `OV_GPU_PA_K_TOKEN_MAJOR=1`); three runs alone and the
  group rerun passed. A token-major K page takes the generic SG8 MIXED kernel; not investigated.
- `sdpa_micro` misreads a PA PREFILL input with a 2 B static feature padding on DG2 (head 128: the output follows V of
  the wrong rows); `smoke_paged_attention_prefill_h128` skips its padded cases where `sdpa_ocl` does not serve the op.
- `test/sdpa_ocl_ab.py` synthetic modes are stale on Xe2 corpora: `kv2d0` must also set `PREFETCH_V=0` and `dv2d0`
  `V_TWIN_PAIR=0`, or those jobs fail to preprocess (lost coverage, not a false pass).

### Test failures that predate the refactor

Measured on an Arc Pro B70 (Xe2), identical before and after the refactor. Of these only `SDPAFusion.Inference/0` still
fails (kit F2, 2026-11-06):

- `SDPAWithKVCacheTest.MultipleIterationStateful`, f16 with compressed KV at head 512
  (`..._et=f16_num_iter=5_num_groups=4_..._compressed=1k_head=512v_head=512`): 4724 of 8192
  elements wrong, max difference 0.99. Not `sdpa_ocl`: the KV-cache quantize kernel
  (`dynamic_quantize_kernel_opt_kv_cache`) refused heads above 256, so head 512 ran the reference kernel, which cannot
  append, and the cache was corrupted. Fixed by upstream #38466 (in this branch); both head-512 cases pass.
- The 14 bf16 compressed cases of the same test failed in `add_required_reorders` (no i8 layout for
  `dynamicquantize`) before an SDPA implementation was chosen. Upstream #38532 made them reach SDPA,
  and `sdpa_ocl` then misread their bf16 scales ("Dequantisation"; fixed). The 13 bf16 compressed cases F2 has today
  pass.
- `SDPAFusion.Inference/0` does not find the fused SDPA node. It is the only 3D case, and from the code (not run)
  `should_decompose_sdpa_for_memory_size()` in `transformations_pipeline.cpp` decomposes a static 3D SDPA again on an
  XMX device whose Q.K buffer fits, so the runtime model has no SDPA node; not `sdpa_ocl`.

### Test coverage gaps

No test reaches paged-attention bf16, qq_bias with an f16, u4 or
BY_TOKEN cache, a compressed cache with token_type_ids, or the u4 scalar page arms (head 48, 96,
112). For a u4 V head whose packed row (`V / 2`) is not a multiple of 16 (48, 80, 112) the harness cannot build a
case: it lays the u4 V row out as `V / 2` (+ 4 comp bytes on the d-major page) where the kernels pad it to a multiple of
16 (`PACKED_V_HEAD_SIZE`, `PACKED_ADJUSTED_V_HEAD_SIZE`), so every reader, on either page layout, returns inf / NaN
there; the scalar u4 V reads of `paged_attention_opt.cl` and `sdpa_ocl` are unverified. `sdpa_ocl_decode`'s
low-register layout at `Q_PER_WG = 8` is reached by
`smoke_paged_attention` basic/186-189 (u4, kv group 8, heads 512 with `V_TWIN_PAIR` and 256), but only
the `sharp_softmax` rows (187, 189) fail when the twin store or the zp-correction bias is broken; the
plain rows pass either way. Lower-right causal masking is checked only by `sdpa_gpu_causal_mask` (element-wise on its
`prefill_static_*` and 300 / 2300 rows, cosine >= 0.99 on the rest), and several suites check only finiteness, cache
contents or cosine >= 0.95.

`force_implementations` with a kernel name does not choose between the `ocl_v2` SDPA impls: only the impl type and the
format are used (the name reaches kernel_selector impls only), so a network forced to `"sdpa_ref"` runs `SDPAOpt`
(`sdpa_ocl` on its lane) unless `SDPAOpt` rejects the shape. Measured with cliloader per test
(`test/xe3_test/b70_repro/logs/refaudit.txt`): `smoke_sdpa_gpu_test` and
`sdpa_gpu_custom.static_mismatched_kv_head_size_multi_token_uses_sg_scale_factor_2` compared the kernel with itself; the
sliding-window tests (`smoke/sdpa_sliding_window_test`, `causal_true_with_explicit_mask`,
`opt_stage_kernels_apply_sliding_window`) and the dequantized-KV reference of `run_compressed_kv_sdpa_test()` ran the
same kernel on another path (explicit mask; uncompressed KV), so the shared attention math went unchecked. All of them,
and `sdpa_gpu_causal_mask` / `sdpa_gpu_noncausal_tail`, now check against `sdpa_host_reference()` (`ov::reference`, f32,
per head on all host threads; GQA heads repeated). Bounds and B70 maxima: plain f16 4e-3 (1.2e-3), bf16 1.5e-2 (1.2e-3),
compressed per-token f16 1e-2 (1.9e-3), bf16 3e-2 (9.3e-3), per-channel 0.1 (3.5e-2, Q in [-8, 8]). With the reference
perturbed on purpose, the predicted rows failed: no mask 56 of 141 (every row whose mask matters), causal flipped 108,
scale x1.25 every row whose output depends on the scale. The per-token / bf16 compressed decode rows are causal
(upper-left) with one query, so they see key 0 only and check only that key's V. Broadcast, per-channel, GQA
decomposition, transposed-V and the dynamic-V-head tests do run `sdpa_ref`.

The paged-attention harness data (`generate_realistic_data`, N(0, 0.1)) makes a GENERATE output
about 0.08 / sqrt(keys) in size, so the compressed-cache tolerances (0.025 i8, 0.075 u4) accept an
all-zero output from about 150 keys on (u4 from about 20). A missing output write is seen only
by the f16 tolerance (0.002) and by `paged_attention_swa_one_partition_test`, which poisons the
output with NaN before a second run.

The basic paged-attention rows do not observe the bytes the cache writer stores. Two deliberately broken
`pa_kv_cache_update_ref.cl` variants (wrong nibble in the u4 token-major PREFILL helper; wrong slot for the requantize odd
tail) changed the K cache of 13 and 10 rows (and the output of 8 rows in the second case), and all 186 rows still passed.
A writer change needs a byte comparison of the dumped caches ("The u4 token-major K writer (IGC 2.41)").
`smoke_paged_attention_key_pad_cache` compares the stored K cache with its input, for padded key inputs only.

### Performance opportunities

Each needs its own measured change.

- Plain-SDPA f16 `d_max` 128 PREFILL ("Plain-SDPA PREFILL tiling"): a second shape-agnostic `sdpa_ocl` stage with the
  256 x 128 tile, chosen in `SDPAOptImpl::execute()` by the runtime counts, would give dynamic Q + K the 0.75-0.95x of
  long shapes without the small-query loss. The K-read order the non-causal KQ branch gets from IGC ("Plain-SDPA causal
  tile skip") might also be reachable without the branch for the masked stages and other tilings (not tried).
- Plain-SDPA PREFILL with a 2D mask ("Plain-SDPA PREFILL tiling (`d_max` 64)"): an SDPA-backend LLM prompt is causal
  through the mask, so the kernel computes every key tile and streams the mask: llama-3.2-1b at 4K tokens takes 7.9 ms
  per layer against 1.2 ms with `is_causal` (B70, 1800 MHz), about half its first-token latency. Recognising a causal
  mask (or skipping key tiles the mask fills with -inf) would remove most of it. Not tried.
- Plain-SDPA decode ("Plain-SDPA decode tiling"): with few heads at head 256 and up to 4K keys it still trails
  `sdpa_micro` (8 / 4 heads: 1.05-1.19x). There `PREFETCH_V` costs what it gains on long contexts, so a runtime gate on
  the key count (one stage serves every count) might keep both; a key split or one workgroup per KV head (each query
  head's workgroup reads its whole KV head now, as in `sdpa_micro`) would go past `sdpa_micro` on GQA. bf16, compressed
  K / V, `d_max` 32 and Xe3 keep the table tile, unmeasured with this one.
- `sdpa_ocl_decode` checks pages with the strict `% 64` rule. The relaxed `% 16` page rule of the
  MIXED kernel would give block reads to i8 heads 80, 96, 112 and f16 heads 48, 80. For i8 head 96 it
  would not pay: the per-lane K read with `K_LANE_TILES` measured the same as the K block read, and the
  `V_PAGE_1D` read beat the V block read ("Compressed pages block2d does not take"). f16 heads 48 / 80
  are unmeasured.
- An op the DPAS kernels refuse on Xe2 (alibi, scores output, adaptive R-KV, a head size either rejects) runs
  `paged_attention_opt.cl` on the token-major page. Its i8 BY_TOKEN K read there is a per-lane `vload16`, 10% slower
  than the d-major read in `pa_gqa_single_token` at 4K / 16K ("Token-major K page by a static rule (2026-11-06)"); a
  32-bit transpose read like the f16 one (`USE_2D_BLOCK_IO_K_TM`) might recover it. Not tried.
- `sdpa_ocl_decode` against `paged_attention_opt.cl` on the same token-major page (record only, no routing change):
  phi-3.5-mini (32 heads, head 96) GENERATE is faster on the opt kernel. The attention kernel takes 0.59 / 0.71 / 0.79x
  the time with u4 at 2K / 4K / 16K (0.82-1.00x with i8 and f16; per decode token 0.87-0.90x with u4). This already
  held against the d-major opt kernel at 2K / 4K, so it is a weakness of `sdpa_ocl_decode` on that shape, not of the
  page. On llama-3.1-8b (kv group 4, head 128) `sdpa_ocl_decode` wins except with u4 from 4K (0.83 / 0.93x at 4K / 16K)
  and f16 at 16K (0.98x); at 2K the opt kernel is 1.56-2.11x, because its GQA kernel needs a 4096-token context
  (`can_use_gqa_kernel()`). Method: cliloader device time per decode token (128-token runs minus 1-token runs), B70
  pinned at 1800 MHz, two runs in reversed order within 0.01. End to end (`test/run_daily_bench.py`, its 20 models other
  than the diffusion ones, default and 2K-128K prompts, the two arms back to back alternating which goes first, B70
  pinned at 1800 MHz; 4-bit models, so a u4 cache), second-token latency of the opt kernel against `sdpa_ocl_decode`:
  phi-3.5-mini and phi-3.5-vision 0.85-0.91 from 2K; llama-3.2-1b (kv group 4, head 64) 0.94 / 0.91 / 0.86 at 8K / 16K
  / 32K; qwen3-vl-4b 0.96 at 4.9K; llama-3.1-8b, qwen3-8b and mistral-7b 1.07-1.09 at 2K (below the GQA kernel's 4096),
  then 0.97-1.01; phi-4-mini 1.02 / 1.05 at 2K / 4K, then 0.97-0.99; llama-2-7b (32 heads, head 128, prompts up to 2K)
  1.01-1.02, so phi-3.5's gain does not carry over to every model with one KV head per query head. `sdpa_ocl_decode` is
  ahead on gpt-oss-20b (1.08 / 1.16 / 1.30 at 8K / 16K / 32K), gemma-4-26b (1.08 at 32K, 1.12 at 128K), gemma-4-e2b
  (1.11 at 32K), qwen3.6-35b (1.06 at 32K, 1.12 at 128K), minicpm4-8b (1.02-1.10) and qwen3.8-27b (1.05 at 32K);
  gemma-2-9b, gemma-3-4b and qwen3.5-9b are within 0.99-1.02 (gemma-2-9b 1.06 at 2K). The first-token latency, the
  control, was within 1.5% except phi-3.5-mini's 32-token prompt (0.96). Not measured: batch > 1, i8 and f16 caches end
  to end. Routing head 96 or long contexts to the opt kernel, or fixing `sdpa_ocl_decode` there, was left open.
  Since the per-lane page reads ("Compressed pages block2d does not take", 2026-11-06; same cliloader method), the opt
  kernel takes 1.00 / 1.18 / 1.17x the attention time of `sdpa_ocl_decode` on phi-3.5-mini with u4 at 2K / 4K / 16K,
  1.26-1.32x with i8 BY_CHANNEL and 1.31-1.39x with i8 BY_TOKEN; f16, block2d before and after, stays at 0.87 / 0.99 /
  1.00x. llama-3.2-1b (u4): 1.71 / 1.05 / 1.01 / 1.07x at 2K / 8K / 16K / 32K. End to end the opt kernel is now
  1.00-1.15 on phi-3.5-mini / vision and 1.00-1.09 on llama-3.2-1b (second-token latency, same method).
- The MIXED K block reads use only the even entries of `k_page[]`, and the per-k0 K hoists sit in up
  to three separate `if (from_cache)` blocks.
- The single-element runtime mask is re-read for every logit, plain SDPA takes `d` as a runtime
  argument although `K_HEAD_SIZE` is jitted, and the u4 V gather selects the nibble with `?:`
  where a shift would do.
- The S*V block trim past `k_chunk` and the Vc prefetch are u4 only (`IS_PA_K_U4 &&
  PA_CUR_KV_F16`) and were never measured on i8 BY_CHANNEL MIXED.
- `sdpa_ocl_decode` compiles one `SG_PER_WG`, but which value wins depends on the partition count. With
  an i8 cache 16 subgroups took 0.85-0.89x the GENERATE device time of 8 at 1-5 partitions and
  1.26-1.29x at 16-17 (heads 64 and 128, B70), so i8 keeps 8 and gives up the short-context gain.
  Building both variants and choosing per dispatch from `num_of_partitions` against the device's
  thread count would keep both. The same holds with the per-lane page reads (probe, 2026-11-06): u4
  head 64 at kv group 4 (llama-3.2-1b) is 0.81-0.88x with 16 at 300-2.3K keys and 1.01-1.16x from 4.3K,
  so it keeps the 16 of the u4 rule; head 96 MHA loses 6-9% with 16 at 1.1K only.
- `PREFETCH_V` at `d_max` <= 256 still asks for each 256 B L3 line up to four times (a 64 B wide block prefetch).
  `PREFETCH_V_L3` in its place lost 1-5%, but that also stopped filling L1; one request per line into L1 and L3
  (`L1C_L3C` at a 256 B stride) was measured at `d_max` 512 only (+1.2% / -8.9% at 8K / 32K there).
- `sdpa_ocl_mixed` with a u4 cache at `d_max` 64 (table tiling) spills under IGC 2.41: llama-3.2-1b reports
  `REG128 SPILL=448` (384 before the V page indices were hoisted; cliloader, B70). With GenAI's continuous-batching
  default scheduler (256-token chunks) every chunk after the first runs this kernel, so it carries the prefill there.
  Whether it spilled under IGC 2.36, and what the spill costs, were not measured. The `d_max` 512 MIXED tile spills
  1,920 B at 256 GRF. Small steps (at most 16 new tokens) no longer reach it on the token-major page: a llama-3.2-1b
  draft model's MIXED steps took 310 / 593 / 1148 us per call on it at 2K / 4K / 8K context and take 26 / 30 / 51 us on
  the GQA fallback ("Small GQA MIXED step on the token-major page (2026-11-08)"); record only.
- MIXED sliding-window layers with a short window lose to `sdpa_micro` per call: gpt-oss-20b's 128-key window at `d_max`
  64 with every cache (f16 52 against 44 us, i8 61 against 55, u4 69 against 57; 8K, 256-token chunks) and gemma-4-e2b's
  512-key window at `d_max` 256 with f16 (63 against 61). Not investigated ("Paged-attention MIXED: i8 and f16 caches").
- The u4 token-major K writer's head-256 GENERATE dispatch is still 3-4% slower than master's d-major writer
  (gemma-4-26b 9.5 against 9.2 us per call): each thread writes 8 bytes into 16 token rows, i.e. 16 different 128 B
  lines, where a d-major thread writes 2 ("The u4 token-major K writer (IGC 2.41)").
- Small GQA MIXED steps (record only, 2026-11-08; "Small GQA MIXED step on the token-major page (2026-11-08)"):
  `pa_small_q_opt.cl` leaves the GPU mostly idle with few new tokens or a short context (a work-group per KV head and
  256-key partition with ceil(tokens / TOKENS_PER_SG) subgroups, each dequantizing every K/V tile into SLM: 32
  work-groups of 2 subgroups at 4 new tokens and 1K context, 32/8 heads), which is why the fallback keeps that region; a
  split of a partition's tiles over more subgroups, or a dequant whose cost does not depend on the token count, would
  let the DPAS kernel take it. `sdpa_ocl` has no split over keys, so its few-query MIXED time grows with the context
  whatever the token count (f16 480 us at 8K for 2-16 new tokens with the KV out of L2, 32/8 heads, head 128); a few-query tiling with a key
  split would serve every cache kind, token_type_ids included. Neither was profiled.
- PA PREFILL at 32 tokens is still slower than `sdpa_micro` per call after the causal tile skip: qwen3-8b 14.1 against
  12.6 us (1.12x), phi-3.5-mini 15.8 against 12.8 (1.23x), and at `d_max` 64, where the skip is off, llama-3.2-1b 13.1
  against 9.2 (1.42x) ("Paged-attention PREFILL: short prompts"). From 1K they are level or faster.
