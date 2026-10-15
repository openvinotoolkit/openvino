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
| SDPA prefill and single-token | `SDPAOclGenerator` | `sdpa_ocl.cl` | `sdpa_opt.cpp`: `SDPAOpt::supports_micro_sdpa()` and `SDPAOclGenerator::supported()` |
| PA PREFILL / MIXED | `SDPAOclGenerator(prefill)` | `sdpa_ocl.cl` | `paged_attention_opt.cpp`: `choose_dpas_backend()` at construction, `can_use_micro_sdpa_for()` per dispatch |
| PA GENERATE | `SDPAOclDecodeGenerator` | `sdpa_ocl_decode.cl` | `SDPAOclDecodeGenerator::supported()` |

Which DPAS kernel plain SDPA and PA PREFILL/MIXED use is decided per device,
`paged_attention::sdpa_ocl_selected()`: `sdpa_ocl` on Xe2+ XMX, `SDPAMicroGenerator` (`sdpa_micro`)
on the other XMX parts (xe_hpg: DG2/Arc-A, ARL-H) as upstream, and everywhere with
`TEST_USE_SDPA_OCL=0`. There is no fallback from one to the other: where the device's kernel
refuses an op, the opt kernels run. On Xe2 that keeps, for example, f32 PA and EAGLE3's qq_bias MIXED
on the opt kernels even where `sdpa_micro` could serve them; switching kernels per op would change
Xe2 behaviour and needs its own measurement. Both generator types are created in every impl's
default constructor, the params constructor adds one of them, and every later decision reads
`has_stage()`: a loaded or cloned impl is default-constructed and only gets `_order` (indices into
`_stages`) back, so it must not re-derive the kernel from the device or the environment.

The `sdpa_micro` lane keeps upstream's limits: k == v, no single-element runtime mask, no unaligned
single-token (a static unaligned decode runs the opt multi-tokens kernel), ARL-H static decode on
the opt kernel, and the xe3p workarounds (plain SDPA head <= 64, PA always) -- in the kernel choice
only, `SDPAOpt::validate_impl()` does not apply them. It also refuses an f32 query or output,
because `sdpa_micro.cl` loads Q as packed halves and stores half; its codegen happens to fail on
f32 too ("No matching kernel", swallowed by `add_stage()`). Two gaps are upstream's as well:
`sdpa_micro` MIXED has no bidirectional token_type_ids mask, so that combination runs
`pa_multi_token`, which ignores the mask too (upstream dispatched `sdpa_micro` there), and an empty
token_type_ids tensor is honoured only by `sdpa_ocl`.

`TEST_USE_SDPA_OCL_DECODE=0` disables `SDPAOclDecodeGenerator`, so GENERATE falls back to the
`paged_attention_opt` single-token kernels. Either switch also keeps the i8/u4 BY_CHANNEL K cache
d-major, the page those fallbacks read ("Paged-attention cache layouts").
`SDPAOclGenerator::supported()` requires Xe2 or later (xe_hpg only during the bring-up, below) and
f16/bf16 Q and output. Plain SDPA
additionally requires equal K and V head sizes (`SDPAOpt::supports_micro_sdpa()`), paged
attention does not.

Paged attention adds the PREFILL and MIXED stages together, so both variants are compiled even
when only one is dispatched. `can_use_micro_sdpa_for()` keeps MIXED off `sdpa_ocl` for cache
layouts its dequant cannot read (d-major BY_CHANNEL, INT4 without the token-major page);
those compile but fall back to `pa_multi_token`.

Reachable configurations on Xe2 with XMX and no environment overrides:

| Kernel | Default | With user settings | Compiled, not dispatched |
|---|---|---|---|
| `sdpa_ocl` | PA PREFILL f16 (MICRO_MATH=1); PA MIXED i8/u4 BY_CHANNEL token-major with raw f16 Kc/Vc; plain SDPA f16 prefill and single-token | PA MIXED f16 and i8 BY_TOKEN d-major; plain i8 KV compression (always asymmetric + planar); plain bf16; token-major f16/i8 BY_TOKEN (`OV_GPU_PA_K_TOKEN_MAJOR=1`) | d-major BY_CHANNEL MIXED, plain int4 |
| `sdpa_ocl_decode` | PA GENERATE i8/u4 BY_CHANNEL token-major | f16 and i8 BY_TOKEN (`OV_GPU_PA_K_TOKEN_MAJOR=1`) | — |

### xe_hpg bring-up (DG2, SG8)

xe_hpg (SG8, no 2D block IO) can take the `sdpa_ocl` lane only with `TEST_USE_SDPA_OCL_HPG=1`; without
it nothing about xe_hpg changes. Only the kernel arms that exist are served: `supported()` requires the
op's `HpgTier` bits (`sdpa_ocl_hpg.hpp`, `hpg_tier_required()`) to be set in `kHpgTiersReady`, which
has `PLAIN_F16_STATIC` (plain f16 SDPA, static shape, more than one query, no mask / causal / sink / runtime
scale) and `PLAIN_EXT` (the rest of uncompressed plain SDPA: bf16, a mask, causal, a sink, a runtime scale, a dynamic
shape, one query; the mask, causal, sink and softmax code is lane = query and shared with SG16, only the 16-key
`mask_full` vector of a full 2D mask is wider than the subgroup). Every other op is refused and, while the bring-up lasts (`TEMP(S9)` in `sdpa_opt.cpp` and
`choose_dpas_backend()`), goes to `sdpa_micro` as it does with the switch off. The SG8 arm (`SG8` in
`sdpa_ocl_config.cl`, i.e. `SUBGROUP_SIZE == 8`) feeds the DPAS an `int8` A operand, two halves per lane
(lane l = reduction elements 2l, 2l+1): the K tile is read by `k_tile_dword` (a dword block read per key row, a
two-ushort fallback for an unaligned or tail tile) and the S*V A operand is one 8-row dword block read of
`S_slm`; everything else (Q staging, softmax, V gather, output) is lane = query / value column and is shared with
SG16. `SDPA_OCL_NEG_SG8=1..3` (SG8 only) breaks one of the three mappings on purpose for the sharp-softmax test,
`=4` forces the unaligned-K fallback, `=5` / `=6` read the full / per-key mask with the wrong lane width / one lane off.
`PLAIN_I8` is plain SDPA on an i8 KV cache (always asymmetric, planar): the K tile comes from `k_tile_dword_i8`, which
dequantizes `(q - zp) * scale` per element from the hoisted per-key scale/zp into the same dword A operand (byte
loads, no alignment assumption), and V goes through the shared scalar `v_tile_gather`. A plain int4 cache, and i8 data
without the scale/zp tensors, are refused by `supported()` on xe_hpg. `NEG_SG8=1` swaps the head-dim pair there too.
The K dword read is guarded at run time (`k_dword_ok`: even row pitch and a 4 B aligned base), so a dynamic shape or a
sliced K view needs no host proof. The SG8 host jit is
still generated: the 2D block and 1D page flags are forced to 0 whatever the `SDPA_OCL_*_2D` overrides
say (`block2d_io_allowed()`), the tiling is 16 x 16 keys x queries with 4 x 2 subgroups, 256 GRF is
always requested, and `tiling_fits_device()` (used by both `choose_config()` and `supported()`, so
the `SDPA_OCL_KQ_*` overrides are judged too) keeps local memory within 64 KiB and the work-group
within 1024 work-items. `sdpa_ocl_config.cl` `#error`s on SG8 (`SUBGROUP_SIZE != 16`) at every spot
that is silently wrong there (2D block IO, the compressed and paged-MIXED K/V readers), because DG2
compiles a `short8` DPAS operand without a DPAS.

To look at the SG8 jit and compile it offline without a DG2: build with `ENABLE_DEBUG_CAPS`, run a
test group with `OV_GPU_ARCH_OVERRIDE=xe_hpg TEST_USE_SDPA_OCL_HPG=1 SDPA_OCL_HPG_TIERS=all` and
`OV_GPU_DUMP_SOURCES_PATH` (`test/sdpa_ocl_gtests.sh dump`), then `sdpa_ocl_ab.py corpus` and
`sdpa_ocl_ab.py hpg --device dg2 --grf256` (the `dpas` column must be nonzero for every in-tier kernel: DG2
drops a SG16 DPAS silently). The forged arch makes every kernel choice believe in xe_hpg,
so the results of such a run mean nothing; only the dumped sources do.

## Source layout

| File | Contents |
|---|---|
| `impls/ocl_v2/sdpa_ocl.cl` | The kernel: extension pragmas, batch-header includes, the helper-header includes and the kernel body |
| `impls/ocl_v2/sdpa_ocl_config.cl` | Every compile-time macro: dtype and DPAS selection, derived tiling, DKS, mask/bidir predicates, PA page geometry, u4 labelling, 1D page mapping, expression macros, and all host/kernel invariants as `#error` |
| `impls/ocl_v2/sdpa_ocl_mask.cl` | Attention-mask helpers: the bidirectional image-group boundary scans and the full 2D mask tile |
| `impls/ocl_v2/sdpa_ocl_qk_load.cl` | Q staging, the per-k0 K hoists (plain comp, PA pages, PA comp, u4 1D page) and the K tiles (block reads, Kc, gathers) |
| `impls/ocl_v2/sdpa_ocl_v_load.cl` | V page base, Vc prefetch and the V tiles (block reads, comp fold and dequant, Vc, gathers) |
| `impls/ocl_v2/sdpa_ocl_decode.cl` | The decode kernel (self-contained) |
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
  (7680 -> 7552 bytes). Scratch arrays (`kt`, `kw`, `vt_pa`, `vzp4`, `zpb4`, `v_pg`) are declared by
  the caller and passed as `__private` pointers.
- **Loop-carried outputs belong to the caller.** `q_pack` is written element-wise by the u4 staging;
  as the caller's loop variable SROA keeps its previous value as a phi, which a helper-local vector
  dropped (register allocation moved on u4 head 512). It is an out-parameter.
- From the start: arguments keep the caller's types, unrolled trip counts are macros (a runtime or
  select-based bound, or an early `return` in a helper, lost the unroll in
  `pa_kv_cache_update_ref.cl`), private arrays are indexed only by unrolled constants, pointer
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
  MIXED fallback, `pa_multi_token`, reads a token-major BY_CHANNEL K cache d-major, so a rejected
  shape produced NaN rather than a slower result. Tier 2 is unreachable for `k_head_size ==
  v_head_size`.
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
key tile accumulated from zero and added afterwards (`A_tile1`), as `ugemm_vs` does.

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

For `CAUSAL_MASK_LOWER_RIGHT` (plain SDPA) every query moves by `causal_offset = max(0, k - q)`,
the shift `sdpa_micro` / reference / `sdpa_opt` apply. Without it a single new token (q == 1,
k == 512) would only see the first tile of keys.

### Paged-attention MIXED: current tokens from Kc/Vc

With `PA_CUR_KV_F16` the MIXED kernel reads keys `[0, past_len)` from the paged cache and the new
keys `[past_len, k)` from the raw f16 K/V inputs (`Kc`/`Vc`). A compressed cache is a lossy round
trip, so reading the new rows back from it is not equivalent. A key tile that crosses `past_len` is
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

## Paged-attention cache layouts

A K cache page is either d-major (`[head_size, block_size]`, the upstream layout) or token-major
(`[block_size, head_size]`, the same as V). Token-major is selected by
`paged_attention::k_token_major_for()` (`OV_GPU_PA_K_TOKEN_MAJOR`) for f16 and i8 BY_TOKEN, and by
the separate staging switch `paged_attention::k_by_channel_token_major_for()`
(`OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR`, on by default) for i8/u4 BY_CHANNEL. The MIXED kernel reads
d-major pages only through a per-key scalar gather; token-major pages can use block reads.
`PA_K_TOKEN_STRIDE` / `PA_K_HIDDEN_STRIDE` let one gather path address both.

The token-major BY_CHANNEL page has four readers: the cache writer, rotate, `sdpa_ocl` (MIXED) and
`sdpa_ocl_decode` (GENERATE). Every other one -- `pa_single_token`, `pa_gqa_single_token`,
`pa_multi_token`, `sdpa_micro` MIXED, `pa_kv_reorder`, adaptive R-KV -- reads it d-major. The
layout is decided once per model in `transformations_pipeline.cpp`, the reader per dispatch, so
the page is only token-major where `paged_attention::by_channel_token_major_readable()` says both
token-major readers get chosen for every paged-attention op: XMX on Xe2 or later, microkernel
support, a oneDNN build, both `TEST_USE_SDPA_OCL*` switches on, f16 inference, and per op no alibi,
scores output, adaptive R-KV or qq_bias, head sizes that are multiples of 16 and at most 512, and
an `sdpa_ocl` tiling when k != v. The rules mirror `supports_micro_sdpa()` /
`can_use_micro_sdpa_for()` and `SDPAOclDecodeGenerator::supported()` and must change with them.
Everything else keeps the d-major page, which every fallback reads, so a mismatch only costs
speed in that direction. The other direction is caught: `PagedAttentionOptImpl::update_rt_params()`
throws when a d-major reader is chosen for a token-major page, and `pa_kv_reorder` refuses one.
qq_bias is on the list only because its one user, EAGLE3, reorders the cache from a separate
kv-update model through `pa_kv_reorder`; `smoke_qq_bias_token_major` keeps the `sdpa_ocl` qq_bias
paths covered on a forced token-major page.

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
  takes a PREFILL-only case down with it. The dequant is only correct for i8 BY_TOKEN and the
  token-major BY_CHANNEL layouts (i8 and u4); d-major BY_CHANNEL and the other INT4 caches compile
  and are kept off dispatch by `can_use_micro_sdpa_for()`.
- u4 is recognised from the configured kv-cache precision: the tensor is u8 (an i4 cache is i8),
  so its layout type cannot tell. u4 only, not i4: the int4 quantizer clamps to [0, 15], so u4
  nibbles are unsigned, while i4 would need a signed widen that nothing implements.
- Whether BY_CHANNEL was relayed token-major is read from the physical cache shape (the adjusted
  block size sits at dim 2), the single decision made in `transformations_pipeline.cpp`.
- BY_CHANNEL's per-channel scale and zero point fold as a per-lane scalar. The KQ A operand is K,
  so the lane index is the head dim, which is what BY_CHANNEL indexes; BY_TOKEN's are per key,
  i.e. per element within a lane, and need a broadcast each. This is the mirror of
  `sdpa_ocl_decode`, where Q is the A operand and the fold lands on Q.
- `IS_PA_K_TOKEN_MAJOR` is jitted even when no K block read is enabled, because the gather
  fallbacks address the page through `PA_K_TOKEN_STRIDE` / `PA_K_HIDDEN_STRIDE`.

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

## Dequantisation

Paged-attention caches: `(q - zp) * scale`, in half. The cache writer stores `1 / scale`, so the
value read back is already the multiplier. The block-read and scalar paths use exactly the same
arithmetic, which makes the `*_PA_I8_2D` and `*_PA_1D` switches clean bisection toggles.

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
  `sdpa_micro.cl` and `sdpa_opt.cl` omit the subsequence bump of the buffer, which only agrees for a
  single subsequence starting at token 0.
- The causal bound and the window start are extended to cover the groups that straddle them.
  Groups never leave the new-token region, and reading a future key from the cache relies on
  `pa_kv_cache_update` running before this stage, as the reference does.
- The mask loop needs only per-query group bounds `[begin, end)`, computed once per workgroup;
  `sdpa_micro` instead re-scans `token_type_ids` per (query, key) pair.
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
  tested, and "the largest M that does not spill" reproduced the measured optimum.
- `live_grf_estimate()` counts the arrays that stay live across the KQ loop. It was calibrated on
  gemma-4 (u4 BY_CHANNEL, M in {1, 2, 4, 8} x `SG_PER_WG` in {8, 16}, head 512 and 256) and tracks
  spill volume monotonically (head 512 at `SG_PER_WG` 8 scores 136/164/220/332 against 0/6400/15936/
  34432 bytes), but it cannot resolve differences under about 15 GRF: (8, M = 1) scores 136 and does
  not spill, (16, M = 2) scores 126 and spills 640. So it is a coarse gate, and the budget of 112
  brackets the measured optimum instead of sitting at 128: head 512 at `SG_PER_WG` 16 wants M = 1
  (100) and rejects M = 2 (126, 34% slower), head 256 wants M = 2 (94), and llama's head 128 at
  M = 4 scores 104. `SDPA_OCL_DECODE_M` bypasses the cap but not the local-memory clamp (half the
  arena), which guards the build.
- `SG_PER_WG` is 16 when there are at least 16 V head-dim tiles and 8 otherwise. One subgroup is
  one thread, and at 8 the kernel had a quarter of `pa_sdpa_opt`'s threads per Xe core: on gemma-4
  at M = 1, 16 was 2.07x faster at head 512 and 1.26x at head 256, and 4 was 2.2x slower. With
  fewer V tiles the surplus subgroups split the key axis instead, which adds the `slm_out` staging
  and a barrier; that is why an earlier sweep found 16 neutral or worse at head 128. `SG_PER_WG`
  must divide `SEQ_LEN_PARTITION_SIZE / SUBGROUP_SIZE` and not exceed the subgroup size.
- The cache is f16, i8 or u4. i8 only among the 8-bit types: the dequant widens the byte as signed,
  as `kv_cache_update` wrote it, so a u8 cache would decode with the wrong sign. i4 is rejected. qq_bias
  is accepted: one new token per sequence makes the tree mask the 1 x 1 identity, which plain causal
  masking already gives.
- `SDPA_OCL_DECODE_256GRF=1` is for the larger M values: at M = 8 and head 128 the live set is about
  110 GRF (the S*V accumulator alone is `V_TILES` x float8 = 64). Read SPILL= from a runtime
  cliloader line; ocloc mispredicts it.
- The key axis is split into partitions of `SEQ_LEN_PARTITION_SIZE`, and the kernel writes
  `pa_sdpa_finalization_stage`'s intermediates unchanged (see the contract at the top of the file).
- S*V splits the head dim, not the keys: a subgroup that owns a head-dim tile reduces over every
  key and produces a final result. The earlier key split left every subgroup holding a partial over
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

## Debug and bisection switches

All are read on the host when the kernel is compiled.

| Variable | Effect |
|---|---|
| `TEST_USE_SDPA_OCL`, `TEST_USE_SDPA_OCL_DECODE` | `0` selects the micro / opt path instead, and keeps the i8/u4 BY_CHANNEL K cache d-major for it |
| `SDPA_OCL_KQ_TILE_KEYS`, `_TILE_QUERIES`, `_PER_WG_KEYS`, `_PER_WG_QUERIES` | Override the KQ tiling (`kq_sg_tile_keys` must be 16 or 32) |
| `SDPA_OCL_TRACE_CONFIG`, `SDPA_OCL_TRACE_STAGE` | Print the chosen tiling / PA stage |
| `SDPA_OCL_256GRF`, `SDPA_OCL_DECODE_256GRF` | Large-GRF compile (`SDPAOclGenerator` always on xe_hpg) |
| `TEST_USE_SDPA_OCL_HPG` | `1`: xe_hpg may take the `sdpa_ocl` lane (default off; see "xe_hpg bring-up") |
| `SDPA_OCL_HPG_TIERS` | `all` or a comma list of `HpgTier` names: pretend those xe_hpg tiers are ready (dump only) |
| `SDPA_OCL_NEG_SG8` | xe_hpg (SG8) only: `1` swaps the K pair order, `2` reads the S*V A operand transposed, `3` swaps the S_slm pair order (each must fail the sharp-softmax test); `4` forces the unaligned-K fallback (must pass) |
| `OV_GPU_ARCH_OVERRIDE` | Debug caps only: report another arch (`xe_hpg`, `xe2`, ...) so its host code runs on this device (dump only) |
| `SDPA_OCL_Q_2D`, `_KV_2D`, `_A_2D`, `_K_I8_2D`, `_V_I8_2D` | Plain-SDPA block IO paths |
| `SDPA_OCL_K_PA_2D`, `_V_PA_2D`, `_K_PA_I8_2D`, `_V_PA_I8_2D`, `_K_PA_1D`, `_V_PA_1D` | Cache page read paths (`0` = scalar gather, same dequant) |
| `SDPA_OCL_PA_CUR_F16` | MIXED current tokens from Kc/Vc (`0` = from the cache) |
| `SDPA_OCL_MICRO_MATH` | `0` disables `sdpa_micro` rounding on PA prefill |
| `SDPA_OCL_BIDIR`, `SDPA_OCL_BIDIR_GATE` | Negative controls for the image-group mask and its empty-buffer gate |
| `SDPA_OCL_DECODE_M`, `_SG_PER_WG`, `_PREFETCH`, `_K_2D`, `_V_2D` | Decode tuning and block IO paths |
| `OV_GPU_PA_K_TOKEN_MAJOR`, `OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR` | K cache layout staging switches |
| `OV_GPU_PA_K_TM_BREAK=<consumer>` | Forces one consumer back to the d-major read, to test whether a suite observes it |

The negative controls must fail the token-type suites when disabled; a pass means the suite does
not observe the feature.

## Known issues

Found in the 2026-09 review of these files and deliberately left out of the refactor, because each
fix changes behaviour or lies outside them. Everything here comes from code reading unless a test
result is quoted.

### Correctness

- Compressed BY_CHANNEL with f32 or ACCURACY (dynamic) inference precision:
  `transformations_pipeline.cpp` sizes the per-channel comp by the inference precision (f32: i8
  page row 24, u4 16; dynamic has size 0: 16 and 8), while `graph/paged_attention.cpp` and
  `ops/paged_attention.cpp` assume f16 comp (20 / 12), so the block-size assert should fire at run
  time on either layout. Master has the same code. From code reading only.
- `pa_kv_cache_update_ref.cl`, `quantize_and_save_by_channel_block_with_requantize`, steps to the next new
  token with the unpadded stride `K_HEAD_SIZE * KV_HEADS_NUM` and never uses its `in_data_pitch` argument, so a
  key input with feature padding is misread for the second and later new tokens of a partially filled
  i8 BY_CHANNEL block (a MIXED step that continues a prompt at `past_len % 16 != 0`). Found while testing the
  block2d gate: the `paged_attention_feature_pad_test` compressed case therefore pads Q and V only.
- `sdpa_opt.cl` types a paged-attention runtime scale as `INPUT3_TYPE` (`SCALE_TYPE` follows
  `HAS_ATTN_MASK_INPUT`, which the paged-attention generator never sets, and INPUT3 is a 32-bit index
  input there), so `pa_sdpa_opt` PREFILL and MIXED read the 16-bit scale as an int32. Real models give a
  constant scale and never reach it. `paged_attention_runtime_scale_test` is skipped below Xe2 because of it.
- `sdpa_micro` still jits `SCALE_DATA_T` as `half` (`sdpa_gen_micro.cpp`, `sdpa_micro.cl`), so a bf16 or
  f32 runtime scale input would be misread there. `sdpa_ocl` types the scale from its layout.
- `SDPA_OCL_KQ_TILE_KEYS=32` gives wrong results when `kq_sg_per_wg_keys >= 4` and a subgroup has
  two query blocks (`kq_sg_tile_queries = 32`): deterministically, the first query block is right
  and the second is not. It survives every memory-path switch and `SDPA_OCL_256GRF=1`, so the fault
  is in the indexing rather than the loads, and it has not been found. The tuned tables always use
  16 keys, so only the override reaches it (last reproduced 2026-09-10 on
  `paged_attention_test.basic/31`).

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
- The duplicate-macro asserts in `common_utils/jitter.hpp` (`register_macro()` /
  `unregister_macro()`) are commented out on this branch. Restored, they let a Debug build catch a
  jit constant emitted twice.

### Test failures that predate the refactor

Measured on an Arc Pro B70 (Xe2), identical before and after the refactor:

- `SDPAWithKVCacheTest.MultipleIterationStateful`, f16 with compressed KV at head 512
  (`..._et=f16_num_iter=5_num_groups=4_..._compressed=1k_head=512v_head=512`): 4724 of 8192
  elements wrong, max difference 0.99. The kernel compiled for it is the plain i8 KV-compressed
  `sdpa_ocl` at head 512 with K/V block IO, which makes it the prime suspect. Not investigated.
- The 14 bf16 compressed cases of the same test fail in `add_required_reorders` (no i8 layout for
  `dynamicquantize`) before an SDPA implementation is chosen, so plain bf16 with compressed KV is
  unreachable. `SDPAFusion.Inference/0` does not find the fused SDPA node.

### Test coverage gaps

No test reaches paged-attention bf16, qq_bias with an f16, u4 or
BY_TOKEN cache, a compressed cache with token_type_ids, or the u4 scalar page arms (head 48, 96,
112). `sdpa_ocl_decode` reaches `Q_PER_WG = 8` only under `SDPA_OCL_DECODE_M=8` (the GRF cap limits
the test shapes to M <= 4), and even then no GENERATE test uses u4, so the u4 M = 8 kernel is only
compiled. Lower-right causal masking is checked only by `sdpa_gpu_causal_mask` (cosine >= 0.99),
and several suites check only finiteness, cache contents or cosine >= 0.95.

The paged-attention harness data (`generate_realistic_data`, N(0, 0.1)) makes a GENERATE output
about 0.08 / sqrt(keys) in size, so the compressed-cache tolerances (0.025 i8, 0.075 u4) accept an
all-zero output from about 150 keys on (u4 from about 20). A missing output write is seen only
by the f16 tolerance (0.002) and by `paged_attention_swa_one_partition_test`, which poisons the
output with NaN before a second run.

### Performance opportunities

Each needs its own measured change.

- `sdpa_ocl_decode` checks pages with the strict `% 64` rule. The relaxed `% 16` page rule of the
  MIXED kernel would give block reads to i8 heads 80, 96, 112 and f16 heads 48, 80.
- A model that `by_channel_token_major_readable()` turns down on Xe2 (alibi, qq_bias, a head size
  either reader rejects) keeps the d-major BY_CHANNEL page, so MIXED runs `pa_multi_token` and
  GENERATE `pa_single_token`. `sdpa_micro` could read that page for MIXED in the qq_bias case (it
  has the MIXED tree mask), but on Xe2 its stage is never built (one DPAS kernel per device); alibi
  and k != v have no `sdpa_micro` path either. Teaching `pa_kv_reorder` the token-major page (K, and
  the split u4 V page) would lift the qq_bias case, i.e. EAGLE3.
- MIXED looks each V page up twice per S*V key block (`pa_v_page_base()` for the comp and again
  for the data), the K block reads use only the even entries of `k_page[]`, and the per-k0 K hoists
  sit in up to three separate `if (from_cache)` blocks.
- The single-element runtime mask is re-read for every logit, plain SDPA takes `d` as a runtime
  argument although `K_HEAD_SIZE` is jitted, and the u4 V gather selects the nibble with `?:`
  where a shift would do.
- The S*V block trim past `k_chunk` and the Vc prefetch are u4 only (`IS_PA_K_U4 &&
  PA_CUR_KV_F16`) and were never measured on i8 BY_CHANNEL MIXED.
