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
| `sdpa_ocl` | PA PREFILL f16 (MICRO_MATH=1 except `d_max` 512); PA MIXED i8/u4 BY_CHANNEL token-major with raw f16 Kc/Vc; plain SDPA f16 prefill and single-token | PA MIXED f16 and i8 BY_TOKEN d-major; plain i8 KV compression (always asymmetric + planar); plain bf16; token-major f16/i8 BY_TOKEN (`OV_GPU_PA_K_TOKEN_MAJOR=1`) | d-major BY_CHANNEL MIXED, plain int4 |
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

### Paged-attention PREFILL tiling

PA PREFILL on Xe2 at `d_max` 64, 128, 256 and 512 does not use the table row: `solve_pa_prefill_tiling()`
gives it the workgroup of `sdpa_micro`'s PA prefill config for that head size, compiled with 256 GRF (one
workgroup per Xe-core). MIXED has its own tiles at `d_max` 128, 256 and 512 ("Paged-attention MIXED tiling");
plain SDPA, `d_max` 32 and xe_hpg keep the table. The PA query-block stride
follows the stage, which is why `get_query_block_size()` takes it.

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

Within the diagonal tile, PA PREFILL also skips the KQ of a subgroup whose keys all follow its queries and the S*V
blocks after a subgroup's score rows (`CAUSAL_TILE_SKIP`, "Paged-attention PREFILL: short prompts").

For `CAUSAL_MASK_LOWER_RIGHT` (plain SDPA) every query moves by `causal_offset = max(0, k - q)`,
the shift `sdpa_micro` / reference / `sdpa_opt` apply. Without it a single new token (q == 1,
k == 512) would only see the first tile of keys.

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

**f16** (`OV_KV_CACHE_PRECISION=f16`, never a default): the K page is d-major, `[head_size, 16]`, a 32-byte row per
channel under the block2d minimum, so MIXED gathered K one key per message (qwen3.5-9b: 1,465 us against `sdpa_micro`'s
522). A token-major page (`OV_GPU_PA_K_TOKEN_MAJOR=1`) took 441 us, but changes the layout for every reader of the page
("Paged-attention cache layouts"). `pa_k_tile_dt()` (`USE_2D_BLOCK_IO_K_PA_DT`) reads the d-major page in place instead:

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

## Paged-attention cache layouts

A K cache page is either d-major (`[head_size, block_size]`, the upstream layout) or token-major
(`[block_size, head_size]`, the same as V). Token-major is selected by
`paged_attention::k_token_major_for()` (`OV_GPU_PA_K_TOKEN_MAJOR`) for f16 and i8 BY_TOKEN, and by
the separate staging switch `paged_attention::k_by_channel_token_major_for()`
(`OV_GPU_PA_BY_CHANNEL_TOKEN_MAJOR`, on by default) for i8/u4 BY_CHANNEL. The MIXED kernel reads a
d-major f16 page with a 32-bit transpose read ("Paged-attention MIXED: i8 and f16 caches") and a d-major i8 BY_TOKEN page
through a per-key scalar gather; token-major pages use block reads.
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

The f16 d-major page lands the same labelling from a 32-bit transpose read (`PA_K_PERMUTED`, "Paged-attention MIXED: i8
and f16 caches"), so it shares the Q staging and the Kc reads.

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
  4K, against 0.85-0.89x at 1K), and head 96 measured mixed, so they keep 8. `SG_PER_WG`
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
| `SDPA_OCL_256GRF`, `SDPA_OCL_DECODE_256GRF` | Large-GRF compile (`SDPAOclGenerator` always on xe_hpg; `SDPA_OCL_256GRF=0` turns it off for the PA PREFILL tiling) |
| `SDPA_OCL_V_PREFETCH`, `_SM_BRANCHLESS`, `_MAX_SPLIT`, `_MASK_LAST` | PA PREFILL loop changes ("Paged-attention PREFILL tiling"): `0`/`1` forces each off/on |
| `SDPA_OCL_V_PREFETCH_L3` | PA PREFILL V tile L3 prefetch (on at `d_max` 512): `0`/`1` forces it off/on; on replaces `PREFETCH_V` |
| `SDPA_OCL_WIDE_READS` | PA PREFILL wide reads (`d_max` 64: `5`, `0` with a sliding window; 128 / 256: `1`; 512: `7`): bit mask, `1` K pair, `2` V pair, `4` Q_slm as `ulong4`; `0` = narrow |
| `TEST_USE_SDPA_OCL_HPG` | `1`: xe_hpg may take the `sdpa_ocl` lane (default off; see "xe_hpg bring-up") |
| `SDPA_OCL_HPG_TIERS` | `all` or a comma list of `HpgTier` names: pretend those xe_hpg tiers are ready (dump only) |
| `SDPA_OCL_NEG_SG8` | xe_hpg (SG8) only: `1` swaps the K pair order, `2` reads the S*V A operand transposed, `3` swaps the S_slm pair order (each must fail the sharp-softmax test); `4` forces the unaligned-K fallback (must pass) |
| `OV_GPU_ARCH_OVERRIDE` | Debug caps only: report another arch (`xe_hpg`, `xe2`, ...) so its host code runs on this device (dump only) |
| `SDPA_OCL_Q_2D`, `_KV_2D`, `_A_2D`, `_K_I8_2D`, `_V_I8_2D` | Plain-SDPA block IO paths |
| `SDPA_OCL_K_PA_2D`, `_V_PA_2D`, `_K_PA_I8_2D`, `_V_PA_I8_2D`, `_K_PA_1D`, `_V_PA_1D` | Cache page read paths (`0` = scalar gather, same dequant) |
| `SDPA_OCL_K_PA_DT` | MIXED f16 d-major K page transpose read: `16` / `32` rows per read (default 32 with 256 GRF, 16 with 128), `0` = per-key gather |
| `SDPA_OCL_PA_CUR_F16` | MIXED current tokens from Kc/Vc (default for compressed caches only; `0` = from the cache) |
| `SDPA_OCL_MIXED_PREFETCH_V`, `_K` | u4 / i8 MIXED page prefetches ("Paged-attention MIXED tiling", "... i8 and f16 caches"): `0`/`1` forces each off/on where the 8-bit page read is on |
| `SDPA_OCL_CAUSAL_SKIP` | PA PREFILL causal tile skip ("Paged-attention PREFILL: short prompts"): `0`/`1` forces it off/on (default on at `d_max` 128 / 256 / 512) |
| `SDPA_OCL_MICRO_MATH` | `0` / `1` forces `sdpa_micro` rounding off / on for PA prefill (default on except the `d_max` 512 tile) |
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
- The duplicate-macro asserts in `common_utils/jitter.hpp` (`register_macro()` /
  `unregister_macro()`) are commented out on this branch. Restored, they let a Debug build catch a
  jit constant emitted twice.
- `tiling_fits_device()` allows 1,024 work-items per group whatever the GRF mode, but a 256 GRF kernel
  cannot dispatch that many: `SDPA_OCL_KQ_PER_WG_KEYS=16 SDPA_OCL_KQ_PER_WG_QUERIES=4` (64 subgroups) on a
  PA PREFILL tiling compiled with 256 GRF fails at enqueue with `CL_INVALID_WORK_GROUP_SIZE` (B70,
  gemma-4-e2b). The shipped tilings stay at 32 subgroups or fewer, so only an override reaches it.

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

The basic paged-attention rows do not observe the bytes the cache writer stores. Two deliberately broken
`pa_kv_cache_update_ref.cl` variants (wrong nibble in the u4 token-major PREFILL helper; wrong slot for the requantize odd
tail) changed the K cache of 13 and 10 rows (and the output of 8 rows in the second case), and all 186 rows still passed.
A writer change needs a byte comparison of the dumped caches ("The u4 token-major K writer (IGC 2.41)").

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
  thread count would keep both.
- `PREFETCH_V` at `d_max` <= 256 still asks for each 256 B L3 line up to four times (a 64 B wide block prefetch).
  `PREFETCH_V_L3` in its place lost 1-5%, but that also stopped filling L1; one request per line into L1 and L3
  (`L1C_L3C` at a 256 B stride) was measured at `d_max` 512 only (+1.2% / -8.9% at 8K / 32K there).
- `sdpa_ocl_mixed` with a u4 cache at `d_max` 64 (table tiling) spills under IGC 2.41: llama-3.2-1b reports
  `REG128 SPILL=448` (384 before the V page indices were hoisted; cliloader, B70). With GenAI's continuous-batching
  default scheduler (256-token chunks) every chunk after the first runs this kernel, so it carries the prefill there.
  Whether it spilled under IGC 2.36, and what the spill costs, were not measured. The `d_max` 512 MIXED tile spills
  1,920 B at 256 GRF.
- MIXED sliding-window layers with a short window lose to `sdpa_micro` per call: gpt-oss-20b's 128-key window at `d_max`
  64 with every cache (f16 52 against 44 us, i8 61 against 55, u4 69 against 57; 8K, 256-token chunks) and gemma-4-e2b's
  512-key window at `d_max` 256 with f16 (63 against 61). Not investigated ("Paged-attention MIXED: i8 and f16 caches").
- The u4 token-major K writer's head-256 GENERATE dispatch is still 3-4% slower than master's d-major writer
  (gemma-4-26b 9.5 against 9.2 us per call): each thread writes 8 bytes into 16 token rows, i.e. 16 different 128 B
  lines, where a d-major thread writes 2 ("The u4 token-major K writer (IGC 2.41)").
- PA PREFILL at 32 tokens is still slower than `sdpa_micro` per call after the causal tile skip: qwen3-8b 14.1 against
  12.6 us (1.12x), phi-3.5-mini 15.8 against 12.8 (1.23x), and at `d_max` 64, where the skip is off, llama-3.2-1b 13.1
  against 9.2 (1.42x) ("Paged-attention PREFILL: short prompts"). From 1K they are level or faster.
