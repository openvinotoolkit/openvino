#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io               : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                         : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                   : enable

#include "include/batch_headers/sdpa_utils.cl"

float __builtin_IB_atomic_max_local_f32(__local float *, float);

#define kq_wg_tile_keys      (kq_sg_tile_keys * kq_sg_per_wg_keys)
#define kq_wg_tile_queries   (kq_sg_tile_queries * kq_sg_per_wg_queries)
#define kq_key_blocks        (kq_sg_tile_keys / DPAS_ROWS)
#define kq_query_blocks      (kq_sg_tile_queries / SUBGROUP_SIZE)

#define sg_per_wg (kq_sg_per_wg_keys * kq_sg_per_wg_queries)

#define sv_score_blocks      (sv_sg_tile_scores / DPAS_ROWS)
#define sv_value_blocks      (sv_sg_tile_values / SUBGROUP_SIZE)
#define sv_key_blocks        (kq_wg_tile_keys / DPAS_K)
#define q_blocks             (kq_wg_tile_queries / SUBGROUP_SIZE)

// Depth (head-dim) tiles that actually hold data, for the KQ contraction only.
//
// DKS is D_MAX / DPAS_K, and D_MAX is HEAD_SIZE rounded UP to a power of two, so whenever
// HEAD_SIZE is not itself a power-of-two multiple of DPAS_K the top tiles address channels past
// the head dim entirely: they load nothing (the `head < d` guard masks every lane) and their
// dpas adds zero. At HEAD_SIZE 72 -> D_MAX 128, DKS is 8 and only ceil(72/16) == 5 tiles are
// real -- three whole tiles plus half of the fourth are pure waste, 1.78x the necessary KQ dpas
// and K loads. sdpa_micro never pays this because its ugemm_kq takes the reduction extent as a
// RUNTIME argument (k = d), so it loops 5 blocks with remainder handling.
//
// This bounds the KQ depth loop and the Q->SLM staging. It deliberately does NOT touch D_MAX
// itself: the S*V split (sv_sg_tile_values * sv_sg_per_wg_values == d_max) and the alpha-rescale
// nesting are derived from D_MAX, and shrinking it there would break the WG coverage invariants
// documented in choose_config().
// CAVEAT, measured with ocloc on the head-72 prelude (test/splice_head72.sh): shrinking the depth
// loop makes the 2D-block path strictly better (fewer dpas AND fewer messages, spill stays 0) but
// makes the SCALAR fallback path spill MORE, monotonically:
//     DKS_ACTIVE 8 -> instCount 7043, spill 2688     (== what ships today)
//     DKS_ACTIVE 6 -> instCount 5808, spill 13568
//     DKS_ACTIVE 5 -> instCount 5631, spill 16064
// Less work, more spill -- IGC schedules the smaller unrolled body more aggressively and peak
// pressure goes up. Which of the two wins on the scalar path is not decidable from static counts,
// so USE_DKS_ACTIVE exists to A/B it per config with no rebuild (SDPA_OCL_DKS_ACTIVE=0). It only
// matters where DKS_FULL < DKS *and* the config still takes a scalar fallback -- head 48 and 96;
// head 64/128 have DKS_FULL == DKS and are unaffected either way.
#if USE_DKS_ACTIVE
#  define DKS_FULL ((HEAD_SIZE + DPAS_K - 1) / DPAS_K)
#  if IS_PA_K_U4
// The u4 depth permutation pairs tiles (2g, 2g+1) over one 32-channel window, so an odd active
// count would leave the last pair half-formed. Round up; DKS is even (D_MAX is a power of two
// >= 32) and DKS_FULL <= DKS, so the rounded value still fits.
#    define DKS_ACTIVE (((DKS_FULL + 1) / 2) * 2)
#  else
#    define DKS_ACTIVE DKS_FULL
#  endif
#else
#  define DKS_FULL DKS
#  define DKS_ACTIVE DKS
#endif
#if DKS_ACTIVE > DKS
#  error "sdpa_ocl.cl: DKS_ACTIVE must not exceed DKS (D_MAX is HEAD_SIZE rounded up, so it cannot)"
#endif
#if DKS_ACTIVE < 1
#  error "sdpa_ocl.cl: DKS_ACTIVE must cover at least one depth tile"
#endif

// Reading the new tokens' K/V from Kc/Vc needs pointers that only exist in the paged-attention
// non-prefill signature, so fold that precondition into the host flag here rather than repeating it at
// each of the five sites below. The host already scopes SDPA_OCL_PA_CUR_F16 to that variant; this makes
// the kernel independent of that invariant, so a stray define is a no-op instead of a build failure.
#if !(IS_PAGED_ATTENTION && !IS_PREFILL)
#  undef PA_CUR_KV_F16
#  define PA_CUR_KV_F16 0
#endif

// Mask-kind predicates. When the host proved the mask shape at compile time
// (MASK_KIND in {0,1,2}) these fold to compile-time constants so IGC drops the
// dead mask branches; MASK_KIND == -1 keeps the original runtime MSK_D2/MSK_D3
// checks. 2 = full 2D [q>1,k>1], 1 = per-key [q==1,k>1], 0 = scalar/broadcast.
#if MASK_KIND == -1
#  define MASK_IS_PER_KEY  (MSK_D2 == 1 && MSK_D3 > 1)
#  define MASK_IS_FULL_2D  (MSK_D2 > 1 && MSK_D3 > 1)
#else
#  define MASK_IS_PER_KEY  (MASK_KIND == 1)
#  define MASK_IS_FULL_2D  (MASK_KIND == 2)
#endif

// Bidirectional attention over image-token groups (gemma-4 and friends). token_type_ids[t] == 1
// marks an "image" token; a maximal contiguous run of them is a group whose members attend to each
// other in BOTH directions, on top of the usual causal + sliding-window region.
//
// HAS_TOKEN_TYPE_IDS declares the kernel parameter and is set by the host for paged attention with a
// token_type_ids input. USE_BIDIR_MASK is the no-rebuild bisection toggle (SDPA_OCL_BIDIR=0) and is
// deliberately a SEPARATE macro: it elides the logic but leaves the parameter declared, so the env
// knob can never desync get_arguments_desc() from the signature. The IS_PAGED_ATTENTION term is
// redundant with the host gate and kept only so the scope of the feature is readable from the .cl
// alone.
//
// PREFILL and MIXED both take this path. token_type_ids covers the NEW tokens only, so it is indexed
// in LOCAL (subsequence-relative, new-token) coordinates while keys/queries run in KEY coordinates
// key = query_position_offset + local -- see the extensions below. PREFILL is the past_len == 0 case
// of the same code. GENERATE never reaches this kernel and would not need it anyway: that stage is
// defined by every subsequence having exactly ONE new token, so a query's image group is the query
// itself and the bidirectional rule degenerates to plain causal.
#define BIDIR_MASK (HAS_TOKEN_TYPE_IDS && USE_BIDIR_MASK && IS_PAGED_ATTENTION)

// Third axis, and the one that makes the feature safe: HAS_TOKEN_TYPE_IDS is decided at COMPILE time
// from an input shape that may still be dynamic, while the op contract is "[B_token | 0]" -- a
// runtime-EMPTY token_type_ids is legal and means "no image tokens". The host cannot re-jit per shape
// (the paged attention impl is shape-agnostic), so it passes the runtime element count as a scalar and
// every read below is gated on it. USE_BIDIR_GATE=0 (SDPA_OCL_BIDIR_GATE) removes only that gate, so
// the empty-buffer case can be A/B'd in one binary. Needs a default because, unlike the macros above,
// it is used in a C expression rather than a preprocessor conditional.
#ifndef USE_BIDIR_GATE
#  define USE_BIDIR_GATE 1
#endif

// Row pitch of a K / V cache page's DATA region, in elements of the cache dtype. That is HEAD_SIZE
// for f16 and i8 -- so those paths preprocess to exactly what they were and the host does not jit
// these at all -- but a u4 page packs two values per byte while its layout dtype is u8, so the pitch
// is NOT derivable from HEAD_SIZE and sizeof() and the host has to supply it:
//   K u4 BY_CHANNEL  exactly HEAD_SIZE/2, deliberately NOT aligned up. 16*(h/2) data bytes + 4*h comp
//                    bytes == 12*h is what makes the token-major page a byte-exact fit into the
//                    allocation the upstream d-major INT4 page already has; aligning the pitch up
//                    would overflow that page whenever h % 32 != 0.
//   V u4 BY_TOKEN    Align(HEAD_SIZE/2, SUBGROUP_SIZE). Aligning is free here because the trailing
//                    comp slack absorbs it (16*PV + 64 == 16*(PV+4)), and it keeps the pitch a
//                    multiple of 16 for every head size.
// Matches sdpa_ocl_decode.cl's K_ROW_ELEMS / V_ROW_ELEMS and the writer's phys_{k,v}_head_size.
#ifndef PA_K_ROW_ELEMS
#  define PA_K_ROW_ELEMS HEAD_SIZE
#endif
#ifndef PA_V_ROW_ELEMS
#  define PA_V_ROW_ELEMS HEAD_SIZE
#endif

// In-page addressing for a paged-attention K cache, in K elements. d-major pages
// ([.., k_head_size, block_size]) make the tokens of one head dim contiguous; token-major ones
// ([.., block_size, k_head_size]) make one token's head dims contiguous, matching the V cache.
// The scalar-gather branches below express every read as
// (head * PA_K_HIDDEN_STRIDE + token * PA_K_TOKEN_STRIDE) so one code path serves both -- they are
// the fallback whenever the block-read pitch rule fails, which for a token-major cache means
// head 48/80 (f16), head 32/48/80/96 (i8) or head % 128 != 0 (u4).
#if IS_PA_K_TOKEN_MAJOR
#  define PA_K_TOKEN_STRIDE  PA_K_ROW_ELEMS
#  define PA_K_HIDDEN_STRIDE 1
#else
#  define PA_K_TOKEN_STRIDE  1
#  define PA_K_HIDDEN_STRIDE PAGED_ATTENTION_BLOCK_SIZE
#endif

// Distance in K elements from one (block, kv_head) cache page to the next. The comp region a
// compressed cache appends grows whichever of the two factors it is INDEXED BY, so the host jits the
// pair and every layout comes out of the same product:
//   uncompressed   (head_size,     block_size)      no comp at all
//   i8 BY_TOKEN    (head_size + 4, block_size)      one (scale, zp) pair per token   -> wider row
//   i8 BY_CHANNEL  (head_size,     block_size + 4)  one pair per channel -> 4 more head_size rows
// The DATA row pitch is HEAD_SIZE in every case -- the +4 is never inside a data row.
#if IS_PAGED_ATTENTION
#  define PA_K_PAGE_STRIDE (ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE)
// An uncompressed cache has no comp region, so both factors collapse and PA_K_PAGE_STRIDE is
// PAGED_ATTENTION_BLOCK_SIZE * HEAD_SIZE exactly. Asserted rather than assumed because the f16 block
// read below spells that product out literally (see the comment there).
#  if !IS_PA_KV_COMPRESSED
#    if (ADJUSTED_K_HEAD_SIZE != HEAD_SIZE) || (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE != PAGED_ATTENTION_BLOCK_SIZE)
#      error "sdpa_ocl.cl: an uncompressed PA K page must have ADJUSTED_* equal to the plain sizes"
#    endif
#  endif
#endif

// Offset from a K / V page base to the comp region that follows the data rows, in cache-dtype
// elements (which is bytes in every compressed mode). Same place for every quant mode -- only the
// CONTENT differs:
//   BY_TOKEN   two per-token f16 arrays, scale at [token], zp at [PAGED_ATTENTION_BLOCK_SIZE + token]
//   BY_CHANNEL HEAD_SIZE interleaved (scale, zp) f16 pairs, so channel c's pair is the DWORD at [c]
// The packing shrinks the DATA region, not the comp region, which is why the multiplier is
// PA_*_ROW_ELEMS rather than HEAD_SIZE.
// Matches pa_kv_cache_update_ref.cl's BC_COMP_OFF / quantize_and_save_per_token.
#if IS_PAGED_ATTENTION
#  define PA_K_COMP_OFF ((size_t)PA_K_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#  define PA_V_COMP_OFF ((size_t)PA_V_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#endif

// ---------------------------------------------------------------------------------------------
// u4 (INT4) token-major BY_CHANNEL K cache: the DPAS depth axis is PERMUTED.
//
// This kernel's KQ A operand is K itself, so lane == head dim (the DPAS depth index) -- the exact
// mirror of sdpa_ocl_decode.cl, where K is the B operand and lane == token. The K page uses the
// upstream ADJACENT nibble order (byte b holds channel 2b in the low nibble, 2b+1 in the high), which
// the writer is forced into: NUM_K_HEAD_SIZE_PARTITIONS splits the channel range across WORKGROUPS,
// so a split-at-k/2 convention would put a byte's two channels in different workgroups and race.
//
// A byte column is therefore a channel PAIR, so the 8b VNNI-transform read -- whose lane IS the byte
// column -- hands lane L channels (2L, 2L+1) of the window, never the contiguous (base + L) a DPAS
// tile wants. No lane-local rearrangement can fix that; only a cross-lane shuffle could.
//
// The way out is that depth is a CONTRACTION axis: permuting it identically in A and B leaves
// sum_d K[key][d] * Q[query][d] unchanged. So both operands adopt this labelling, in which tile db
// of the pair covering the 32-channel window win = (db>>1)*32 owns the even channels for db even and
// the odd ones for db odd:
//
//     PA_K_U4_CHANNEL(db, L) = win + 2L + (db & 1)          byte = win/2 + L,  nibble = db & 1
//
// Every consumer then falls out cheaply:
//   - the 2D read at byte column win/2 lands exactly this, one read per tile PAIR;
//   - the per-lane fallback addresses byte (win/2 + L) -- LANE-CONTIGUOUS, i.e. better coalesced
//     than the natural order would be, and the nibble select is lane-uniform;
//   - the per-channel comp is one vload2 per window, serving both tiles of the pair at once;
//   - Q pays the whole cost, ONCE per workgroup, in the SLM staging loop below.
// ---------------------------------------------------------------------------------------------
#if IS_PA_K_U4
#  define PA_K_U4_WIN(db)        (((db) >> 1) * (2 * DPAS_K))
#  define PA_K_U4_PAR(db)        ((db) & 1)
#  define PA_K_U4_CHANNEL(db, l) (PA_K_U4_WIN(db) + 2 * (int)(l) + PA_K_U4_PAR(db))
#endif

// V keeps the SPLIT convention instead (byte b holds dim b and dim b + PA_V_ROW_ELEMS), because V is
// the S*V B operand where lane == head dim as well -- with adjacent packing a lane would own two
// different dims and no DPAS N axis could express it. So a read at the folded byte column hands lane
// c head dim base + c in BOTH halves, and the whole S*V tile loop, vb indexing and output store are
// unchanged; only the column fold and a nibble select are new. Both are the identity for i8/f16, so
// those paths preprocess to exactly what they were.
#if IS_PA_K_U4
#  define PA_V_U4_HI(base)  ((base) >= PA_V_ROW_ELEMS)
#  define PA_V_U4_COL(base) (PA_V_U4_HI(base) ? ((base) - PA_V_ROW_ELEMS) : (base))
#else
#  define PA_V_U4_HI(base)  0
#  define PA_V_U4_COL(base) (base)
#endif

// Host/kernel drift guards for the token-major BY_CHANNEL K page. Each of these would otherwise
// silently read the wrong bytes rather than fail to build.
#if IS_PA_K_BY_CHANNEL
#  if !IS_PA_KV_COMPRESSED
#    error "sdpa_ocl.cl: IS_PA_K_BY_CHANNEL requires a compressed (i8 or u4) K cache"
#  endif
#  if !IS_PA_K_TOKEN_MAJOR
// The data region must be [block_size tokens, PA_K_ROW_ELEMS]; upstream BY_CHANNEL is d-major and its
// comp lives inline at the end of every column, which nothing below can address.
#    error "sdpa_ocl.cl: IS_PA_K_BY_CHANNEL is only valid for the token-major BY_CHANNEL page"
#  endif
#  if ADJUSTED_K_HEAD_SIZE != HEAD_SIZE
// BY_CHANNEL's comp is sized by CHANNEL, so it grows the page's row COUNT
// (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE), not its row pitch. A host that added the BY_TOKEN +4 here
// would put every page base 4 * block_size bytes too far apart.
#    error "sdpa_ocl.cl: BY_CHANNEL must leave ADJUSTED_K_HEAD_SIZE == HEAD_SIZE"
#  endif
#endif

#if IS_PA_K_U4
#  if !IS_PA_K_BY_CHANNEL
// A u4 PA key cache is always BY_CHANNEL -- execution_config.cpp asserts against 4-bit BY_TOKEN keys
// -- so there is no u4 BY_TOKEN path to write and none is implemented.
#    error "sdpa_ocl.cl: IS_PA_K_U4 requires the token-major BY_CHANNEL page"
#  endif
#  if (HEAD_SIZE % 2) != 0
// PA_K_ROW_ELEMS is HEAD_SIZE/2 exactly, with no rounding anywhere.
#    error "sdpa_ocl.cl: u4 needs an even HEAD_SIZE"
#  endif
#  if (DKS % 2) != 0 || (DKS_ACTIVE % 2) != 0
// The depth permutation pairs DPAS tiles (2g, 2g+1) over a 32-channel window. DKS is D_MAX/DPAS_K and
// D_MAX is a power of two >= 32, so this always holds -- it is here to make a drift loud.
// DKS_ACTIVE (which is what the loops below are bounded by) is rounded up to even for exactly this
// reason; the test covers it too so a change to that rounding cannot silently half-form a pair.
#    error "sdpa_ocl.cl: u4 needs an even DKS/DKS_ACTIVE so the depth tiles pair up"
#  endif
#  if (PA_V_ROW_ELEMS % SUBGROUP_SIZE) != 0
// PA_V_U4_COL folds whole 16-wide byte-column groups, which assumes the split point is one.
#    error "sdpa_ocl.cl: u4 needs PA_V_ROW_ELEMS to be a multiple of SUBGROUP_SIZE"
#  endif
#  if PA_CUR_KV_F16
// The Kc read for u4 uses intel_sub_group_2d_block_read_32b_8r16x1c, whose 8r16x1c geometry is
// hard-coded to exactly one DPAS row-block of keys (8) and one 32-channel u4 window (16 dwords).
#    if DPAS_ROWS != 8
#      error "sdpa_ocl.cl: the u4 Kc dword read is 8r; DPAS_ROWS must be 8"
#    endif
#    if DPAS_K != 16 || SUBGROUP_SIZE != 16
#      error "sdpa_ocl.cl: the u4 Kc dword read is 16 dwords wide; DPAS_K and SUBGROUP_SIZE must be 16"
#    endif
#  endif
#endif

// ---------------------------------------------------------------------------------------------
// 1D subgroup block read of a whole cache page.
//
// A u4 page's row is HEAD_SIZE/2 bytes, so at head 64 it is 32 -- below the 64-byte block2d
// minimum, and no head size can fix that for BOTH K (row = h/2) and V (row = Align(h/2, 16)).
// The host therefore leaves USE_2D_BLOCK_IO_{K,V}_PA_I8 off and the loads fall back to a per-lane
// byte gather: measured on the gpt-oss-20b mixed kernel, 64 K + 128 V SIMD-16 scattered messages
// per k0 iteration against 16 dpas.
//
// But the page's DATA REGION is one contiguous run of PAGED_ATTENTION_BLOCK_SIZE * <ROW> bytes,
// and intel_sub_group_block_read_uc16 lands component i of lane L on byte SUBGROUP_SIZE * i + L.
// Both consumers want byte (token t, 16-wide column group c) + L, i.e.
//
//     SUBGROUP_SIZE * (PA_PAGE_UC16 * r + i) + L  ==  t * <ROW> + SUBGROUP_SIZE * c + L
//                                             <=>  PA_PAGE_UC16 * r + i == t * COLS + c
//
// with COLS = <ROW> / SUBGROUP_SIZE. So r and i below place any (t, c) with no shuffle and no
// per-lane address, and COLS reads cover all PAGED_ATTENTION_BLOCK_SIZE tokens of a column group.
//
// r is a compile-time constant whenever t is, EVEN IF c is not: c < COLS and COLS divides
// PA_PAGE_UC16 (host gate: COLS is a power of two), so [t*COLS, t*COLS + COLS) never straddles a
// read boundary. i is not, so a caller whose c is a runtime value (V, whose column group comes from
// sg_j0_sv) instead BIASES THE BASE by SUBGROUP_SIZE * c and passes c = 0 -- identical arithmetic,
// and it keeps i constant so the uchar16 component select stays a register subscript rather than
// an indirect address. K's c is PA_K_U4_WIN(db)/2/SUBGROUP_SIZE == db >> 1, a constant, so K reads
// at the plain page base and hoists the reads out of the db loop entirely.
//
// Bound on what is touched: SUBGROUP_SIZE*c + (COLS-1)*PA_PAGE_RD_BYTES + PA_PAGE_RD_BYTES - 1,
// i.e. 527 B for the head-64 u4 V page (COLS = 2, c <= 1) against a 16 * ADJUSTED_V_HEAD_SIZE = 576 B
// page, and 511 B for K against 768 B. The overhang past the data region only ever lands in the
// page's own trailing comp arrays, never outside the allocation, and only unused components read it.
// ---------------------------------------------------------------------------------------------
#define PA_PAGE_UC16          16                                     // components of a uchar16
#define PA_PAGE_RD_BYTES      (SUBGROUP_SIZE * PA_PAGE_UC16)
#define PA_PAGE_COLS(ROW)     ((ROW) / SUBGROUP_SIZE)                // 16-byte column groups per row
#define PA_PAGE_READS(ROW)    PA_PAGE_COLS(ROW)                      // reads to cover one column group
#define PA_PAGE_R(ROW, t, c)  (((t) * PA_PAGE_COLS(ROW) + (c)) / PA_PAGE_UC16)
#define PA_PAGE_I(ROW, t, c)  (((t) * PA_PAGE_COLS(ROW) + (c)) % PA_PAGE_UC16)

#if USE_1D_BLOCK_IO_K_PA_U4 || USE_1D_BLOCK_IO_V_PA_U4
#  if !IS_PA_K_U4
// The dequant reused below is the u4 nibble one; i8 pages take the block2d paths or the gather.
#    error "sdpa_ocl.cl: the 1D page read is implemented for the u4 BY_CHANNEL token-major page only"
#  endif
#  if PAGED_ATTENTION_BLOCK_SIZE != SUBGROUP_SIZE
// One key group == one page == one subgroup width is what makes the page's token index equal the
// key's subgroup-local index, which is what makes t a compile-time constant above.
#    error "sdpa_ocl.cl: the 1D page read assumes PAGED_ATTENTION_BLOCK_SIZE == SUBGROUP_SIZE"
#  endif
#endif

// The row geometry the mapping needs, asserted per tensor because the host gates them
// independently: at head 48 the u4 V row is Align(24, 16) == 32 and qualifies while the K row is 24
// and does not. Without these a host-gate drift would not fail to build -- it would read the wrong
// bytes, because PA_PAGE_COLS silently truncates for a row that is not a whole number of column
// groups, and a COLS that is not a power of two makes the READ index depend on the column group,
// which the V branch has already committed to being constant (it passes c = 0 and biases the base).
#if USE_1D_BLOCK_IO_K_PA_U4
#  if (PA_K_ROW_ELEMS % SUBGROUP_SIZE) != 0 || PA_PAGE_COLS(PA_K_ROW_ELEMS) > PA_PAGE_UC16 || \
      (PA_PAGE_COLS(PA_K_ROW_ELEMS) & (PA_PAGE_COLS(PA_K_ROW_ELEMS) - 1)) != 0
#    error "sdpa_ocl.cl: the 1D K page read needs PA_K_ROW_ELEMS = SUBGROUP_SIZE * 2^n, n <= 4"
#  endif
#endif
#if USE_1D_BLOCK_IO_V_PA_U4
#  if (PA_V_ROW_ELEMS % SUBGROUP_SIZE) != 0 || PA_PAGE_COLS(PA_V_ROW_ELEMS) > PA_PAGE_UC16 || \
      (PA_PAGE_COLS(PA_V_ROW_ELEMS) & (PA_PAGE_COLS(PA_V_ROW_ELEMS) - 1)) != 0
#    error "sdpa_ocl.cl: the 1D V page read needs PA_V_ROW_ELEMS = SUBGROUP_SIZE * 2^n, n <= 4"
#  endif
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)))
KERNEL(sdpa_ocl)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
#if IS_PAGED_ATTENTION && !IS_PREFILL
        const global QRY_DATA_T *Kc,
        const global QRY_DATA_T *Vc,
#endif
    global half *A,
#if IS_PAGED_ATTENTION
        const __global INPUT3_TYPE* subsequence_begins,
    #if !IS_PREFILL
        const __global INPUT3_TYPE* past_lens,
        const __global INPUT3_TYPE* block_indices,
        const __global INPUT3_TYPE* block_indices_begins,
    #endif
#endif
#if WITH_ATTN_MASK
        const global half *msk,
#endif
#if WITH_SCALE
        global SCALE_DATA_T *scale_ptr,
#endif
#ifdef HAS_SINK_INPUT
        const global SINK_DATA_T *sink_ptr,
#endif
#if HAS_TOKEN_TYPE_IDS
        const __global int* token_type_ids,
        const int token_type_ids_count,
#endif
#if IS_PAGED_ATTENTION
        const __global int* blocked_indexes_start_and_gws_mapping
#else
        const int d,
        const int k,
        const int q
#endif
    #ifdef KV_COMPRESSED
        , const global KEY_ATTR_SCALES_DATA_T *K_scales
    #if KEY_ZERO_POINTS
        , const global KEY_ATTR_ZP_DATA_T *K_zp
    #endif
        , const global VAL_ATTR_SCALES_DATA_T *V_scales
    #if VAL_ZERO_POINTS
        , const global VAL_ATTR_ZP_DATA_T *V_zp
    #endif
    #endif
        )
{
#if IS_PAGED_ATTENTION
    const uint query_block_idx = get_group_id(0) << 1;
    const uint block_start_pos = blocked_indexes_start_and_gws_mapping[query_block_idx];
    const uint gws_mapping = blocked_indexes_start_and_gws_mapping[query_block_idx + 1];
    const uint subsequence_begin = subsequence_begins[gws_mapping];
    const uint subsequence_end = subsequence_begins[gws_mapping + 1];
    const uint subsequence_query_block_idx = block_start_pos - subsequence_begin;
    int q = subsequence_end - subsequence_begin;
    #if HAS_QQ_BIAS
        const uint qq_bias_num = qq_bias_begins[gws_mapping + 1] - qq_bias_begins[gws_mapping];
        const uint cumulated_spec_num = qq_bias_begins[gws_mapping];
    #endif
    #if IS_PREFILL
        const int past_len = 0;
        const int k = q;
    #else
        const int past_len = past_lens[gws_mapping];
        const int k = q + past_len;
    #endif
    const int d = HEAD_SIZE;
#endif

    const size_t lane  = get_sub_group_local_id();
    const size_t sg_ij = get_local_id(1);
#if IS_PAGED_ATTENTION
    // Query blocks are handed out through blocked_indexes_start_and_gws_mapping, so the block this
    // workgroup owns is not a plain function of get_group_id(0): it is the block start recorded in
    // that buffer, expressed relative to the beginning of its own subsequence.
    const size_t wg_j0 = subsequence_query_block_idx;
#else
    const size_t wg_j0 = get_group_id(0) * kq_wg_tile_queries;
#endif
    const size_t b0 = get_group_id(1);     // heads_num
    const size_t b1 = get_group_id(2);     // batch
    const size_t b0_kv = b0 / KV_GROUP_SIZE;

    const size_t sg_i_kq  = sg_ij % kq_sg_per_wg_keys;
    const size_t sg_j_kq  = sg_ij / kq_sg_per_wg_keys;
    const size_t sg_i0_kq = sg_i_kq * kq_sg_tile_keys;
    const size_t sg_j0_kq = sg_j_kq * kq_sg_tile_queries;

    const size_t sg_i_sv = sg_ij / sv_sg_per_wg_values;
    const size_t sg_j_sv = sg_ij % sv_sg_per_wg_values;
    const size_t sg_i0_sv = sg_i_sv * sv_sg_tile_scores;
    const size_t sg_j0_sv = sg_j_sv * sv_sg_tile_values;

    const float LOG2E = 1.4426950408889634f;

    #if WITH_SCALE
        /* Load scale */
        #if INVERT_SCALE
            float iscale = convert_float(*scale_ptr);
            float scale = native_recip(iscale);
        #else
            float scale = convert_float(*scale_ptr);
            float iscale = native_recip(scale);
        #endif
    #else
        #ifdef STATIC_SCALE_VALUE
            #if INVERT_SCALE
                float iscale = convert_float(STATIC_SCALE_VALUE);
                float scale = convert_float(STATIC_SCALE_VALUE_INV);
            #else
                float scale = convert_float(STATIC_SCALE_VALUE);
                float iscale = convert_float(STATIC_SCALE_VALUE_INV);
            #endif
        #else
            float iscale = sqrt(convert_float(HEAD_SIZE));
            float scale = native_recip(iscale);
        #endif
    #endif

    scale *= LOG2E;

#ifdef HAS_SINK_INPUT
    // Attention sink: one extra per-head logit that joins the softmax max and denominator but
    // carries a ZERO value vector, so it never reaches the numerator. That makes it exactly the
    // online-softmax state SEEDED with one synthetic key -- running max = sink, running sum = 1,
    // A_tile = 0 -- which costs three initialisers and leaves the key loop untouched. sdpa_micro.cl
    // instead injects it per k0 tile from the subgroup owning the last key (its is_last_m_sg), and
    // has to fork LOG_2_E_MUL_SCALE to pre-scale S_tile; neither is needed here because S_max_slm
    // below is a RUNNING max that spans the whole loop.
    //
    // Domain: S_tile stays unscaled and S_max_slm holds the max in that same raw domain, because the
    // softmax recovers m_log2 = m_new * scale (with LOG2E already folded into scale). The sink is a
    // real logit, so it enters divided by the attention scale -- the same convention the attn mask
    // uses at its `* iscale` load below.
    const float sink_raw = convert_float(sink_ptr[b0]) * iscale;
#endif

    /* Row stride (in elements) of the Q/K/V/A matrices. */
#if IS_PAGED_ATTENTION
    // Paged attention Q/K/V/output are 2D [total_tokens, num_heads * head_size]: there is no Y
    // dimension, so the generic QRY_S2/KEY_S2/VAL_S2/DST_S2 (Y pitch) macros are all 0 here and the
    // token stride has to be derived from the head layout instead.
    const uint ldq = HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint ldk = HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    const uint lda = HEAD_SIZE * HEADS_NUM;
#else
    const uint ldq = QRY_S2;
    const uint ldk = KEY_S2;
    const uint ldv = VAL_S2;
    const uint lda = DST_S2;
#endif

#if IS_PAGED_ATTENTION
    // Tokens of all subsequences are packed into one matrix, so a batch index does not exist:
    // seek to the first token of this workgroup's subsequence and to this head's column slice.
    Q += (size_t)subsequence_begin * ldq + b0 * HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    A += (size_t)subsequence_begin * lda + b0 * HEAD_SIZE;
    #if IS_PREFILL
        K += (size_t)subsequence_begin * ldk + b0_kv * HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
        V += (size_t)subsequence_begin * ldv + b0_kv * HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    #else
        const uint base_block_index = block_indices_begins[gws_mapping];
        #if PA_CUR_KV_F16
            // Kc/Vc are this iteration's raw f16 K/V -- the same tensors the PREFILL variant gets as
            // K/V -- so they take the same bump as the IS_PREFILL branch above. Row index into them is
            // (key - past_len): they hold only the q NEW tokens of the flattened batch, while `key`
            // counts from the start of the CACHED context.
            Kc += (size_t)subsequence_begin * ldk + b0_kv * HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
            Vc += (size_t)subsequence_begin * ldv + b0_kv * HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
        #endif
    #endif
    #if BIDIR_MASK
        // Workgroup-uniform, so every branch on it below is uniform too: no divergence, and the
        // subgroup reductions / loop trip counts stay reachable by every lane. False means the host
        // handed us an empty token_type_ids ("[B_token | 0]", no image tokens), in which case the
        // buffer must not be touched at all -- not even to form a bumped pointer past its end.
        #if USE_BIDIR_GATE
            const bool bidir_active = (token_type_ids_count > 0);
        #else
            // Negative control (SDPA_OCL_BIDIR_GATE=0): read token_type_ids unconditionally, exactly
            // as this kernel did before the gate existed. Folds to a constant, so every branch below
            // collapses back to its pre-gate form.
            const bool bidir_active = true;
        #endif

        // token_type_ids is [B_token]: one entry per NEW token of the FLATTENED batch, exactly like
        // the Q rows above, so it needs the same subsequence bump. Every index derived from it below
        // is then LOCAL -- subsequence-relative and new-token-relative, i.e. in [0, q) -- matching
        // wg_j0 and `query`, but NOT `key` / `causal_k` / `window_k_begin`, which count from the
        // start of the CACHED context (key = query_position_offset + local). PREFILL is the
        // query_position_offset == 0 case where the two spaces coincide. NOTE: sdpa_micro.cl and
        // sdpa_opt.cl omit this bump and index the flattened buffer with subsequence-relative
        // positions, which only agrees for a single subsequence starting at token 0.
        if (bidir_active)
            token_type_ids += subsequence_begin;
    #endif
#else
    Q += QRY_OFF(b1, b0, 0, 0) + INPUT0_OFFSET;
    K += KEY_OFF(b1, b0_kv, 0, 0) + INPUT1_OFFSET;
    V += VAL_OFF(b1, b0_kv, 0, 0) + INPUT2_OFFSET;
    A += DST_OFF(b1, b0, 0, 0, 0);
#endif
#if WITH_ATTN_MASK
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
#endif
#ifdef KV_COMPRESSED
    // Hoist dynamic compression-layout batch/head pitches out of the hot loops.
    const uint k_comp_base = KEY_COMP_OFF(b1, b0_kv, 0, 0);
    #if USE_2D_BLOCK_IO_V_I8
    const uint v_comp_base = VAL_COMP_OFF(b1, b0_kv, 0, 0);
    #endif
#endif

    const int QD_w = d * (int)sizeof(QRY_DATA_T), QD_h = q, QD_p = (int)ldq * (int)sizeof(QRY_DATA_T);
    const int KD_w = d * (int)sizeof(KEY_DATA_T), KD_h = k, KD_p = (int)ldk * (int)sizeof(KEY_DATA_T);
    const int VD_w = d * (int)sizeof(VAL_DATA_T), VD_h = k, VD_p = (int)ldv * (int)sizeof(VAL_DATA_T);
    const int AD_w = d * (int)sizeof(half), AD_h = q, AD_p = (int)lda * (int)sizeof(half);

#if PA_CUR_KV_F16
    // Surfaces for the NEW-token half of the key range. Deliberately NOT KD_*/VD_*: those describe the
    // CACHE for this variant, whose element type is uchar for a compressed cache and whose height is k.
    // Kc/Vc are always f16 and only q rows tall -- rows past q are hardware zero-filled, which is the
    // same OOB behaviour the `key < k` masking already assumes, so no extra guard is needed.
    const int KcD_w = d * (int)sizeof(half), KcD_h = q, KcD_p = (int)ldk * (int)sizeof(half);
    const int VcD_w = d * (int)sizeof(half), VcD_h = q, VcD_p = (int)ldv * (int)sizeof(half);
    const global half *Kc_b2d = (const global half *)Kc;
    const global half *Vc_b2d = (const global half *)Vc;
    int KcD_w_b2d = KcD_w, VcD_w_b2d = VcD_w;
    int KcD_x0 = 0, VcD_x0 = 0;
    #if BLOCK2D_KV_CUR_BASE_FIXUP
    // Same repair as BLOCK2D_KV_BASE_FIXUP below, and needed for the same reason: b0_kv * HEAD_SIZE is
    // a whole number of rows but not necessarily of 64 B, and INPUT1/INPUT2_PAD_BEFORE_FEATURE_NUM can
    // be dynamic (a Q/K/V that is a crop view of one fused QKV tensor), so the base is not provably
    // 64B-aligned. Round down, widen, shift x.
    {
        const uint kc_prem = (uint)(as_long(Kc_b2d) & 63);
        const uint vc_prem = (uint)(as_long(Vc_b2d) & 63);
        Kc_b2d = (const global half *)((const global char *)Kc_b2d - kc_prem);
        Vc_b2d = (const global half *)((const global char *)Vc_b2d - vc_prem);
        KcD_w_b2d = KcD_w + (int)kc_prem;
        VcD_w_b2d = VcD_w + (int)vc_prem;
        KcD_x0 = (int)(kc_prem / sizeof(half));
        VcD_x0 = (int)(vc_prem / sizeof(half));
    }
    #endif
    #if IS_PA_K_U4
    // The u4 K read below addresses Kc as a DWORD surface (see the PA_K_U4 branch at the K load), so its
    // x shift is in dwords, not halves. Two things have to hold for that to be exact:
    //   - the surface ORIGIN must be 4B-aligned. It is: the host forces BLOCK2D_KV_CUR_BASE_FIXUP on for
    //     u4, which rounds the base down to 64 B.
    //   - this head's first channel must be an EVEN number of halves from that origin, or a dword would
    //     straddle the (2c, 2c+1) channel pair the permuted depth axis is built on.
    // The second one is a RUNTIME property -- it comes out of subsequence_begin * ldk +
    // b0_kv * HEAD_SIZE + the feature padding, and the padding can be dynamic -- so it is tested here
    // rather than in the host gate. When it fails, from_cache below is forced true for every tile and
    // the kernel behaves exactly as it did before this path existed: correct, just slower.
    const int KcD_x0_dw = KcD_x0 / 2;
    const bool kc_dword_ok = ((KcD_x0 & 1) == 0);
    #endif
#endif

#if USE_2D_BLOCK_IO_KV
    // 2D block IO surface origin for the f16 K/V loads.
    //
    // The builtins require a 64B-aligned base. K and V have already been advanced to this head's
    // rows above, by an offset that is an integer multiple of the row width
    // (head_size * element_size) but not necessarily of 64: at HEAD_SIZE 72 the f16 row is 144 B,
    // so every head with (head % 4) != 0 lands 16, 32 or 48 bytes past a boundary.
    //
    // Repair it the way sdpa_micro's block2d_load helper always has: round the base DOWN to the 64 B
    // boundary, then compensate by shifting the x coordinate and WIDENING the surface by the same
    // number of bytes. The widening extends backwards -- the rounded base sits `prem` bytes before
    // this head's first element -- so the surface still ends exactly at this head's last element.
    // Columns past the head dim therefore stay out of bounds and the hardware zero-fills them, which
    // is precisely what the `head < d` guard did on the scalar path. Nothing from the next head can
    // leak in.
    //
    // The x shift must divide exactly. The base is buffer_base (>= 64B aligned) plus an offset in
    // ELEMENTS, so prem is always a multiple of sizeof(KEY_DATA_T) -- which is all these 16b
    // builtins need, and it holds even when a view offset (INPUT*_OFFSET) makes the base something
    // other than a whole number of rows. The stronger property (prem a multiple of 16, from
    // base = m * row_bytes and the host's row_bytes % 16 == 0 gate) is what a 32-bit transposed read
    // would need; Q is not routed through here yet.
    //
    // as_long() on a global pointer is the same form sdpa_micro's block2d_load uses, so it is known
    // to compile on this toolchain.
    //
    // Hoisted out of the k0 loop: neither base moves once the head is fixed.
    const global KEY_DATA_T *K_b2d = K;
    const global VAL_DATA_T *V_b2d = V;
    int KD_w_b2d = KD_w, VD_w_b2d = VD_w;
    int KD_x0 = 0, VD_x0 = 0;
    #if BLOCK2D_KV_BASE_FIXUP
    {
        const uint k_prem = (uint)(as_long(K) & 63);
        const uint v_prem = (uint)(as_long(V) & 63);
        K_b2d = (const global KEY_DATA_T *)((const global char *)K - k_prem);
        V_b2d = (const global VAL_DATA_T *)((const global char *)V - v_prem);
        KD_w_b2d = KD_w + (int)k_prem;
        VD_w_b2d = VD_w + (int)v_prem;
        KD_x0 = (int)(k_prem / sizeof(KEY_DATA_T));
        VD_x0 = (int)(v_prem / sizeof(VAL_DATA_T));
    }
    #endif
#endif

    local uint  Q_slm[DKS_ACTIVE * q_blocks * Q_DWORDS * SUBGROUP_SIZE];
    local uint  S_slm[kq_wg_tile_keys * kq_wg_tile_queries / 2];
    local float S_sum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
#ifdef HAS_SINK_INPUT
        // Seeded, not -INFINITY: this slot is written once here and thereafter only atomic-maxed in
        // the key loop, so it is the running max over every k0 tile AND every subgroup. Seeding it
        // is what counts the sink exactly once.
        S_max_slm[qi] = sink_raw;
#else
        S_max_slm[qi] = -INFINITY;
#endif

    // Cooperative Q->SLM staging: the Q tile is q_blocks query-blocks x DKS_ACTIVE head-dim
    // chunks = q_blocks*DKS_ACTIVE independent (q_block, db) tiles. Distribute them round-robin
    // across the workgroup subgroups so all subgroups load Q and every tile is staged even
    // when q_blocks*DKS_ACTIVE exceeds sg_per_wg (e.g. D_MAX >= 256), shrinking the prologue
    // Q-load latency. The loop bound guarantees q_block < q_blocks, so no guard is needed.
    // Bounded by DKS_ACTIVE, not DKS: the tiles past the head dim would stage all-zero Q and the
    // KQ loop no longer reads them.
    for (int tile = sg_ij; tile < q_blocks * DKS_ACTIVE; tile += sg_per_wg) {
        const int q_block = tile / DKS_ACTIVE;   // 0..q_blocks-1
        const int db      = tile % DKS_ACTIVE;   // 0..DKS_ACTIVE-1
        const int query_base = wg_j0 + q_block * SUBGROUP_SIZE;
        uint8 q_pack;
#if IS_PA_K_U4
        // The u4 K page forces a permuted depth axis (see PA_K_U4_CHANNEL): chunk db holds the even
        // channels of its 32-channel window for db even and the odd ones for db odd. Q is the other
        // operand of the same contraction, so it has to adopt the identical labelling -- and the
        // cheapest place to pay for that is here, in the staging that runs ONCE per workgroup, rather
        // than per k0 tile in the KQ loop.
        //
        // A chunk therefore spans 32 consecutive channels instead of DPAS_K, read as two windows w0/w1
        // (the block read returns 8 dwords == 16 halves). Both parities read the same 32 channels, so
        // Q's staging traffic doubles -- irrelevant next to K/V, and nothing downstream changes.
        //
        // The deinterleave is pure dword arithmetic: w0 dword m holds channels (win+2m, win+2m+1) as
        // (low, high), and q_pack dword j must hold channels (win+4j+par, win+4j+2+par), i.e. half
        // `par` of dwords 2j and 2j+1 of the same window. Three ops per output dword, 8 dwords.
        const int u4_win = PA_K_U4_WIN(db);
        const int u4_par = PA_K_U4_PAR(db);
        uint8 w0, w1;
    #if USE_2D_BLOCK_IO_Q
        if (query_base + SUBGROUP_SIZE <= q && u4_win + 2 * DPAS_K <= d) {
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(u4_win / 2, query_base), (private uint *)&w0);
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(u4_win / 2 + DPAS_K / 2, query_base), (private uint *)&w1);
        } else
    #endif
        {
            const int query = query_base + lane;
            ushort16 qv0 = (ushort16)0;
            ushort16 qv1 = (ushort16)0;
            if (query < q) {
                const global ushort *q_row = (const global ushort *)(Q + (size_t)query * ldq + u4_win);
                if (u4_win + 2 * DPAS_K <= d) {
                    qv0 = vload16(0, q_row);
                    qv1 = vload16(1, q_row);
                } else {
                    #pragma unroll
                    for (int head_offset = 0; head_offset < DPAS_K; ++head_offset) {
                        if (u4_win + head_offset < d)
                            qv0[head_offset] = q_row[head_offset];
                        if (u4_win + DPAS_K + head_offset < d)
                            qv1[head_offset] = q_row[DPAS_K + head_offset];
                    }
                }
            }
            w0 = as_uint8(as_short16(qv0));
            w1 = as_uint8(as_short16(qv1));
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint a = (j < 4) ? w0[2 * j] : w1[2 * (j - 4)];
            const uint b = (j < 4) ? w0[2 * j + 1] : w1[2 * (j - 4) + 1];
            q_pack[j] = u4_par ? ((a >> 16) | (b & 0xFFFF0000u)) : ((a & 0x0000FFFFu) | (b << 16));
        }
#else
        const int head_base = db * DPAS_K;
#if USE_2D_BLOCK_IO_Q
        if (query_base + SUBGROUP_SIZE <= q && head_base + DPAS_K <= d) {
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(head_base / 2, query_base), (private uint *)&q_pack);
        } else
#endif
        {
            const int query = query_base + lane;
            ushort16 qv = (ushort16)0;
            if (query < q) {
                if (head_base + DPAS_K <= d) {
                    qv = vload16(0, (global ushort *)(Q + (size_t)query * ldq + head_base));
                } else {
                    #pragma unroll
                    for (int head_offset = 0; head_offset < DPAS_K; ++head_offset) {
                        if (head_base + head_offset < d) {
                            qv[head_offset] = as_ushort(Q[(size_t)query * ldq + head_base + head_offset]);
                        }
                    }
                }
            }
            q_pack = as_uint8(as_short16(qv));
        }
#endif
        intel_sub_group_block_write8(
            (local uint *)&Q_slm[((db * q_blocks + q_block) * Q_DWORDS) * SUBGROUP_SIZE], q_pack);
    }

    float S_max_tile[kq_query_blocks];
    float S_sum_tile[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
#ifdef HAS_SINK_INPUT
        // The private half of the same seed. S_max_tile holds the max already multiplied by scale
        // (the softmax stores m_log2 back into it), so it is seeded from the SAME sink_raw * scale
        // product the loop will compute for m_log2 -- which makes the first rescale alpha come out
        // exactly 1.0 when the sink dominates, rather than 1.0 +/- a ulp.
        S_max_tile[qb] = sink_raw * scale;
        // The sink's exp2(sink - sink) = 1 goes to ONE subgroup only. Each subgroup writes its own
        // S_sum_slm[query * kq_sg_per_wg_keys + sg_i_kq] slot and the epilogue sums all of them, so
        // seeding every subgroup would count the sink kq_sg_per_wg_keys times.
        S_sum_tile[qb] = (sg_i_kq == 0) ? 1.0f : 0.0f;
#else
        S_max_tile[qb] = -INFINITY;
        S_sum_tile[qb] = 0.0f;
#endif
    }

    float8 A_tile[sv_score_blocks][sv_value_blocks];
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r)
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            A_tile[r][cd] = (float8)0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

#if IS_PAGED_ATTENTION && !IS_PREFILL
    const int query_position_offset = past_len;
#else
    const int query_position_offset = 0;
#endif

    // Causal upper bound on the key loop. With IS_CAUSAL every key > query is masked to -inf, so a
    // workgroup owning queries [wg_j0, wg_j0 + kq_wg_tile_queries) can never need a key beyond
    // wg_j0 + kq_wg_tile_queries - 1. Without this bound the loop walks the FULL key range and
    // every k0 tile past the diagonal is pure waste: it loads K/V, runs the DPAS, then throws the
    // result away in the causal mask. At q = k = 1024 with kq_wg_tile_queries = 32 that is 256
    // tile-iterations instead of 144 (1.78x), and it is why sdpa_micro -- which has had this bound
    // since day one (causal_k = min(k, wg_j0 + wg_tile_n)) -- wins despite a less efficient
    // per-tile inner loop.
    // The non-causal case keeps the full range, and the bound is a no-op there.
#if IS_CAUSAL
    int causal_k = min(k, query_position_offset + (int)wg_j0 + kq_wg_tile_queries);
#else
    const int causal_k = k;
#endif

#if IS_CAUSAL && BIDIR_MASK
    // An image group is bidirectional, so a query inside one also needs the FUTURE keys of its own
    // group -- which the causal bound above cuts away. Extend it to the end of the group that
    // overlaps this workgroup's query range. Scanning only the workgroup's LAST query is sufficient:
    // a group is contiguous, so any query whose group runs past wg_j0 + kq_wg_tile_queries contains
    // wg_q_end in that same group and therefore has the same group end.
    // Counterpart of sdpa_micro.cl's bidir_causal_k, but the scan is subgroup-cooperative
    // (SUBGROUP_SIZE positions per step) instead of a lane-0 serial walk. Everything here is
    // subgroup-uniform -- wg_q_end, q and the reduction result -- so the loop trip count and the
    // break are uniform and sub_group_reduce_min() is reached by every lane.
    //
    // The whole scan runs in LOCAL space and is bounded by q, the NEW token count, not by k: groups
    // never leave the new-token region (openvino/reference/paged_attention.hpp builds image_group_*
    // over [seq_begins[s], seq_begins[s+1]) alone and maps it to keys as past + (idx - t_begin)), and
    // token_type_ids only has q entries for this subsequence. PREFILL, where q == k, is unchanged.
    //
    // Reaching a future key means reading it from the cache, which in MIXED holds the new tokens only
    // because pa_kv_cache_update runs before this stage (PagedAttentionOptImpl::execute()). The
    // reference relies on the same ordering and states it explicitly ("Populate cache with all new
    // tokens before computing attention"); nothing here asserts it.
    {
        const int wg_q_end = min((int)wg_j0 + kq_wg_tile_queries, q) - 1;
        if (bidir_active && wg_q_end >= 0 && token_type_ids[wg_q_end] == 1) {
            int group_end = wg_q_end + 1;
            while (group_end < q) {
                const int chunk_end = min(q, group_end + SUBGROUP_SIZE);  // exclusive
                const int idx = group_end + (int)lane;
                const bool ends_group = (idx < chunk_end) && (token_type_ids[idx] != 1);
                const int first = sub_group_reduce_min(ends_group ? idx : INT_MAX);
                if (first != INT_MAX) {
                    group_end = first;
                    break;
                }
                group_end = chunk_end;
            }
            // group_end <= q: the loop only ever assigns an index below q or the clamped chunk end,
            // so the KEY-space result stays <= query_position_offset + q == k.
            causal_k = max(causal_k, query_position_offset + group_end);
        }
    }
#endif

    // Sliding-window lower bound, the mirror of causal_k and the counterpart of sdpa_micro's
    // window_k0_begin. The mask below keeps only (query - SLIDING_WINDOW_SIZE, query], so the
    // smallest key this workgroup can need is for its smallest query, wg_j0:
    //   key > wg_j0 - SLIDING_WINDOW_SIZE  =>  first needed key = wg_j0 - SLIDING_WINDOW_SIZE + 1.
    // Every k0 tile below that is entirely outside the window and would be masked away wholesale,
    // exactly the waste causal_k removes at the top end. Round down to a k0 tile boundary so the
    // loop keeps its kq_wg_tile_keys stride and key_base stays tile-aligned (the 2D block reads
    // and the S_slm indexing both assume that).
#if IS_CAUSAL && SLIDING_WINDOW_SIZE
    int window_k_begin = max(0, query_position_offset + (int)wg_j0 - SLIDING_WINDOW_SIZE + 1);
    #if BIDIR_MASK
    // Mirror of the causal_k extension at the bottom end: a query inside an image group also needs
    // the group's PAST keys even when the sliding window has already dropped them. Extend the window
    // start back to the beginning of the group straddling it. Same sufficiency argument -- if a query
    // at or above wg_j0 has a group starting below window_k_begin, then window_k_begin lies inside
    // that (contiguous) group, so scanning from window_k_begin finds the right start.
    //
    // window_k_begin is a KEY coordinate, token_type_ids is indexed LOCALly, hence the shift. Once
    // window_begin_local <= 0 the window already reaches at or below the first new token, and since
    // groups never extend below that there is nothing left to extend -- which is also what keeps the
    // index in range. In PREFILL query_position_offset is 0 and this reduces to the old form.
    const int window_begin_local = window_k_begin - query_position_offset;
    if (bidir_active && window_begin_local > 0 && token_type_ids[window_begin_local] == 1) {
        int group_begin = window_begin_local;
        while (group_begin > 0) {
            const int chunk_begin = max(0, group_begin - SUBGROUP_SIZE);
            const int idx = chunk_begin + (int)lane;
            const bool ends_group = (idx < group_begin) && (token_type_ids[idx] != 1);
            const int last = sub_group_reduce_max(ends_group ? idx : -1);
            if (last >= 0) {
                group_begin = last + 1;
                break;
            }
            group_begin = chunk_begin;
        }
        window_k_begin = query_position_offset + group_begin;
    }
    #endif
    const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;
#else
    const int window_k0_begin = 0;
#endif

#if IS_CAUSAL && BIDIR_MASK
    // Per-query image-group bounds [begin, end), the ONLY state the mask loop needs to un-mask a
    // bidirectional pair. The allowed key set of a query is
    //     (causal n sliding-window)  u  (the query's own image group)
    // -- see the reference in openvino/reference/paged_attention.hpp -- so the group membership of
    // the KEY never has to be looked up: every position in [begin, end) is in the query's group by
    // construction. sdpa_micro instead re-derives it per (query, key) pair with an O(|q - k|) scan
    // over token_type_ids inside its mask loop; here the loop below runs ONCE per workgroup, keeps
    // 2 * kq_query_blocks ints in registers, and leaves the hot loop with two integer compares.
    //
    // Both scans are clamped to the key loop's own window, which is exact rather than approximate:
    // keys below window_k0_begin are never visited by the k0 loop, and keys at or above causal_k
    // already read as -INFINITY through k_mask, so widening either bound cannot change a score. The
    // two extensions above make the clamps no-ops for real inputs anyway; they are the safety net.
    // An empty range (0, 0) is the "not an image token" encoding -- no key satisfies it, and it stays
    // valid whatever query_position_offset is, because the offset is only applied to a REAL group.
    //
    // The scan runs on token_type_ids, i.e. in LOCAL space, so the window bounds are mapped back
    // through key = query_position_offset + local and then clipped to the [0, q) range the buffer
    // actually covers. The stored result goes back to KEY space, which is what the mask loop compares
    // `key` against. PREFILL keeps its old form: query_position_offset folds to 0.
    const int bidir_scan_lo = max(0, window_k0_begin - query_position_offset);
    const int bidir_scan_hi = min(q, causal_k - query_position_offset);
    int bidir_group_begin[kq_query_blocks];
    int bidir_group_end[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        // Same expression the mask loop below uses for `query`, so the per-lane mapping cannot drift.
        const int query = (int)(wg_j0 + sg_j0_kq) + qb * SUBGROUP_SIZE + (int)lane;
        int group_begin = 0;
        int group_end = 0;
        if (bidir_active && query < q && token_type_ids[query] == 1) {
            group_begin = query;
            while (group_begin > bidir_scan_lo && token_type_ids[group_begin - 1] == 1)
                --group_begin;
            group_end = query + 1;
            while (group_end < bidir_scan_hi && token_type_ids[group_end] == 1)
                ++group_end;
            group_begin += query_position_offset;
            group_end += query_position_offset;
        }
        bidir_group_begin[qb] = group_begin;
        bidir_group_end[qb] = group_end;
    }
#endif

#if PA_CUR_KV_F16
    // Where the paged cache stops being the right place to read K/V from.
    //
    // The cache holds the whole key range (pa_kv_cache_update runs before this stage), so reading
    // everything from it is CORRECT -- it is just expensive: a compressed page costs a page read plus
    // a nibble/byte extract, a zero-point subtract and a scale multiply per element, ~250 extra
    // instructions per subgroup per k0 tile at head 64, against ~256 cycles of dpas in the same tile.
    // The keys at or above past_len are this iteration's NEW tokens, which are ALSO sitting in Kc/Vc
    // as exact f16, where a plain 2D block read hands the dpas its operand with no dequant at all.
    // sdpa_micro's MIXED kernel has always split its key loop this way (ugemm_kq on the pages below
    // past_len, ugemm_kcq on Kc above it) and that -- not the tiling, the SLM or the causal bound,
    // which all match -- is the whole 2.09x it won by on llama-3.2-1b's 1059-token prefill.
    //
    // Rounded up to the CACHE PAGE, which is the finest granularity that stays exact: a page is 16
    // consecutive keys and is written as a unit, so the page holding past_len is the only one that can
    // mix cached and new tokens, and it has to come from the cache (which holds both). Every page below
    // this bound is wholly cached, every page at or above it is wholly new.
    //
    // The page, NOT kq_wg_tile_keys, because the decision is made per page on both sides: a subgroup's K
    // tile is key_base = k0 + kq_sg_tile_keys * sg_i_kq, and one S*V cp block is
    // k0 + SUBGROUP_SIZE * cp -- both page-aligned, both exactly one page wide at the default tiling.
    // Rounding to the 128-key WG tile instead would drag up to kq_wg_tile_keys - 1 new tokens onto the
    // dequant path per query block, which at the measured past_len of 32 was 96 wasted keys x 17 query
    // blocks: 875 page-units of work against the 753 this bound gives (~171 us vs ~148 us).
    //
    // past_len == 0 collapses this to 0 and the cache path disappears entirely, which is what a
    // MIXED-stage dispatch of a plain prefill reduces to.
    //
#if PA_CUR_KV_GRAN
    const int pa_key_end =
        ((past_len + PAGED_ATTENTION_BLOCK_SIZE - 1) / PAGED_ATTENTION_BLOCK_SIZE) * PAGED_ATTENTION_BLOCK_SIZE;
#else
    const int pa_key_end = min(((past_len + kq_wg_tile_keys - 1) / kq_wg_tile_keys) * kq_wg_tile_keys, causal_k);
#endif
#endif

    for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == window_k0_begin);
        const bool last = (k0 + kq_wg_tile_keys >= causal_k);
#if IS_PAGED_ATTENTION && !IS_PREFILL
    #if PA_CUR_KV_F16
        // Where THIS SUBGROUP's K tile comes from. Per tile, not per k0 iteration: key_base is
        // page-aligned, so `key_base < pa_key_end` is true exactly when the tile's first key is cached,
        // and a tile that straddles the bound (only possible if kq_sg_tile_keys exceeds the page) then
        // takes the cache for all of its keys -- correct, since the cache holds the new tokens too.
        //
        // Subgroup-uniform (past_len, k0, sg_i0_kq and the Kc base parity are all uniform), so every
        // branch on it is uniform and costs an untaken jump, not divergence. That is why this is one loop
        // with two load paths rather than sdpa_micro's two loops -- the body is ~700 lines and
        // duplicating it would double the compile time and the instruction footprint for no gain.
        #if IS_PA_K_U4
        const bool from_cache = ((PA_CUR_KV_SIDE & 1) == 0) || (key_base < pa_key_end) || !kc_dword_ok;
        #else
        const bool from_cache = ((PA_CUR_KV_SIDE & 1) == 0) || (key_base < pa_key_end);
        #endif
    #else
        // Folds away, so every `if (from_cache)` guard below collapses back to its pre-change form.
        const bool from_cache = true;
    #endif
#endif

        float8 S_tile[kq_key_blocks][kq_query_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

#ifdef KV_COMPRESSED
        // Per-token K scale/zp depend only on the key (shared across the head dim), so load them
        // ONCE per k0-tile with a subgroup-cooperative wide load (lane L -> key key_base+L) and
        // keep them in registers. The old code fetched K_scales[KEY_COMP_OFF(...)]/K_zp[...] inside
        // the db x mb x key_offset dequant loop; since that offset is key-only (db-independent) and
        // lane-uniform, IGC emitted a per-key SIMD-1 (1|M0) scalar load, reloaded every db -> ~128
        // such loads per k0 iter (measured in the GEN ISA). This collapses them to one 16-wide load
        // each (mirrors the V vs_c/vz_c pattern and the k_mask lane=key layout below).
        // scale/zp kept in HALF for the bias-trick dequant below. NOTE: half here does NOT hit
        // the GEN <2> widen penalty that made an earlier all-half K dequant slower — that penalty
        // is in the convert_float/(half) WIDEN of the int8 byte, which the bias trick eliminates
        // (it never widens via convert). zp folds the widen bias (+1152.0h) so the per-byte dequant
        // is just: reinterpret (0x6480 ^ byte) as half, subtract (zp+1152), multiply scale.
        half k_scale_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        #if KEY_ZERO_POINTS
        half k_zpb_lane[kq_sg_tile_keys / SUBGROUP_SIZE];   // zp + 1152.0h (bias-trick bias folded in)
        #endif
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
            const int sc_key = key_base + ii * SUBGROUP_SIZE + lane;
            const uint sc_off = k_comp_base + KEY_COMP_OFF(0, 0, sc_key, 0);
            k_scale_lane[ii] = (sc_key < k) ? convert_half(K_scales[sc_off]) : (half)0.0f;
            #if KEY_ZERO_POINTS
            k_zpb_lane[ii] = (sc_key < k) ? (convert_half(K_zp[sc_off]) + (half)1152.0h) : (half)1152.0h;
            #endif
        }
#endif

#if IS_PAGED_ATTENTION && !IS_PREFILL
        // Paged K/V live in 16-key cache pages reached through block_indices[]. That lookup depends
        // only on the key, so it is invariant in db (the head-dim chunk) AND constant across the 8
        // keys of one DPAS row-block: key_base = k0 + sg_i0_kq is a multiple of kq_sg_tile_keys and
        // k0 a multiple of kq_wg_tile_keys, both multiples of PAGED_ATTENTION_BLOCK_SIZE, so an
        // 8-key block starting at an 8-aligned offset from key_base never straddles a page.
        // The old code issued it from the innermost (db, mb, key_offset) position -- lane-uniform,
        // so IGC emitted a per-key SIMD-1 scalar load, DKS * kq_key_blocks * DPAS_ROWS of them per
        // k0 iteration (64 at head 64). Hoisting to one per row-block leaves kq_key_blocks (2).
        // Guarded by (mb_key0 < k) rather than the old per-key (key < k): key >= mb_key0, so
        // mb_key0 >= k implies no key in the block would have been read anyway -- equivalent, and
        // it keeps the lookup inside block_indices[] for this subsequence (key_base can run past k
        // in the final k0 tile, where an unguarded load would index past the allocated blocks).
        // Every hoist from here to the end of this block feeds the CACHE read only, so all of them are
        // under `if (from_cache)`: on a PA_CUR_KV_F16 tile the page lookup, the scale/zp fetch and the
        // whole-page read are all dead work, and the branch is workgroup-uniform.
        uint k_page[kq_key_blocks];
        if (from_cache) {
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                const int mb_key0 = key_base + mb * DPAS_ROWS;
                k_page[mb] = (mb_key0 < k) ? block_indices[base_block_index + mb_key0 / PAGED_ATTENTION_BLOCK_SIZE]
                                           : 0u;
            }
        }

    #if IS_PA_K_BY_CHANNEL
        // BY_CHANNEL's comp is indexed by CHANNEL, and this kernel's KQ A operand is K itself with
        // lane == head dim (see the DPAS call below: A = as_short8(k_raw[mb]), so lane is the depth
        // index and the 8 elements are the keys). So a channel's scale/zp is a plain PER-LANE scalar,
        // exactly like Q -- there is no sub_group_broadcast anywhere in the dequant, unlike BY_TOKEN
        // where sc/zp are per key, i.e. per ELEMENT within a lane, and every one of the 16 keys in a
        // page needs its own broadcast. The subtract/multiply count is identical; the broadcasts are
        // the whole difference, which is why BY_CHANNEL is the cheaper mode here too (it is cheaper in
        // sdpa_ocl_decode.cl as well, but for the opposite reason: there K is the B operand and the zp
        // term collapses to one key-independent scalar).
        //
        // The pairs are interleaved, so channel c's (scale, zp) is exactly one DWORD at [c], and a
        // uint subgroup block read at dword offset db * SUBGROUP_SIZE hands lane L the pair for
        // channel db * DPAS_K + L -- one message per (page, head-dim tile), nothing re-read in the
        // db loop below. The comp base is 64 B-aligned whenever HEAD_SIZE % 16 == 0 (page stride is
        // HEAD_SIZE * 20 and the offset HEAD_SIZE * 16), so the dword read is always aligned.
        //
        // Two guards, BOTH mandatory and neither needed by BY_TOKEN:
        //  - DKS is D_MAX / DPAS_K and D_MAX is HEAD_SIZE rounded UP to a power of two, so at head
        //    48/80/96 the last tiles address channels past HEAD_SIZE. The comp region is exactly
        //    HEAD_SIZE dwords, so an unguarded read walks into the NEXT page's data rows -- or past
        //    the whole cache for the last page. BY_TOKEN indexes comp by token, always < 16, so it
        //    never had this. The `db < HEAD_SIZE / DPAS_K` test is compile-time in this unrolled loop,
        //    and the per-lane `else` only survives for a HEAD_SIZE that is not a multiple of DPAS_K.
        //  - A key group at/past k had its page index clamped to 0 above, and page 0's comp bytes are
        //    arbitrary in that case. sc = zp = 0 makes the dequant produce a finite 0; a NaN would
        //    NOT be discarded, because the mask below ADDS -INFINITY to the score and
        //    (NaN + -INFINITY) is NaN. Rows of a partially-filled page read as 0 and dequant to
        //    (0 - zp_c) * sc_c != 0, which is fine -- finite garbage plus -INFINITY is -INFINITY --
        //    so all this path has to guarantee is FINITENESS, which kv_cache_update's by-channel
        //    range-expansion guard provides for any page holding at least one written token.
        half k_pa_sc_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        half k_pa_zp_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            const global uint *k_comp_ch = (const global uint *)(
                K + ((size_t)k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)] * KV_HEADS_NUM + b0_kv) *
                        PA_K_PAGE_STRIDE +
                PA_K_COMP_OFF);
            const bool sc_valid = (key_base + kg * SUBGROUP_SIZE) < k;
        #if IS_PA_K_U4
            // The permuted depth labelling makes this CHEAPER than the i8 one, not dearer: tiles 2g
            // and 2g+1 want the comp dwords for channels (win + 2L) and (win + 2L + 1), which are
            // ADJACENT, so one uint2 per-lane load covers the whole pair. Across the subgroup that is
            // 16 lanes x 8 bytes over one contiguous 128-byte span -- fully coalesced, and half as
            // many messages as i8's one block read per tile.
            // Same two guards as i8: the window can run past HEAD_SIZE when HEAD_SIZE is not a
            // multiple of 2*DPAS_K (the comp region is exactly HEAD_SIZE dwords, so an unguarded read
            // walks into the next page), and a key group at/past k had its page clamped to 0, whose
            // comp bytes are then arbitrary -- sc = zp = 0 keeps the dequant finite, which is all the
            // masked-out score needs.
            #pragma unroll
            for (int g = 0; g < DKS_ACTIVE / 2; ++g) {
                const int u4_win = g * (2 * DPAS_K);
                uint2 pair2 = (uint2)(0u, 0u);
                if (u4_win + 2 * DPAS_K <= HEAD_SIZE) {
                    if (sc_valid)
                        pair2 = vload2(lane, k_comp_ch + u4_win);
                } else {
                    const int c0 = u4_win + 2 * (int)lane;
                    pair2.s0 = (sc_valid && c0 < HEAD_SIZE) ? k_comp_ch[c0] : 0u;
                    pair2.s1 = (sc_valid && c0 + 1 < HEAD_SIZE) ? k_comp_ch[c0 + 1] : 0u;
                }
                const half2 sc_zp0 = as_half2(pair2.s0);
                const half2 sc_zp1 = as_half2(pair2.s1);
                k_pa_sc_ch[kg][2 * g + 0] = sc_zp0.s0;
                k_pa_zp_ch[kg][2 * g + 0] = sc_zp0.s1;
                k_pa_sc_ch[kg][2 * g + 1] = sc_zp1.s0;
                k_pa_zp_ch[kg][2 * g + 1] = sc_zp1.s1;
            }
        #else
            #pragma unroll
            for (int db = 0; db < DKS_ACTIVE; ++db) {
                uint pair = 0u;
                if (db < HEAD_SIZE / DPAS_K) {
                    pair = sc_valid ? intel_sub_group_block_read(k_comp_ch + db * SUBGROUP_SIZE) : 0u;
                } else if (db * DPAS_K + (int)lane < HEAD_SIZE) {
                    pair = sc_valid ? k_comp_ch[db * SUBGROUP_SIZE + lane] : 0u;
                }
                const half2 sc_zp = as_half2(pair);
                k_pa_sc_ch[kg][db] = sc_zp.s0;
                k_pa_zp_ch[kg][db] = sc_zp.s1;
            }
        #endif
        }
        }
    #elif IS_PA_KV_COMPRESSED
        // Same hoist, applied to the i8 cache's per-key scale/zp. They live in the two trailing f16
        // arrays of the page and are indexed by TOKEN only -- independent of db (the head-dim chunk)
        // and of the head. Read from the innermost (db, mb, key_offset) position they are lane-uniform,
        // so IGC emitted one SIMD-1 (1|M0) load each: kq_key_blocks * DPAS_ROWS * 2 * DKS = 256 per k0
        // tile at head 128, which is why the mixed kernel's load count was dominated by scale/zp rather
        // than by K data (ISA: d16u32 272 vs d8u32 128). One subgroup-cooperative 16-wide load per
        // array per page replaces them; the dequant then recovers each key's scalar with a
        // sub_group_broadcast at a compile-time-constant lane, which folds into the consuming add/mul's
        // source region and costs no instruction. This is the same Stage-1 fix the plain-SDPA i8 path
        // already carries (k_scale_lane/k_zpb_lane above).
        //
        // Lane L holds token L of the group's page: one 16-key group is exactly one page (key_base is
        // a multiple of kq_sg_tile_keys and k0 of kq_wg_tile_keys, both multiples of
        // PAGED_ATTENTION_BLOCK_SIZE == SUBGROUP_SIZE), so token == the key's subgroup-local index.
        // A group at/past k had its page index clamped to 0 above, so the address is always in bounds
        // (page 0's comp region is allocated). Keys at/past k are nevertheless forced to scale = 0,
        // like the plain-SDPA k_scale_lane path: kv_cache_update never wrote those slots, so their
        // scale/zp bytes are arbitrary and could decode to NaN -- and the block-read branch below has
        // no per-key guard to discard them with (rows past the surface height read as 0, but
        // (0 - NaN) * NaN is NaN, which would survive the masked-out score). zp is zeroed too so the
        // product is 0 for any finite quantized byte.
        half k_pa_sc_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        half k_pa_zp_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            const global half *k_comp = (const global half *)(
                K + ((size_t)k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)] * KV_HEADS_NUM + b0_kv) *
                        PA_K_PAGE_STRIDE +
                PA_K_COMP_OFF);
            const bool sc_valid = (key_base + kg * SUBGROUP_SIZE + (int)lane) < k;
            k_pa_sc_lane[kg] = sc_valid ? k_comp[lane] : (half)0.0h;
            k_pa_zp_lane[kg] = sc_valid ? k_comp[PAGED_ATTENTION_BLOCK_SIZE + lane] : (half)0.0h;
        }
        }
    #endif

    #if USE_1D_BLOCK_IO_K_PA_U4
        // Whole-page read, hoisted OUT of the db loop below -- that hoist is the entire win. The
        // page bytes do not depend on db (a byte is a channel PAIR, so the tile pair (2g, 2g+1)
        // shares one byte and the four tiles at head 64 share two 32-channel windows), while the
        // gather it replaces re-issued DKS * kq_key_blocks * DPAS_ROWS == 64 messages per k0. Two
        // uc16 reads is the whole subgroup's K for this k0 iteration.
        // Live state is PA_PAGE_READS * 16 bytes per lane: 32 at head 64. It stays small because the
        // host gate only fires where block2d could not, which for u4 means a row of 16 or 32 bytes.
        uchar16 k_pg[kq_sg_tile_keys / SUBGROUP_SIZE][PA_PAGE_READS(PA_K_ROW_ELEMS)];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            // Same page as the comp loop above, and page-aligned for the same reason: key_base is a
            // multiple of kq_sg_tile_keys and k0 of kq_wg_tile_keys, both multiples of
            // PAGED_ATTENTION_BLOCK_SIZE. A group at/past k had its index clamped to 0, so the
            // address is always inside an allocated page.
            const global uchar *k_pg_base = (const global uchar *)(
                K + ((size_t)k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)] * KV_HEADS_NUM + b0_kv) *
                        PA_K_PAGE_STRIDE);
            #pragma unroll
            for (int r = 0; r < PA_PAGE_READS(PA_K_ROW_ELEMS); ++r)
                k_pg[kg][r] = intel_sub_group_block_read_uc16(k_pg_base + r * PA_PAGE_RD_BYTES);
        }
        }
    #endif
#endif

        #pragma unroll
        for (int db = 0; db < DKS_ACTIVE; ++db) {
            int8 qB[kq_query_blocks];
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int q_block = sg_j0_kq / SUBGROUP_SIZE + qb;
                qB[qb] = as_int8(intel_sub_group_block_read8(
                    (local void *)&Q_slm[((db * q_blocks + q_block) * Q_DWORDS) * SUBGROUP_SIZE]));
            }

            ushort8 k_raw[kq_key_blocks];
#if IS_PAGED_ATTENTION && !IS_PREFILL
            if (from_cache) {
    #if USE_2D_BLOCK_IO_K_PA
            // Token-major K cache: a (block, kv_head) page is a
            // [PAGED_ATTENTION_BLOCK_SIZE keys x HEAD_SIZE head dims] ROW-MAJOR tile -- exactly the
            // [key, head] geometry the prefill branch below reads, just with the page as the surface
            // origin and HEAD_SIZE (not ldk, which spans a whole page) as the pitch. So the same
            // non-transform 16b builtin applies and lands the A operand as lane=head / elem=key,
            // replacing the DPAS_ROWS-per-block per-key SIMD-1 scalar loads with one block message
            // per 16-key group. Mirrors USE_2D_BLOCK_IO_V_PA in the S*V loop below.
            //
            // One read covers SUBGROUP_SIZE keys == one full page (kq_sg_tile_keys is 16 by default,
            // but SDPA_OCL_KQ_TILE_KEYS can raise it to 32), so loop per 16-key group and take that
            // group's page from the k_page[] lookup hoisted above -- index mb = kg * (SUBGROUP_SIZE /
            // DPAS_ROWS), i.e. the first row-block of the group, since k_page is indexed per
            // row-block. Each group is page-aligned: key_base is a multiple of kq_sg_tile_keys and
            // k0 of kq_wg_tile_keys, both multiples of PAGED_ATTENTION_BLOCK_SIZE.
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
                const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
                // Surface height is clamped to the keys this page actually holds (k - kg_key0, at
                // most a full page): slots at or past k were never written by kv_cache_update, and a
                // NaN pulled from there would survive the masked-out score as NaN. Rows past the
                // height read as 0. A group entirely at/past k (height <= 0) is not a legal block
                // read, so it is zero-filled instead -- reachable because the key loop is a full
                // unroll over the WG tile and key_base can run past k on the final k0 iteration.
                const int kp_rows = min((int)PAGED_ATTENTION_BLOCK_SIZE, k - kg_key0);
                if (kp_rows > 0) {
                    // PA_K_PAGE_STRIDE by value (the #if above proves the two ADJUSTED_* collapse for
                    // an uncompressed cache), but spelled out so the generated code is bit-identical to
                    // what it was before the macro existed: IGC strength-reduces (x * 16) * HEAD_SIZE
                    // and x * (16 * HEAD_SIZE) differently when HEAD_SIZE is not a power of two (one
                    // shl becomes one mov at head 48/96), and this path must stay byte-identical so the
                    // ISA A/B can serve as evidence that nothing but BY_CHANNEL was touched.
                    const global half *Kp =
                        (const global half *)(K + (((size_t)k_page[kg_mb] * KV_HEADS_NUM + b0_kv) *
                                                   PAGED_ATTENTION_BLOCK_SIZE * HEAD_SIZE));
                    const int KP_w = d * (int)sizeof(half);
                    const int KP_p = HEAD_SIZE * (int)sizeof(half);
                    intel_sub_group_2d_block_read_16b_16r16x1c(
                        (global void *)Kp, KP_w, kp_rows, KP_p,
                        (int2)(db * DPAS_K, 0), (private ushort *)&k_raw[kg_mb]);
                } else {
                    #pragma unroll
                    for (int mb = 0; mb < SUBGROUP_SIZE / DPAS_ROWS; ++mb)
                        k_raw[kg_mb + mb] = (ushort8)0;
                }
            }
    #elif USE_2D_BLOCK_IO_K_PA_I8
            // Token-major i8 K cache, EITHER quant mode: BY_CHANNEL's data region is byte-identical in
            // geometry to BY_TOKEN's (that is the point of the token-major BY_CHANNEL layout), so the
            // read below is shared verbatim and only the comp source and the dequant's index differ.
            // The page's data region is a
            // [PAGED_ATTENTION_BLOCK_SIZE tokens x HEAD_SIZE head dims] ROW-MAJOR i8 tile with a
            // HEAD_SIZE-BYTE row pitch (128 at head 128, so >= 64 and % 64 -- checked host-side),
            // which is exactly the geometry the V cache already block-reads below. So the same
            // 8-bit VNNI-transform builtin applies and lands the A operand as lane=head with each
            // uint packing 4 consecutive keys as bytes.
            //
            // The builtin has a hard 32-row minimum on Xe2 (no _8b_16r form exists), while a page is
            // only PAGED_ATTENTION_BLOCK_SIZE tokens, so the surface height is clamped to the keys
            // the page actually holds and only uints 0..3 are consumed; rows past the height read
            // as 0. A group entirely at/past k is not a legal block read, so it is zero-filled --
            // reachable because the key loop is a full unroll over the WG tile. Pairing two key
            // groups into one read is NOT possible: consecutive groups live in non-adjacent pages.
            //
            // Dequant reuses the scale/zp hoisted above -- the k_pa_sc_lane/k_pa_zp_lane broadcasts for
            // BY_TOKEN, the plain per-lane k_pa_sc_ch/k_pa_zp_ch for BY_CHANNEL -- and the same
            // shift/mask byte extract as the plain-SDPA USE_2D_BLOCK_IO_K_I8 path, which avoids the
            // `:b` deinterleave. Kept as an explicit (q - zp) * scale in half rather than the
            // bias-trick form so the arithmetic stays identical to the scalar branch below, which is
            // what makes SDPA_OCL_K_PA_I8_2D=0 a clean bisection toggle. (The bias trick would also
            // have to round zp to an f16 whose ulp at 1152 is 1.0, and the writer's zp is not an
            // integer, so it would roughly double the quantization error -- see sdpa_ocl_decode.cl.)
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
                const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
                #pragma unroll
                for (int mb = 0; mb < SUBGROUP_SIZE / DPAS_ROWS; ++mb)
                    k_raw[kg_mb + mb] = (ushort8)0;
                const int kp_rows = min((int)PAGED_ATTENTION_BLOCK_SIZE, k - kg_key0);
                if (kp_rows > 0) {
                    uint kt[8];
                    #if IS_PA_K_U4
                    // Same builtin, half the surface: the row is PA_K_ROW_ELEMS bytes and a byte
                    // column is a CHANNEL PAIR, so one read at byte column PA_K_U4_WIN(db)/2 covers
                    // 32 channels == the tile pair (db, db^1) under the permuted labelling. The
                    // partner tile re-issues the identical read and the L1 absorbs it; folding the
                    // pair into one read would have to hoist k_raw out of the db loop, which at
                    // head 128 is 128 extra ushorts of live state -- deferred, exactly as the V-tile
                    // pairing is in sdpa_ocl_decode.cl.
                    // x is in elements (bytes here) and PA_K_U4_WIN(db)/2 is a multiple of
                    // SUBGROUP_SIZE, which satisfies the spec's "multiple of four for 8-bit data".
                    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                        (global void *)(K + ((size_t)k_page[kg_mb] * KV_HEADS_NUM + b0_kv) *
                                                PA_K_PAGE_STRIDE),
                        PA_K_ROW_ELEMS, kp_rows, PA_K_ROW_ELEMS, (int2)(PA_K_U4_WIN(db) / 2, 0),
                        (private uint *)&kt[0]);
                    #else
                    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                        (global void *)(K + ((size_t)k_page[kg_mb] * KV_HEADS_NUM + b0_kv) *
                                                PA_K_PAGE_STRIDE),
                        d, kp_rows, HEAD_SIZE, (int2)(db * DPAS_K, 0), (private uint *)&kt[0]);
                    #endif
                    #if IS_PA_K_BY_CHANNEL
                    // Per-channel scale/zp are per LANE, so they leave the key loop entirely: one pair
                    // for the whole (page, head-dim tile) instead of BY_TOKEN's broadcast per key.
                    const half k_sc = k_pa_sc_ch[kg][db];
                    const half k_zp = k_pa_zp_ch[kg][db];
                    #endif
                    #pragma unroll
                    for (int u = 0; u < SUBGROUP_SIZE / 4; ++u) {
                        const uint w = kt[u];
                        #pragma unroll
                        for (int bb = 0; bb < 4; ++bb) {
                            const int krel = kg * SUBGROUP_SIZE + u * 4 + bb;
                            #if !IS_PA_K_BY_CHANNEL
                            const half k_sc = sub_group_broadcast(k_pa_sc_lane[kg], u * 4 + bb);
                            const half k_zp = sub_group_broadcast(k_pa_zp_lane[kg], u * 4 + bb);
                            #endif
                            #if IS_PA_K_U4
                            // The nibble select is lane-UNIFORM (the parity is the tile's, not the
                            // lane's), so it folds into the shift amount rather than a per-lane sel.
                            // Unsigned by construction: the int4 quantizer clamps to [0, 15] with
                            // zp = -min*scale, so there is no CHAR_MIN and no sign extension.
                            const uint kb_ = (w >> (bb * 8)) & 0xFFu;
                            const half deq_k =
                                ((half)(PA_K_U4_PAR(db) ? (kb_ >> 4) : (kb_ & 0x0Fu)) - k_zp) * k_sc;
                            #else
                            const half deq_k = ((half)(char)((w >> (bb * 8)) & 0xFFu) - k_zp) * k_sc;
                            #endif
                            k_raw[krel / DPAS_ROWS][krel % DPAS_ROWS] = as_ushort(deq_k);
                        }
                    }
                }
            }
    #elif USE_1D_BLOCK_IO_K_PA_U4
            // Byte-for-byte the same dequant, the same guards and the same k_raw writes as the
            // scalar branch below -- ONLY the load changes, from one message per (db, mb, key) to a
            // register subscript into the page read hoisted above. That is what makes
            // SDPA_OCL_K_PA_1D=0 an exact bisection toggle: the two branches are numerically
            // identical, so a difference can only come from the read itself.
            //
            // The column group is PA_K_U4_WIN(db) / 2 / SUBGROUP_SIZE == db >> 1, a compile-time
            // constant in this unrolled loop, so both PA_PAGE_R and PA_PAGE_I fold and no base bias
            // is needed. `head` is still the CHANNEL (the guard below is unchanged); it just no
            // longer has to be turned into an address.
            // The scalar branch's per-key `head < d && key < k` guard is NOT reproduced, for the same
            // reason the block2d branches above do not have one either -- and dropping it is most of
            // the win, since it costs a compare and a predicated write per key:
            //  - `key < k`. A key at or past k is also at or past causal_k (causal_k <= k), so the
            //    mask below ADDS -INFINITY to its score. All this path therefore owes is FINITENESS,
            //    and a u4 nibble is bounded to [0, 15] by construction while sc/zp were already
            //    clamped to 0 for a key group at/past k. The block2d branches lean on exactly this:
            //    they clamp the surface height, and rows past it read as 0, which still dequants to
            //    a nonzero (0 - zp) * sc. Note this is STRONGER than the i8 case needs to be -- there
            //    is no NaN to exclude, because the nibble select bounds the value before the dequant.
            //  - `head < d`. Folded into the scale instead: (nibble - zp) * 0 is exactly 0, so one
            //    select per (mb, db) replaces DPAS_ROWS of them, and k_raw needs no zero init.
            const int head = PA_K_U4_CHANNEL(db, lane);
            const int k_pg_col = (PA_K_U4_WIN(db) / 2) / SUBGROUP_SIZE;
            const bool head_ok = (head < d);
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                const int kg = mb / (SUBGROUP_SIZE / DPAS_ROWS);
                // Per-channel comp is this lane's own and constant across the row-block's keys,
                // exactly as in the scalar branch.
                const half k_sc = head_ok ? k_pa_sc_ch[kg][db] : (half)0.0h;
                const half k_zp = k_pa_zp_ch[kg][db];
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int krel = mb * DPAS_ROWS + key_offset;   // key's subgroup-local index
                    const int tok = krel % PAGED_ATTENTION_BLOCK_SIZE;
                    const uint kb_ = (uint)k_pg[kg][PA_PAGE_R(PA_K_ROW_ELEMS, tok, k_pg_col)]
                                                  [PA_PAGE_I(PA_K_ROW_ELEMS, tok, k_pg_col)];
                    const half deq_k =
                        ((half)(PA_K_U4_PAR(db) ? (kb_ >> 4) : (kb_ & 0x0Fu)) - k_zp) * k_sc;
                    k_raw[mb][key_offset] = as_ushort(deq_k);
                }
            }
    #elif IS_PA_KV_COMPRESSED
            // i8 K cache, per-key scalar gather. Serves two situations: a d-major page (data at
            // head_dim * PAGED_ATTENTION_BLOCK_SIZE + token, whose PAGED_ATTENTION_BLOCK_SIZE-byte row
            // is far below the 64 B block2d minimum, so a gather is the only correct load -- the same
            // reason the f16 d-major branch below gathers), and a token-major page whose HEAD_SIZE byte
            // pitch misses the block2d rule (head 32/48/80/96). PA_K_TOKEN_STRIDE /
            // PA_K_HIDDEN_STRIDE select the addressing, so one code path covers both.
            // Comp is hoisted above: BY_TOKEN's one scale/zp PER KEY in the two f16 arrays that follow
            // the data region, or BY_CHANNEL's one interleaved pair PER CHANNEL.
            // Dequant is (q - zp) * scale, done in half: the writer stores 1/scale, so the value read
            // back IS the multiplier. The bias-trick used by the plain-SDPA i8 paths is deliberately
            // not used here -- it pays off when amortised over a wide transform read, whereas this
            // gather already costs one message per key, and (q - zp) * scale on a plain char->half
            // convert keeps the arithmetic identical to the reference dequant.
            #if IS_PA_K_U4
            // Permuted depth (see PA_K_U4_CHANNEL): head IS still the channel index, so the `head < d`
            // guard below is unchanged, but two channels share a byte so the ADDRESS is head >> 1.
            // Consecutive lanes then hit consecutive bytes -- better coalesced than the natural
            // labelling, where lane pairs would collide on one byte.
            const int head = PA_K_U4_CHANNEL(db, lane);
            const int head_addr = head >> 1;
            #else
            const int head = db * DPAS_K + lane;
            const int head_addr = head;
            #endif
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                // Page base for this row-block, hoisted above; only the intra-page key offset and
                // the head vary here.
                const size_t mb_page_base =
                    ((size_t)k_page[mb] * KV_HEADS_NUM + b0_kv) * PA_K_PAGE_STRIDE +
                    (size_t)head_addr * PA_K_HIDDEN_STRIDE;
                #if IS_PA_K_BY_CHANNEL
                // head == db * DPAS_K + lane here too, so the per-channel pair is this lane's own and
                // is constant across the row-block's keys: no broadcast, and it lifts out of the key
                // loop. A row-block maps to key group mb / (SUBGROUP_SIZE / DPAS_ROWS).
                const half k_sc = k_pa_sc_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
                const half k_zp = k_pa_zp_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
                #endif
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int krel = mb * DPAS_ROWS + key_offset;   // key's subgroup-local index
                    const int key = key_base + krel;
                    #if !IS_PA_K_BY_CHANNEL
                    // sub_group_broadcast is a subgroup COLLECTIVE, so it must run on every lane --
                    // keep it outside the per-lane-divergent (head < d) guard below. krel is a
                    // compile-time constant in this fully unrolled loop, so the broadcast folds into
                    // the consuming add/mul source region rather than emitting a shuffle.
                    const half k_sc = sub_group_broadcast(k_pa_sc_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                    const half k_zp = sub_group_broadcast(k_pa_zp_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                    #endif
                    if (head < d && key < k) {
                        // key_base is a multiple of PAGED_ATTENTION_BLOCK_SIZE, so the key's token
                        // within its page is just its subgroup-local index.
                        const int tok = krel % PAGED_ATTENTION_BLOCK_SIZE;
                        #if IS_PA_K_U4
                        const uint kb_ = (uint)(uchar)K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE];
                        const half deq_k =
                            ((half)(PA_K_U4_PAR(db) ? (kb_ >> 4) : (kb_ & 0x0Fu)) - k_zp) * k_sc;
                        #else
                        const half deq_k = ((half)K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE] - k_zp) * k_sc;
                        #endif
                        k_raw[mb][key_offset] = as_ushort(deq_k);
                    }
                }
            }
    #else
            const int head = db * DPAS_K + lane;
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                // Page base for this row-block, hoisted above; only the intra-page key offset and
                // the head vary here.
                const size_t mb_page_base =
                    ((size_t)k_page[mb] * KV_HEADS_NUM + b0_kv) * PA_K_PAGE_STRIDE +
                    (size_t)head * PA_K_HIDDEN_STRIDE;
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int key = key_base + mb * DPAS_ROWS + key_offset;
                    if (head < d && key < k) {
                        const int tok = key % PAGED_ATTENTION_BLOCK_SIZE;
                        k_raw[mb][key_offset] = as_ushort(K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE]);
                    }
                }
            }
    #endif
            }
    #if PA_CUR_KV_F16
            else {
        #if IS_PA_K_U4
                // The u4 cache forces a PERMUTED dpas depth axis -- tile db wants lane L to hold
                // channel PA_K_U4_WIN(db) + 2*L + PA_K_U4_PAR(db), because a page byte IS a channel
                // pair -- and Q_slm is staged to match. A 16b block read cannot produce that stride-2
                // gather, so read Kc as a DWORD surface instead: channels (win + 2L) and (win + 2L + 1)
                // are ADJACENT halves, i.e. exactly one dword at dword-column win/2 + L, so a plain 32b
                // read lands the pair in lane L and the parity is a half-select. Same reinterpretation
                // the Q staging already does (transpose_32b at x = u4_win/2); the surface width/pitch
                // stay in BYTES, only x is in dwords.
                //
                // 8 rows == one mb (DPAS_ROWS keys), 16 dwords == 64 B, both inside the plain read's
                // limits. Per db this reads the window twice (once per parity) and uses half of each,
                // which is 2x L1 traffic on a tile that is already in cache -- and still ~4x fewer
                // instructions than the page path's nibble extract + zp subtract + scale multiply.
                //
                // This keeps the permutation confined to a dword half-select, so neither the tuned
                // cache path above nor the Q_slm layout changes at all.
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb) {
                    uint kw[DPAS_ROWS];
                    intel_sub_group_2d_block_read_32b_8r16x1c(
                        (global void *)Kc_b2d, KcD_w_b2d, KcD_h, KcD_p,
                        (int2)(KcD_x0_dw + PA_K_U4_WIN(db) / 2,
                               key_base + mb * DPAS_ROWS - past_len),
                        (private uint *)&kw[0]);
                    #pragma unroll
                    for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset)
                        k_raw[mb][key_offset] = PA_K_U4_PAR(db) ? (ushort)(kw[key_offset] >> 16)
                                                               : (ushort)kw[key_offset];
                }
        #else
                // f16 / i8 cache: no depth permutation, so this is the plain-SDPA [key, head] read
                // verbatim, just pointed at Kc with a (key - past_len) row origin. Rows past q read as
                // zero, which is what the `key < k` masking already assumes.
                #pragma unroll
                for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                    intel_sub_group_2d_block_read_16b_16r16x1c(
                        (global void *)Kc_b2d, KcD_w_b2d, KcD_h, KcD_p,
                        (int2)(KcD_x0 + db * DPAS_K, key_base + kg * SUBGROUP_SIZE - past_len),
                        (private ushort *)&k_raw[kg * (SUBGROUP_SIZE / DPAS_ROWS)]);
                }
        #endif
            }
    #endif
#elif USE_2D_BLOCK_IO_K_I8
            // int8 K via the 8-bit VNNI-transform read (same builtin as V). Reading K's row-major
            // [key, head] memory at (x=db*DPAS_K head-col, y=key_base) gives lane=head with each
            // uint packing 4 consecutive keys as bytes (GPU-probed: lane==head exactly, key order
            // u*4+b linear -> no subgroup shuffle, unlike the earlier non-transform K attempt).
            // One read spans 32 keys; this subgroup uses the first kq_sg_tile_keys (16) = uints 0..3.
            // Dequant reuses the hoisted per-key scale/zp broadcasts (Step 1); result is the f16
            // A operand k_raw[mb][key_offset], mb=krel/8, key_offset=krel%8.
            {
                uint kt[8];
                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                    (global void *)K, KD_w, KD_h, KD_p,
                    (int2)(db * DPAS_K, key_base), (private uint *)&kt[0]);
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb)
                    k_raw[mb] = (ushort8)0;
                // kq_sg_tile_keys keys = kq_sg_tile_keys/4 uints. head=64/128 => 16 keys => uints 0..3.
                // Bias-trick dequant (microbench-validated, mov -69% vs the convert_float widen):
                // extract each key byte with shift+mask (NO as_char4 -> no <4;1,0> :b deinterleave),
                // widen via the denormal-bias reinterpret (0x6480 ^ byte) as half, then the folded
                // (zp+1152) subtract and scale multiply -- all in half.
                #pragma unroll
                for (int u = 0; u < kq_sg_tile_keys / 4; ++u) {
                    const uint w = kt[u];
                    #pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int krel = u * 4 + bb;           // key's subgroup-local index 0..kq_sg_tile_keys-1
                        const ushort wbits = (ushort)0x6480 ^ (ushort)((w >> (bb * 8)) & 0xFFu);
                        const half wide = as_half(wbits);
                        const half k_sc = sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                        #if KEY_ZERO_POINTS
                            const half k_zpb = sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                            const half deq_k = (wide - k_zpb) * k_sc;
                        #else
                            const half deq_k = (wide - (half)1152.0h) * k_sc;
                        #endif
                        k_raw[krel / 8][krel % 8] = as_ushort(deq_k);
                    }
                }
            }
#elif USE_2D_BLOCK_IO_KV
            // The _16r builtin returns exactly 16 key rows == 2 key-blocks of 8. A subgroup owns
            // kq_sg_tile_keys keys == kq_key_blocks blocks, so issue one read per 16-key group
            // instead of assuming a single read covers the whole tile: with kq_sg_tile_keys == 32
            // (kq_key_blocks == 4) a lone read filled only k_raw[0..1] and left k_raw[2..3]
            // uninitialised, silently corrupting S for the upper half of the tile.
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                intel_sub_group_2d_block_read_16b_16r16x1c(
                    (global void *)K_b2d, KD_w_b2d, KD_h, KD_p,
                    (int2)(KD_x0 + db * DPAS_K, key_base + kg * SUBGROUP_SIZE),
                    (private ushort *)&k_raw[kg * (SUBGROUP_SIZE / DPAS_ROWS)]);
            }
#else
            const int head = db * DPAS_K + lane;
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                #pragma unroll
                for (int key_offset = 0; key_offset < 8; ++key_offset) {
                    const int key = key_base + mb * 8 + key_offset;
                    #ifdef KV_COMPRESSED
                        // i8 compressed K: per-token (per-kv-head) asymmetric dequant. Scale/zp
                        // are the hoisted per-key values (lane=key wide load above); recover this
                        // key's scalar with a subgroup broadcast. krel is the key's subgroup-local
                        // index (compile-time constant here), so the broadcast folds to a register
                        // move. It is a subgroup collective, so it MUST run on all lanes -> keep it
                        // OUTSIDE the per-lane-divergent (head < d) guard below.
                        const int krel = mb * 8 + key_offset;
                        const float k_sc = convert_float(sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                        #if KEY_ZERO_POINTS
                            // k_zpb_lane holds zp+1152.0h; recover the raw zp for this scalar path.
                            const float k_zp = convert_float(sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE)) - 1152.0f;
                        #endif
                    #endif
                    if (head < d && key < k) {
                        #ifdef KV_COMPRESSED
                            #if KEY_ZERO_POINTS
                                const float deq_k = (convert_float(K[(size_t)key * ldk + head]) - k_zp) * k_sc;
                            #else
                                const float deq_k = convert_float(K[(size_t)key * ldk + head]) * k_sc;
                            #endif
                            k_raw[mb][key_offset] = as_ushort((half)deq_k);
                        #else
                            k_raw[mb][key_offset] = as_ushort(K[(size_t)key * ldk + head]);
                        #endif
                    }
                }
            }
#endif

            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                #pragma unroll
                for (int qb = 0; qb < kq_query_blocks; ++qb)
                    S_tile[mb][qb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(k_raw[mb]), qB[qb], S_tile[mb][qb]);
            }
        }

        half2 mask_tile;
        float2 k_mask;
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
            const int key = key_base + ii * SUBGROUP_SIZE + lane;
            #if WITH_ATTN_MASK
                if (MASK_IS_PER_KEY)
                    mask_tile[ii] = (key < k) ? msk[MSK_OFF(0, 0, 0, key)] : (half)0.0f;
                else
                    mask_tile[ii] = (half)0.0f;
            #else
                mask_tile[ii] = (half)0.0f;
            #endif
            // Bound against causal_k, not k: the key loop stops at causal_k but its LAST tile can
            // overrun it (causal_k = past_len + wg_j0 + kq_wg_tile_queries is not tile-aligned once
            // past_len is arbitrary, which is the norm in the paged-attention mixed stage). Keys in
            // [causal_k, k) are past every query this workgroup owns, so they must read as -inf.
            // They do get -inf from the causal mask below too, but only as long as that mask
            // actually runs: BLOCK_SKIP_CAUSAL elides it for blocks it proves fully in-region. That
            // proof cannot currently cover such a block, so this is equivalence-preserving -- it
            // just stops the remainder from depending on the block-skip predicate. sdpa_micro has
            // always bounded its k_mask this way (k0 + sg_i0_kq + ... < causal_k).
            k_mask[ii] = (key < causal_k) ? 0.0f : -INFINITY;
        }
        float2 mask_tile_float = convert_float2(mask_tile);
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            mask_tile_float[ii] = mask_tile_float[ii] * iscale;

        #if WITH_ATTN_MASK
            // Full 2D mask [query x key]: each lane loads its own query row (strided,
            // same access pattern as sdpa_micro's tile_load_t). Pre-scale by iscale at
            // load time and keep it as float so the softmax max-loop below only does a
            // branchless add (mirrors micro's tile_elementwise(unscale)+tile_binary add).
            float16 mask_full[kq_query_blocks][kq_sg_tile_keys / SUBGROUP_SIZE];
            if (MASK_IS_FULL_2D) {
                #pragma unroll
                for (int qb = 0; qb < kq_query_blocks; ++qb) {
                    const int mask_query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    #pragma unroll
                    for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
                        const int mask_key = key_base + ii * SUBGROUP_SIZE;
                        half16 mv = (half16)0.0f;
                        if (mask_query < q) {
                            if (mask_key + SUBGROUP_SIZE <= k) {
                                mv = vload16(0, msk + MSK_OFF(0, 0, mask_query, mask_key));
                            } else {
                                #pragma unroll
                                for (int kk = 0; kk < SUBGROUP_SIZE; ++kk) {
                                    if (mask_key + kk < k)
                                        mv[kk] = msk[MSK_OFF(0, 0, mask_query, mask_key + kk)];
                                }
                            }
                        }
                        mask_full[qb][ii] = convert_float16(mv) * iscale;
                    }
                }
            }
        #endif

        float alpha[kq_query_blocks];
        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            float lmax = -INFINITY;
            // Whether this subgroup's (key x query) block can touch the causal boundary at all.
            // Keys run [key_base, key_base + kq_sg_tile_keys), queries run
            // [wg_j0 + sg_j0_kq + qb*SUBGROUP_SIZE, + SUBGROUP_SIZE), so if the block's LAST key is
            // <= its FIRST query every element is inside the causal region and the per-element
            // predicate is a no-op. Measured at q = k = 4096 with the default tiling: 96.6% of
            // blocks are in that class, yet the old code still issued cmp+sel for all 16 keys in
            // every one of them -- 32 sel + 8 cmp per (qb, k0) iteration, and the causal-mask
            // region was 16% of the whole loop body. sdpa_micro has always had this block-level
            // skip (its `if (causal_k_end > causal_q_begin)` guard around
            // tile_predicated_assignment_t); this is the ocl counterpart.
            // causal_block_clear is uniform across the subgroup (no lane term), so IGC turns the
            // branch into straight-line code for the common case rather than a per-lane select.
#if IS_CAUSAL
    #if BLOCK_SKIP_CAUSAL
            const int blk_key_last = key_base + kq_sg_tile_keys - 1;
            const int blk_query_first = query_position_offset + (int)(wg_j0 + sg_j0_kq) + qb * SUBGROUP_SIZE;
        #if SLIDING_WINDOW_SIZE
            // With a window the block must also sit fully inside it: the oldest key the block's
            // LAST query may attend is (blk_query_last - SLIDING_WINDOW_SIZE), so the block's
            // FIRST key must be newer than that.
            const int blk_query_last = blk_query_first + SUBGROUP_SIZE - 1;
            const bool causal_block_clear =
                blk_key_last <= blk_query_first && key_base > blk_query_last - SLIDING_WINDOW_SIZE;
        #else
            const bool causal_block_clear = blk_key_last <= blk_query_first;
        #endif
    #else
            // BLOCK_SKIP_CAUSAL=0 keeps the original always-mask behaviour, so a wrong-result
            // config can be bisected against this optimisation.
            const bool causal_block_clear = false;
    #endif
#endif
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                #pragma unroll
                for (int mm = 0; mm < 8; ++mm) {
                    const int key_rel = mb * 8 + mm;
                    const int mask_idx = key_rel / SUBGROUP_SIZE;
                    const int mask_lane = key_rel - mask_idx * SUBGROUP_SIZE;
                    const int query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    const int query_position = query_position_offset + query;
                    const int key = key_base + key_rel;
                    float s = S_tile[mb][qb][mm] + sub_group_broadcast(k_mask[mask_idx], mask_lane);
#ifdef STATIC_SCALAR_ATTN_MASK_VALUE
                    s += STATIC_SCALAR_ATTN_MASK_VALUE * iscale;
#endif
                    #if WITH_ATTN_MASK
                        if (MASK_IS_PER_KEY) {
                            s += sub_group_broadcast(mask_tile_float[mask_idx], mask_lane);
                        } else if (MASK_IS_FULL_2D) {
                            s += mask_full[qb][mask_idx][mask_lane];
                        } else if (query < q && key < k) {
                            const int mask_query = (MSK_D2 == 1) ? 0 : query;
                            const int mask_key = (MSK_D3 == 1) ? 0 : key;
                            s += convert_float(msk[MSK_OFF(0, 0, mask_query, mask_key)]) * iscale;
                        }
                    #endif
#if IS_CAUSAL
                    if (!causal_block_clear) {
    #if SLIDING_WINDOW_SIZE
                        // Keys outside (query - SLIDING_WINDOW_SIZE, query] are dropped, matching
                        // sdpa_micro's greater_than() predicate.
                        if (key > query_position || key <= query_position - SLIDING_WINDOW_SIZE) {
    #else
                        if (key > query_position) {
    #endif
    #if IS_CAUSAL && BIDIR_MASK
                            // ...unless the key is inside this query's own image group, which is
                            // bidirectional. Only reachable when the base predicate above already
                            // masked, so this can only ever un-mask -- which is also why the
                            // causal_block_clear skip stays sound: it proves the block is wholly
                            // INSIDE the causal+window region, where there is nothing to un-mask.
                            if (key < bidir_group_begin[qb] || key >= bidir_group_end[qb])
    #endif
                            s = -INFINITY;
                        }
                    }
#endif
                    S_tile[mb][qb][mm] = s;
                    lmax = fmax(lmax, s);
                }
            }

            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            __builtin_IB_atomic_max_local_f32(&S_max_slm[query], lmax);
        }

    #if MAX_BARRIER_V_PREFETCH && USE_2D_BLOCK_IO_KV && !(IS_PAGED_ATTENTION && !IS_PREFILL)
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                intel_sub_group_2d_block_prefetch_16b_16r16x1c(
                    (const global void *)V_b2d, VD_w_b2d, VD_h, VD_p,
                    (int2)(VD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE));
            }
        }
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    #else
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            const float m_new = S_max_slm[query];
            // Required when a query has no valid keys in the current prefix, e.g. future
            // remainder/causal/window masks or a fully masked row. In that case m_new is
            // -inf, and unguarded max rescaling would form -inf - -inf and poison S/A.
            const bool ok = isfinite(m_new);
            const float m_log2 = ok ? m_new * scale : 0.0f;
            const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;
            float lsum = 0.0f;

            S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
            alpha[qb] = a;

            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                lsum += exp_tile[0] + exp_tile[1] + exp_tile[2] + exp_tile[3]
                      + exp_tile[4] + exp_tile[5] + exp_tile[6] + exp_tile[7];

                const int key = sg_i0_kq + mb * 8;
                const int key_block = key / SUBGROUP_SIZE;
                const int key_lane = key - key_block * SUBGROUP_SIZE;
                const int s_half_offset = (key_block * kq_wg_tile_queries + query) * SUBGROUP_SIZE + key_lane;
                vstore4(as_uint4(convert_half8(exp_tile)), 0, &S_slm[s_half_offset >> 1]);
            }
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
        }

        if (last) {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                S_sum_slm[query * kq_sg_per_wg_keys + sg_i_kq] = S_sum_tile[qb];
            }
        }

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        if (!first) {
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                float8 av;
                const int rel_query = sg_i0_sv + r * 8 - sg_j0_kq;
                const int alpha_qb = rel_query / SUBGROUP_SIZE;
                const int alpha_lane0 = rel_query - alpha_qb * SUBGROUP_SIZE;
                // alpha_qb is a RUNTIME value (sg_i0_sv and sg_j0_kq come from sg_ij), so reading
                // alpha[alpha_qb] directly is a dynamically indexed private array -- and IGC answers
                // that by putting the whole array in scratch. Measured on the llama-3.2-1b MIXED
                // kernel: `private memory size 128` (== kq_query_blocks floats x 16 lanes x 4 B, which
                // is the TPM=128 cliloader reports) plus 2 scratch stores and 2 scratch loads INSIDE
                // the k0 loop, i.e. per iteration.
                //
                // kq_query_blocks is a compile-time constant, so pick the element with a select chain
                // instead. Same value, no dynamic index, array stays in GRF. The subgroup broadcast's
                // lane argument stays runtime -- that is an indirect register move, not scratch.
                float alpha_sel = alpha[0];
                #pragma unroll
                for (int t = 1; t < kq_query_blocks; ++t)
                    alpha_sel = (t == alpha_qb) ? alpha[t] : alpha_sel;
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr)
                    av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr);
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= av;
            }
        }

        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        #if USE_2D_BLOCK_IO_V_I8
            // Declared outside the cp loop because with V_I8_PAIRED_READ one read serves two
            // consecutive cp blocks (see below). Unpaired, the def-use pattern is unchanged.
            uint vt[8 * sv_value_blocks];
        #endif
#if IS_PAGED_ATTENTION && !IS_PREFILL && PA_CUR_KV_F16 && PA_CUR_KV_GRAN
        // Keep the Vc collective read out of the cp-varying source branch. On BMG, both the
        // transform and ordinary 2D reads produce wrong results when early unrolled cp copies take
        // the cache arm and later copies take the Vc arm, although either arm alone is correct.
        // Only the one tile crossing pa_key_end takes this workaround. Uniform Vc tiles retain the
        // faster transform read below, and cache-only tiles do not touch Vc.
        const bool v_mixed_source_tile = ((PA_CUR_KV_SIDE & 2) != 0) &&
                         (k0 < pa_key_end) &&
                         (k0 + sv_key_blocks * SUBGROUP_SIZE > pa_key_end);
#endif
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
#if IS_PAGED_ATTENTION && !IS_PREFILL
    #if PA_CUR_KV_F16
            // The V side needs its OWN cache/Vc decision: the K one is keyed on key_base, which comes
            // from the KQ key split (sg_i0_kq), while here the keys come from cp -- a different mapping
            // over the same k0 tile, so reusing the K flag would read the wrong source for most
            // subgroups. One cp block is exactly one cache page and cp_key0 is page-aligned, so this
            // test is exact for the same reason the K one is.
            const bool v_from_cache = ((PA_CUR_KV_SIDE & 2) == 0) ||
                                      ((k0 + cp * SUBGROUP_SIZE) < pa_key_end);
    #else
            const bool v_from_cache = true;
    #endif
#endif
            #if USE_2D_BLOCK_IO_V_I8
                // One _8b_32r16x1c read covers a fixed 16 value columns, while a subgroup owns
                // sv_sg_tile_values == sv_value_blocks * SUBGROUP_SIZE of them, so issue one read
                // per cd, stepping x by SUBGROUP_SIZE. Reads are kept ahead of the S_slm (pA)
                // reads below so the global-memory latency overlaps with the SLM traffic.
                // Value columns past d only exist when d < D_MAX; the block read clamps them to 0
                // and the store guard (out_col < d) drops the corresponding A_tile columns.
                //
                // The builtin returns 32 key rows (uints 0..7, 4 rows each) but one cp block is
                // only SUBGROUP_SIZE == 16 keys, so a per-cp read discards half of every message
                // and consecutive cp reads overlap by 16 rows. V_I8_PAIRED_READ issues the read
                // on even cp only and lets it serve two blocks -- uints 0..3 for cp, uints 4..7
                // for cp+1 -- halving the V message count. cp is a full-unroll constant, so both
                // vt_do_read and vt_half fold at compile time: no branch and no dynamic vt index
                // survive. An odd sv_key_blocks needs no tail case; the last even cp simply uses
                // uints 0..3 exactly as the unpaired form does.
                #if V_I8_PAIRED_READ
                    const bool vt_do_read = ((cp & 1) == 0);
                    const int vt_half = (cp & 1) * 4;
                #else
                    const bool vt_do_read = true;
                    const int vt_half = 0;
                #endif
                if (vt_do_read) {
                    // The multi-block x2c / x4c variants fetch this subgroup's whole 32 / 64
                    // value columns in ONE message. GPU-probed on B580 (see
                    // test/microbench/probe_v_multiblock) their destination is BLOCK-MAJOR --
                    // uint u carries block u/8, key (u%8)*4+b, value (u/8)*16+lane -- which is
                    // bit-identical to what the x1c loop below writes into &vt[cd * 8], so the
                    // dequant indexing needs no change. coord.x must be a multiple of 4 for
                    // 8-bit data: sg_j0_sv is a multiple of sv_sg_tile_values (16/32/64), so
                    // that holds. The x1c loop stays as the fallback for any sv_value_blocks
                    // the extension has no single-message variant for.
                    #if V_I8_MULTIBLOCK_READ && sv_value_blocks == 2
                        intel_sub_group_2d_block_read_transform_8b_32r16x2c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE),
                            (private uint *)&vt[0]);
                    #elif V_I8_MULTIBLOCK_READ && sv_value_blocks == 4
                        intel_sub_group_2d_block_read_transform_8b_32r16x4c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE),
                            (private uint *)&vt[0]);
                    #else
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                                (global void *)V, VD_w, VD_h, VD_p,
                                (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE),
                                (private uint *)&vt[cd * 8]);
                        }
                    #endif
                }
            #endif

            short8 pA[sv_score_blocks];
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                const int query0 = sg_i0_sv + r * 8;
                pA[r] = as_short8(intel_sub_group_block_read_us8(
                    (local void *)&S_slm[((cp * kq_wg_tile_queries + query0) * SUBGROUP_SIZE) >> 1]));
            }

            #if USE_2D_BLOCK_IO_V_I8
                // int8 V uses the 8-bit VNNI-transform read below. Per-token V scale depends only
                // on the key (not the value/head index), and pA (the score operand just read above)
                // is already lane=key — same layout as vs_c below — so the scale is folded into pA
                // directly with a per-lane multiply (no broadcast) instead of into V, which would
                // require broadcasting vs_c across the value/head-dim lanes in the dequant loop.
                // zp is a subtraction, not a scalar factor, so it stays on the V side (still needs
                // its per-key value broadcast into the value/head-dim lanes there).
                const int vs_key = k0 + cp * SUBGROUP_SIZE + lane;
                const uint vs_co = v_comp_base + VAL_COMP_OFF(0, 0, vs_key, 0);
                // Keep scale/zp in half: V_scales/V_zp are already half, and the dequant is
                // stored as half — half arithmetic is bit-identical to the float path over the
                // int8 range (verified), so this avoids the half->float->half round trips.
                const half vs_c = (vs_key < k) ? V_scales[vs_co] : (half)0.0f;
                #if VAL_ZERO_POINTS
                    // Fold the bias-trick widen bias (+1152.0h) into zp: the V dequant below widens
                    // via as_half(0x6480 ^ byte) (== signed_byte + 1152), so subtracting (zp+1152)
                    // gives (signed_byte - zp) with no convert_half widen. OOB keys -> vzb_c=1152
                    // (zp=0), and the score-side scale (vs_c=0 for OOB) still zeroes the product.
                    const half vzb_c = (vs_key < k) ? (convert_half(V_zp[vs_co]) + (half)1152.0h) : (half)1152.0h;
                #endif

                #pragma unroll
                for (int r = 0; r < sv_score_blocks; ++r)
                    pA[r] = as_short8(as_half8(pA[r]) * vs_c);
            #elif IS_PA_KV_COMPRESSED && !IS_PREFILL
                // Same scale/zp split as the plain-SDPA i8 path above, but the per-key scale and zp
                // come from INSIDE the V page rather than from separate tensors: two f16 arrays at
                // HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE, indexed [token] then [block_size + token].
                // A cp block is exactly one page, so lane == the key's token index within the page.
                // The writer stores 1/scale, so the value read back is already the multiplier.
                // Scale is folded into pA (lane=key, so no broadcast needed); zp stays on the V side.
                // OOB keys get scale 0, which zeroes their contribution regardless of the V bytes.
                // Cache-only: a PA_CUR_KV_F16 tile reads V from Vc, which is plain f16 with no scale and
                // no zero point, so neither the page comp fetch nor the pA fold may happen there.
                // v_zp_c stays in scope (the dequant below references it) and is left at 0.
                half v_zp_c = (half)0.0f;
                if (v_from_cache) {
                    const int vs_key_pa = k0 + cp * SUBGROUP_SIZE + lane;
                    const size_t vs_page_pa =
                        (size_t)((vs_key_pa < k) ? block_indices[base_block_index +
                                                                 vs_key_pa / PAGED_ATTENTION_BLOCK_SIZE]
                                                 : 0u) *
                            KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE +
                        (size_t)b0_kv * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE;
                    const global half *v_comp_pa =
                        (const global half *)(V + vs_page_pa + PA_V_COMP_OFF);
                    const int vs_tok_pa = vs_key_pa % PAGED_ATTENTION_BLOCK_SIZE;
                    const half vs_c_pa = (vs_key_pa < k) ? v_comp_pa[vs_tok_pa] : (half)0.0f;
                    v_zp_c = (vs_key_pa < k) ? v_comp_pa[PAGED_ATTENTION_BLOCK_SIZE + vs_tok_pa]
                                             : (half)0.0f;

                    #pragma unroll
                    for (int r = 0; r < sv_score_blocks; ++r)
                        pA[r] = as_short8(as_half8(pA[r]) * vs_c_pa);
                }
            #endif

            int8 vb[sv_value_blocks];
#if IS_PAGED_ATTENTION && !IS_PREFILL && PA_CUR_KV_F16 && PA_CUR_KV_GRAN
            if (v_mixed_source_tile) {
                // Cache-side cps in the crossing tile read row zero and then overwrite vb below.
                // Current-token cps use their real Vc row. Both coordinates are subgroup-uniform.
                const int vc_row = max(k0 + cp * SUBGROUP_SIZE - past_len, 0);
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_16b_16r16x1c(
                        (global void *)Vc_b2d, VcD_w_b2d, VcD_h, VcD_p,
                        (int2)(VcD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE, vc_row),
                        (private ushort *)&vb[cd]);
                }
            }
#endif
            #if IS_PAGED_ATTENTION && !IS_PREFILL
            if (v_from_cache) {
                // One cp block is exactly SUBGROUP_SIZE (== DPAS_K == PAGED_ATTENTION_BLOCK_SIZE)
                // consecutive keys starting at k0 + cp * SUBGROUP_SIZE, and k0 is a multiple of
                // kq_wg_tile_keys, so the block coincides with ONE cache page. The page lookup is
                // therefore invariant across BOTH the cd (value-column) and key_pair loops -- the
                // old code repeated it for every key of every cd, i.e. sv_value_blocks * DPAS_ROWS
                // * 2 lane-uniform SIMD-1 loads per cp; this leaves one.
                const int cp_key0 = k0 + cp * SUBGROUP_SIZE;
                // Page stride is ADJUSTED_V_HEAD_SIZE, which is v_head_size for an uncompressed
                // cache (so this is unchanged for f16) and v_head_size + 4 for an i8 one, where the
                // extra 4 bytes hold the page's trailing scale/zp arrays. The DATA row pitch stays
                // HEAD_SIZE in both cases -- the +4 is at the end of the page, not inside each row.
                const size_t v_page_base =
                    (size_t)((cp_key0 < k) ? block_indices[base_block_index +
                                                           cp_key0 / PAGED_ATTENTION_BLOCK_SIZE]
                                           : 0u) *
                    KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE +
                    (size_t)b0_kv * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE;
                #if IS_PA_KV_COMPRESSED
                    // i8 V cache. The cp block coincides with exactly one page, so the page's token
                    // index IS the key's subgroup-local index: token == lane for the per-key scale/zp,
                    // and == key_rel for the dequant below. scale/zp are the two f16 arrays at
                    // HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE, indexed [token] and [block_size + token].
                    // Scale is folded into pA above (score side, already lane=key) exactly as the
                    // plain-SDPA i8 path does, so only the zp subtraction happens here -- which does
                    // need its per-key value broadcast across the value/head-dim lanes.
                    #if USE_2D_BLOCK_IO_V_PA_I8
                    {
                        // Data region is a [PAGED_ATTENTION_BLOCK_SIZE tokens x HEAD_SIZE values]
                        // row-major i8 tile: pitch = HEAD_SIZE bytes (128 at head 128, so the >= 64
                        // and % 64 block2d rule holds -- checked host-side).
                        // The 8b transform builtin has a hard 32-row minimum on Xe2 (no _8b_16r form
                        // exists), while a page is only 16 tokens, so the surface height is clamped to
                        // the tokens the page actually holds and only uints 0..3 are consumed; rows
                        // past the height read as 0. Pairing two cp blocks into one read (what
                        // V_I8_PAIRED_READ does on the flat prefill surface) is NOT possible here:
                        // consecutive key groups live in different, non-adjacent pages.
                        const int vp_rows = min((int)PAGED_ATTENTION_BLOCK_SIZE, k - cp_key0);
                        uint vt_pa[8 * sv_value_blocks];
                        if (vp_rows > 0) {
                            #if IS_PA_K_U4
                            // One byte per value PAIR, so the row is PA_V_ROW_ELEMS bytes wide.
                            const int VP_w = PA_V_ROW_ELEMS;
                            const int VP_p = PA_V_ROW_ELEMS;
                            #else
                            const int VP_w = d;                  // bytes: i8, one byte per value
                            const int VP_p = HEAD_SIZE;          // bytes: data row pitch, NOT ADJUSTED
                            #endif
                            #pragma unroll
                            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                                const int vcol = sg_j0_sv + cd * SUBGROUP_SIZE;
                                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                                    (global void *)(V + v_page_base), VP_w, vp_rows, VP_p,
                                    // u4 folds the upper half of the head dim back onto its low twin;
                                    // the nibble select below picks which one this tile wants. Both
                                    // the base and PA_V_ROW_ELEMS are multiples of SUBGROUP_SIZE, so a
                                    // 16-lane tile never straddles the split. Identity for i8.
                                    (int2)(PA_V_U4_COL(vcol), 0),
                                    (private uint *)&vt_pa[cd * 8]);
                            }
                        } else {
                            #pragma unroll
                            for (int u = 0; u < 8 * sv_value_blocks; ++u)
                                vt_pa[u] = 0u;
                        }
                        // zp broadcasts are per-key and independent of the value index, so hoist them
                        // out of the cd loop (once per cp block instead of once per (cd, u) pair).
                        half4 vzp4[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int k0r = u * 4;
                            vzp4[u] = (half4)(sub_group_broadcast(v_zp_c, k0r + 0),
                                              sub_group_broadcast(v_zp_c, k0r + 1),
                                              sub_group_broadcast(v_zp_c, k0r + 2),
                                              sub_group_broadcast(v_zp_c, k0r + 3));
                        }
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            #if IS_PA_K_U4
                            // Which nibble this tile's head dims live in. Uniform across the subgroup
                            // (the split point is a multiple of SUBGROUP_SIZE), so it folds into the
                            // shift amount rather than a per-lane select.
                            const int v_hi = PA_V_U4_HI(sg_j0_sv + cd * SUBGROUP_SIZE);
                            #endif
                            #pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint w = vt_pa[cd * 8 + u];
                                // Each uint packs 4 consecutive tokens as signed bytes, token u*4+b
                                // in byte b -- the same packing the plain-SDPA i8 V path decodes.
                                #if IS_PA_K_U4
                                const half4 q4 = (half4)((half)((w >> (v_hi ?  4 :  0)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 12 :  8)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 20 : 16)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 28 : 24)) & 0x0Fu));
                                #else
                                const half4 q4 = (half4)((half)(char)((w >>  0) & 0xFFu),
                                                         (half)(char)((w >>  8) & 0xFFu),
                                                         (half)(char)((w >> 16) & 0xFFu),
                                                         (half)(char)((w >> 24) & 0xFFu));
                                #endif
                                const half4 deq4 = q4 - vzp4[u];
                                // f16 VNNI operand: vb[cd][key_pair] packs keys (2*kp, 2*kp+1), and
                                // deq4 already holds keys u*4..u*4+3 in order, so .lo/.hi are exactly
                                // key_pairs (u*2, u*2+1).
                                vb[cd][u * 2 + 0] = as_int(deq4.lo);
                                vb[cd][u * 2 + 1] = as_int(deq4.hi);
                            }
                        }
                    }
                    #elif USE_1D_BLOCK_IO_V_PA_U4
                    {
                        // Same dequant, same guards and the same vb writes as the scalar branch
                        // below -- only the load changes, so SDPA_OCL_V_PA_1D=0 is an exact
                        // bisection toggle. Replaces sv_value_blocks * DPAS_ROWS * 2 == 16 gathers
                        // per cp block with PA_PAGE_READS == 2 messages.
                        //
                        // The column group PA_V_U4_COL(v_base) / SUBGROUP_SIZE is NOT a compile-time
                        // constant (v_base comes from sg_j0_sv), so the base is biased by it and the
                        // index is taken at c = 0 -- see the PA_PAGE_* comment. That makes the read
                        // per-cd rather than per-cp; sv_value_blocks is 1 for every head size this
                        // path can fire on (larger ones satisfy the block2d pitch rule), so the
                        // message count is unchanged by that.
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            vb[cd] = (int8)0;
                            const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                            const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
                            // Nibble select as a UNIFORM SHIFT rather than a select. v_hi comes from
                            // sg_j0_sv, so it is subgroup-uniform but not a compile-time constant
                            // (unlike K's PA_K_U4_PAR(db), which folds): written as
                            // `v_hi ? (b >> 4) : (b & 0xF)` it costs shr+and+sel on every one of the
                            // sv_key_blocks * DPAS_ROWS * 2 elements a subgroup dequants per k0.
                            // Hoisting the shift amount leaves shr+and.
                            const uint v_sh = PA_V_U4_HI(v_base) ? 4u : 0u;
                            uchar16 v_pg[PA_PAGE_READS(PA_V_ROW_ELEMS)];
                            const global uchar *v_pg_base =
                                (const global uchar *)(V + v_page_base) + PA_V_U4_COL(v_base);
                            #pragma unroll
                            for (int r = 0; r < PA_PAGE_READS(PA_V_ROW_ELEMS); ++r)
                                v_pg[r] = intel_sub_group_block_read_uc16(v_pg_base + r * PA_PAGE_RD_BYTES);
                            // No per-key `key < k` guard, mirroring the block2d branch above and the
                            // K branch: a key at or past k already carries a score of exactly 0 out
                            // of the softmax (its logit was -INFINITY), so its V value is multiplied
                            // by zero and only has to be FINITE -- which a u4 nibble is by
                            // construction, and v_zp_c is clamped to 0 there anyway.
                            if (value < d) {
                                #pragma unroll
                                for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                                    // The cp block coincides with one page (see above), so the
                                    // token index IS the key's block-local index. Spelled as the
                                    // loop constant rather than key0 % PAGED_ATTENTION_BLOCK_SIZE
                                    // because PA_PAGE_R/I need it at compile time, and IGC cannot
                                    // prove cp_key0 % PAGED_ATTENTION_BLOCK_SIZE == 0 by itself.
                                    const int t0 = key_pair * 2;
                                    const int t1 = t0 + 1;
                                    const uint vb0 = (uint)v_pg[PA_PAGE_R(PA_V_ROW_ELEMS, t0, 0)]
                                                               [PA_PAGE_I(PA_V_ROW_ELEMS, t0, 0)];
                                    const uint vb1 = (uint)v_pg[PA_PAGE_R(PA_V_ROW_ELEMS, t1, 0)]
                                                               [PA_PAGE_I(PA_V_ROW_ELEMS, t1, 0)];
                                    half2 vv;
                                    vv[0] = (half)((vb0 >> v_sh) & 0x0Fu) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    vv[1] = (half)((vb1 >> v_sh) & 0x0Fu) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    vb[cd][key_pair] = as_int(vv);
                                }
                            }
                        }
                    }
                    #else
                    // Scalar-gather fallback for the i8 cache (SDPA_OCL_V_PA_I8_2D=0). Same dequant,
                    // one message per value per key pair -- this is what attributes a wrong result to
                    // the block read rather than to the dequant math.
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        vb[cd] = (int8)0;
                        const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                        #if IS_PA_K_U4
                        // Two head dims share a byte, so the address is the folded byte column plus
                        // the lane; the nibble is the TILE's, hence uniform across the subgroup.
                        const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
                        const int v_hi = PA_V_U4_HI(v_base);
                        const int v_addr = PA_V_U4_COL(v_base) + (int)lane;
                        #endif
                        if (value < d) {
                            #pragma unroll
                            for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                                const int key0 = cp_key0 + key_pair * 2;
                                const int key1 = key0 + 1;
                                half2 vv = (half2)0.0h;
                                if (key0 < k) {
                                    const int t0 = key0 % PAGED_ATTENTION_BLOCK_SIZE;
                                    #if IS_PA_K_U4
                                    const uint vb0 = (uint)(uchar)V[v_page_base + (size_t)t0 * PA_V_ROW_ELEMS + v_addr];
                                    vv[0] = (half)(v_hi ? (vb0 >> 4) : (vb0 & 0x0Fu)) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    #else
                                    vv[0] = (half)(char)V[v_page_base + (size_t)t0 * HEAD_SIZE + value] -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    #endif
                                }
                                if (key1 < k) {
                                    const int t1 = key1 % PAGED_ATTENTION_BLOCK_SIZE;
                                    #if IS_PA_K_U4
                                    const uint vb1 = (uint)(uchar)V[v_page_base + (size_t)t1 * PA_V_ROW_ELEMS + v_addr];
                                    vv[1] = (half)(v_hi ? (vb1 >> 4) : (vb1 & 0x0Fu)) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    #else
                                    vv[1] = (half)(char)V[v_page_base + (size_t)t1 * HEAD_SIZE + value] -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    #endif
                                }
                                vb[cd][key_pair] = as_int(vv);
                            }
                        }
                    }
                    #endif
                #elif USE_2D_BLOCK_IO_V_PA
                    // The cp block coincides with one (block, head) cache page, and that page is a
                    // [PAGED_ATTENTION_BLOCK_SIZE tokens x HEAD_SIZE values] ROW-MAJOR tile -- the
                    // same [key, value] geometry the prefill path reads, just with the page as the
                    // surface origin and HEAD_SIZE (not HEAD_SIZE*KV_HEADS_NUM) as the pitch. So the
                    // 16b VNNI-transform builtin applies unchanged and lands the operand in the
                    // layout the DPAS below wants, replacing the sv_value_blocks * DPAS_ROWS * 2
                    // per-lane scalar loads with sv_value_blocks coalesced messages.
                    //
                    // Surface height is clamped to the tokens this page actually holds
                    // (k - cp_key0, at most a full page) instead of the full page: cache slots at or
                    // past k were never written by kv_cache_update, so reading them could pull in
                    // uninitialised bits -- and a NaN there would survive the `0 * garbage` score
                    // multiply as NaN rather than vanishing. Rows past the surface height read as 0,
                    // which is the same guarantee the prefill V/A paths already depend on for their
                    // k and d remainders (see the width note in the i8 branch below).
                    // The cp loop is a full unroll over the whole WG key tile, so a block can sit
                    // ENTIRELY at/past k (k - cp_key0 <= 0) on the final k0 iteration. A surface
                    // height of 0 or less is not a valid block-read argument, so such a block is
                    // zero-filled directly and the read is skipped. cp_key0 is a full-unroll
                    // constant only in cp, not in k, so this stays a real (uniform) branch.
                    const int vp_rows = min((int)PAGED_ATTENTION_BLOCK_SIZE, k - cp_key0);
                    if (vp_rows > 0) {
                        const global half *Vp = (const global half *)(V + v_page_base);
                        const int VP_w = d * (int)sizeof(half);
                        const int VP_p = HEAD_SIZE * (int)sizeof(half);
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                                (global void *)Vp, VP_w, vp_rows, VP_p,
                                (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, 0), (private uint *)&vb[cd]);
                        }
                    } else {
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd)
                            vb[cd] = (int8)0;
                    }
                #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    vb[cd] = (int8)0;
                    const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                    if (value < d) {
                        #pragma unroll
                        for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                            const int key0 = cp_key0 + key_pair * 2;
                            const int key1 = key0 + 1;
                            half2 vv = (half2)0.0h;
                            if (key0 < k) {
                                vv[0] = V[v_page_base +
                                          (size_t)(key0 % PAGED_ATTENTION_BLOCK_SIZE) * HEAD_SIZE + value];
                            }
                            if (key1 < k) {
                                vv[1] = V[v_page_base +
                                          (size_t)(key1 % PAGED_ATTENTION_BLOCK_SIZE) * HEAD_SIZE + value];
                            }
                            vb[cd][key_pair] = as_int(vv);
                        }
                    }
                }
                #endif
            }
            #if PA_CUR_KV_F16
                #if PA_CUR_KV_GRAN
            else if (!v_mixed_source_tile) {
                #else
            else {
                #endif
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)Vc_b2d, VcD_w_b2d, VcD_h, VcD_p,
                        (int2)(VcD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE,
                               k0 + cp * SUBGROUP_SIZE - past_len),
                        (private uint *)&vb[cd]);
                }
            }
            #endif
            #elif USE_2D_BLOCK_IO_V_I8
                // int8 V via 8-bit VNNI-transform read: one coalesced read gives a 32-key x
                // 16-value tile (lane=value, each uint packs 4 consecutive keys as bytes). We
                // need this cp-block's 16 keys, which are the 4 uints at vt_half (0 for an even
                // cp, 4 for the odd cp that reuses the previous read). Dequant each byte (per-key
                // scale via the cached vs_c broadcast) and repack into the f16 VNNI operand
                // (2 half-keys per int), matching the scalar vb layout with no subgroup shuffle.
                {
                    // this cp-block = 16 keys = 4 uints (4 keys each). key_rel = u*4 + b.
                    // Bias-trick dequant (microbench-validated, mov -66% vs convert_half4):
                    // extract each key byte with shift+mask (NO as_char4 -> no <4;1,0> :b
                    // deinterleave), widen via the denormal-bias reinterpret (0x6480 ^ byte)
                    // == signed_byte+1152 as half. scale is folded into pA above (score side),
                    // so only the bias-folded zp subtraction (vzb=zp+1152) remains here.
                    #if VAL_ZERO_POINTS
                        // zp is per-key and independent of the value/head index, so the broadcasts
                        // are hoisted out of the cd loop: issued once per cp-block instead of once
                        // per (cd, u) pair.
                        half4 zpb4[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int k0r = u * 4;
                            zpb4[u] = (half4)(sub_group_broadcast(vzb_c, k0r + 0),
                                              sub_group_broadcast(vzb_c, k0r + 1),
                                              sub_group_broadcast(vzb_c, k0r + 2),
                                              sub_group_broadcast(vzb_c, k0r + 3));
                        }
                    #endif
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint w = vt[cd * 8 + vt_half + u];
                            const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >>  8) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 16) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
                            #if VAL_ZERO_POINTS
                                const half4 deq4 = wide4 - zpb4[u];
                            #else
                                const half4 deq4 = wide4 - (half4)((half)1152.0h);
                            #endif
                            // f16 VNNI operand: vb[cd][key_pair] packs keys (2*key_pair, 2*key_pair+1).
                            // deq4 already holds keys u*4..u*4+3 in order, so its .lo/.hi halves
                            // are exactly the two key_pairs (u*2, u*2+1) for this u — store
                            // straight into vb instead of round-tripping through an array.
                            vb[cd][u * 2 + 0] = as_int(deq4.lo);
                            vb[cd][u * 2 + 1] = as_int(deq4.hi);
                        }
                    }
                }
            #elif V_F16_MULTIBLOCK_READ && USE_2D_BLOCK_IO_KV && sv_value_blocks == 2
                intel_sub_group_2d_block_read_transform_16b_16r16x2c(
                    (global void *)V_b2d, VD_w_b2d, VD_h, VD_p,
                    (int2)(VD_x0 + sg_j0_sv, k0 + cp * SUBGROUP_SIZE), (private uint *)&vb[0]);
            #elif USE_2D_BLOCK_IO_KV
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)V_b2d, VD_w_b2d, VD_h, VD_p,
                        (int2)(VD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE),
                        (private uint *)&vb[cd]);
                }
            #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    vb[cd] = (int8)0;
                    const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                    if (value < d) {
                        #pragma unroll
                        for (int key_pair = 0; key_pair < 8; ++key_pair) {
                            const int key0 = k0 + cp * SUBGROUP_SIZE + key_pair * 2;
                            const int key1 = key0 + 1;
                            half2 vv = (half2)0.0h;
                            if (key0 < k) {
                                #ifdef KV_COMPRESSED
                                    // i8 compressed V: per-token (per-kv-head) asymmetric dequant.
                                    // Scale/zp vary per key (token), so they must be indexed by
                                    // key0/key1 here, not by the value (head-dim) index.
                                    const uint v_comp_off0 = VAL_COMP_OFF(b1, b0_kv, key0, 0);
                                    #if VAL_ZERO_POINTS
                                        vv[0] = (half)((convert_float(V[(size_t)key0 * ldv + value]) - convert_float(V_zp[v_comp_off0])) * convert_float(V_scales[v_comp_off0]));
                                    #else
                                        vv[0] = (half)(convert_float(V[(size_t)key0 * ldv + value]) * convert_float(V_scales[v_comp_off0]));
                                    #endif
                                #else
                                    vv[0] = V[(size_t)key0 * ldv + value];
                                #endif
                            }
                            if (key1 < k) {
                                #ifdef KV_COMPRESSED
                                    const uint v_comp_off1 = VAL_COMP_OFF(b1, b0_kv, key1, 0);
                                    #if VAL_ZERO_POINTS
                                        vv[1] = (half)((convert_float(V[(size_t)key1 * ldv + value]) - convert_float(V_zp[v_comp_off1])) * convert_float(V_scales[v_comp_off1]));
                                    #else
                                        vv[1] = (half)(convert_float(V[(size_t)key1 * ldv + value]) * convert_float(V_scales[v_comp_off1]));
                                    #endif
                                #else
                                    vv[1] = V[(size_t)key1 * ldv + value];
                                #endif
                            }
                            vb[cd][key_pair] = as_int(vv);
                        }
                    }
                }
            #endif

            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r)
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] = intel_sub_group_f16_f16_matrix_mad_k16(pA[r], vb[cd], A_tile[r][cd]);
        }
    }

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        float8 inv_l;
        #pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            const int query = sg_i0_sv + r * 8 + rr;
            float l = S_sum_slm[query * kq_sg_per_wg_keys + 0];
            #pragma unroll
            for (int p = 1; p < kq_sg_per_wg_keys; ++p)
                l += S_sum_slm[query * kq_sg_per_wg_keys + p];
            inv_l[rr] = (l > 0.0f) ? native_recip(l) : 0.0f;
        }
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            A_tile[r][cd] *= inv_l;
    }

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd) {
            half8 out = convert_half8(A_tile[r][cd]);
            const int col = sg_j0_sv + cd * SUBGROUP_SIZE;
            const int row = wg_j0 + sg_i0_sv + r * 8;
#if USE_2D_BLOCK_IO_A
            if (row + 7 < q && col + SUBGROUP_SIZE <= d) {
                intel_sub_group_2d_block_write_16b_8r16x1c(
                    (global void *)A, AD_w, AD_h, AD_p,
                    (int2)(col, row),
                    (private ushort *)&out);
            } else {
#endif
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const int out_row = row + rr;
                    const int out_col = col + lane;
                    if (out_row < q && out_col < d)
                        A[(size_t)out_row * lda + out_col] = out[rr];
                }
#if USE_2D_BLOCK_IO_A
            }
#endif
        }
    }
}
