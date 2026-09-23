// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compile-time configuration of sdpa_ocl.cl: derived tiling, dtype and layout macros, and the
// host/kernel invariants as #errors. Included once, by sdpa_ocl.cl only.

// 16-bit element type of Q/K/V/output (f16 or bf16). DT_* bodies are deliberately unparenthesised so
// each use expands to exactly the tokens the per-dtype code it replaced had.
#if INPUT0_IS_BF16
#  define DPAS_MAD_K16 intel_sub_group_bf16_bf16_matrix_mad_k16
#  define PACK_SOFTMAX8(x) _convert_bfloat168_as_ushort8(x)
#  define ACC_TO_OUT8(x)   _convert_bfloat168_as_ushort8(x)
#  define MASK_TO_FLOAT(x) _convert_as_bfloat16_float(as_ushort(x))
#  define MASK_TO_FLOAT2(x) _convert_as_bfloat162_float2(as_ushort2(x))
#  define MASK_TO_FLOAT16(x) _convert_as_bfloat1616_float16(as_ushort16(x))
#  define DT_OUT8_T            ushort8
#  define DT_ELEM2_T           ushort2
#  define DT_ELEM2_ZERO        (ushort2)0
#  define DT_FROM_F32(x)       _convert_bfloat16_as_ushort(x)
#  define DT_BITS_FROM_F32(x)  _convert_bfloat16_as_ushort(x)
#  define DT_FROM_RAW(x)       as_ushort(x)
#else
#  define DPAS_MAD_K16 intel_sub_group_f16_f16_matrix_mad_k16
#  define PACK_SOFTMAX8(x) convert_half8(x)
#  define ACC_TO_OUT8(x)   convert_half8(x)
#  define MASK_TO_FLOAT(x) convert_float(x)
#  define MASK_TO_FLOAT2(x) convert_float2(x)
#  define MASK_TO_FLOAT16(x) convert_float16(x)
#  define DT_OUT8_T            half8
#  define DT_ELEM2_T           half2
#  define DT_ELEM2_ZERO        (half2)0.0h
#  define DT_FROM_F32(x)       (half)(x)
#  define DT_BITS_FROM_F32(x)  as_ushort((half)x)
#  define DT_FROM_RAW(x)       x
#endif

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
// DKS is D_MAX / DPAS_K, and D_MAX is K_HEAD_SIZE rounded UP to a power of two, so whenever
// K_HEAD_SIZE is not itself a power-of-two multiple of DPAS_K the top tiles address channels past
// the head dim entirely: they load nothing (the `head < d` guard masks every lane) and their
// dpas adds zero. At K_HEAD_SIZE 72 -> D_MAX 128, DKS is 8 and only ceil(72/16) == 5 tiles are
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
// pressure goes up. It only matters where DKS_FULL < DKS *and* the config still takes a scalar
// fallback -- head 48 and 96; head 64/128 have DKS_FULL == DKS and are unaffected either way.
#define DKS_FULL ((K_HEAD_SIZE + DPAS_K - 1) / DPAS_K)
#if IS_PA_K_U4
// The u4 depth permutation pairs tiles (2g, 2g+1) over one 32-channel window, so an odd active
// count would leave the last pair half-formed. Round up; DKS is even (D_MAX is a power of two
// >= 32) and DKS_FULL <= DKS, so the rounded value still fits.
#  define DKS_ACTIVE (((DKS_FULL + 1) / 2) * 2)
#else
#  define DKS_ACTIVE DKS_FULL
#endif
#if DKS_ACTIVE > DKS
#  error "sdpa_ocl.cl: DKS_ACTIVE must not exceed DKS (D_MAX is K_HEAD_SIZE rounded up, so it cannot)"
#endif
#if DKS_ACTIVE < 1
#  error "sdpa_ocl.cl: DKS_ACTIVE must cover at least one depth tile"
#endif
#if defined(KV_COMPRESSED) && !(KEY_ZERO_POINTS && VAL_ZERO_POINTS)
// SDPAOclGenerator::supported() admits only asymmetric KV-cache compression with planar scale and
// zero-point tensors -- the only mode kv_cache_compression.cpp produces on the XMX parts this kernel
// runs on -- and every plain-SDPA dequant below assumes a zero point.
#  error "sdpa_ocl.cl: plain-SDPA KV compression must be asymmetric with planar scale/zp tensors"
#endif
// Tiling invariants choose_config() / solve_sv_split() guarantee. A violation does not fail -- it
// silently leaves part of the output uncomputed -- so make any host/kernel drift loud here.
#if (sv_sg_per_wg_values * sv_sg_per_wg_scores) != sg_per_wg
#  error "sdpa_ocl.cl: the S*V split must use exactly the KQ stage's subgroups"
#endif
#if (sv_sg_tile_scores * sv_sg_per_wg_scores) != kq_wg_tile_queries
#  error "sdpa_ocl.cl: the S*V score tiles must cover the workgroup's query tile"
#endif
#if (sv_sg_tile_values * sv_sg_per_wg_values) < V_HEAD_SIZE
#  error "sdpa_ocl.cl: the S*V value tiles must cover the V head dim"
#endif
#if kq_sg_tile_keys != 16 && kq_sg_tile_keys != 32
// k_mask / mask_tile hold kq_sg_tile_keys / SUBGROUP_SIZE <= 2 entries, and one i8 transform read
// covers 32 keys.
#  error "sdpa_ocl.cl: kq_sg_tile_keys must be 16 or 32"
#endif

// Paged attention's MIXED stage: K/V come from the paged cache (plus the raw current tokens).
#define IS_PA_MIXED (IS_PAGED_ATTENTION && !IS_PREFILL)

// Reading the new tokens' K/V from Kc/Vc needs pointers that only exist in the paged-attention
// non-prefill signature, so fold that precondition into the host flag here rather than repeating it at
// every use. The host already scopes SDPA_OCL_PA_CUR_F16 to that variant; this makes
// the kernel independent of that invariant, so a stray define is a no-op instead of a build failure.
#if !IS_PA_MIXED
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

// Bottom-right-aligned causal mask (plain SDPA; paged attention shifts by past_len instead): every
// query position moves by causal_offset = max(0, k - q). Deliberately unparenthesised -- it is only
// used as an operand of a comparison, and expands to exactly the tokens the old #if splices had.
#if !IS_PAGED_ATTENTION && CAUSAL_MASK_LOWER_RIGHT
#  define LOWER_RIGHT_SHIFT(pos) pos + causal_offset
#else
#  define LOWER_RIGHT_SHIFT(pos) pos
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

// Row pitch of a K / V cache page's DATA region, in elements of the cache dtype. That is the head size
// for f16 and i8 -- so those paths preprocess to exactly what they were and the host does not jit
// these at all -- but a u4 page packs two values per byte while its layout dtype is u8, so the pitch
// is NOT derivable from the head size and sizeof() and the host has to supply it:
//   K u4 BY_CHANNEL  exactly K_HEAD_SIZE/2, deliberately NOT aligned up. 16*(h/2) data bytes + 4*h comp
//                    bytes == 12*h is what makes the token-major page a byte-exact fit into the
//                    allocation the upstream d-major INT4 page already has; aligning the pitch up
//                    would overflow that page whenever h % 32 != 0.
//   V u4 BY_TOKEN    Align(V_HEAD_SIZE/2, SUBGROUP_SIZE). Aligning is free here because the trailing
//                    comp slack absorbs it (16*PV + 64 == 16*(PV+4)), and it keeps the pitch a
//                    multiple of 16 for every head size.
// Matches sdpa_ocl_decode.cl's K_ROW_ELEMS / V_ROW_ELEMS and the writer's phys_{k,v}_head_size.
#ifndef PA_K_ROW_ELEMS
#  define PA_K_ROW_ELEMS K_HEAD_SIZE
#endif
#ifndef PA_V_ROW_ELEMS
#  define PA_V_ROW_ELEMS V_HEAD_SIZE
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
// The DATA row pitch is the head size in every case -- the +4 is never inside a data row.
#if IS_PAGED_ATTENTION
#  define PA_K_PAGE_STRIDE (ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE)
// An uncompressed cache has no comp region, so both factors collapse and PA_K_PAGE_STRIDE is
// PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE exactly. Asserted rather than assumed because the f16 block
// read below spells that product out literally (see the comment there).
#  if !IS_PA_KV_COMPRESSED
#    if (ADJUSTED_K_HEAD_SIZE != K_HEAD_SIZE) || (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE != PAGED_ATTENTION_BLOCK_SIZE)
#      error "sdpa_ocl.cl: an uncompressed PA K page must have ADJUSTED_* equal to the plain sizes"
#    endif
#  endif
#endif

// Offset from a K / V page base to the comp region that follows the data rows, in cache-dtype
// elements (which is bytes in every compressed mode). Same place for every quant mode -- only the
// CONTENT differs:
//   BY_TOKEN   two per-token f16 arrays, scale at [token], zp at [PAGED_ATTENTION_BLOCK_SIZE + token]
//   BY_CHANNEL K_HEAD_SIZE interleaved (scale, zp) f16 pairs, so channel c's pair is the DWORD at [c]
// The packing shrinks the DATA region, not the comp region, which is why the multiplier is
// PA_*_ROW_ELEMS rather than K_HEAD_SIZE.
// Matches pa_kv_cache_update_ref.cl's BC_COMP_OFF / quantize_and_save_per_token.
#if IS_PAGED_ATTENTION
#  define PA_K_COMP_OFF ((size_t)PA_K_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#  define PA_V_COMP_OFF ((size_t)PA_V_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#endif

#if IS_PAGED_ATTENTION
// Element offset of the (page, kv_head) K / V cache page. The V form keeps the distributed product
// its call sites always used.
#  define PA_K_PAGE_OFF(page, kvh) (((size_t)(page) * KV_HEADS_NUM + (kvh)) * PA_K_PAGE_STRIDE)
#  define PA_V_PAGE_OFF(page, kvh) ((size_t)(page) * KV_HEADS_NUM * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE + \
                                    (size_t)(kvh) * PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE)
// Rows of the page starting at key0 that hold keys: slots at or past k were never written, and a NaN
// read from one would survive the masked-out score. A block read clamps its surface height to this.
#  define PA_PAGE_ROWS(k, key0) min((int)PAGED_ATTENTION_BLOCK_SIZE, (k) - (key0))
#endif
// Dequant of a PA cache element: the writer stores 1/scale, so the value read back IS the multiplier.
#define PA_DEQ(q, zp, sc) (((q) - (zp)) * (sc))
// High (hi != 0) or low nibble of a u4 byte.
#define U4_NIBBLE_SEL(b, hi) ((hi) ? ((b) >> 4) : ((b) & 0x0Fu))
// Offset of the (depth tile db, query block qb) Q tile in Q_slm.
#define Q_SLM_OFF(db, qb) (((db) * q_blocks + (qb)) * Q_DWORDS * SUBGROUP_SIZE)

// ---------------------------------------------------------------------------------------------
// u4 (INT4) token-major BY_CHANNEL K cache: the DPAS depth axis is PERMUTED.
//
// This kernel's KQ A operand is K itself, so lane == head dim (the DPAS depth index) -- the exact
// mirror of sdpa_ocl_decode.cl, where K is the B operand and lane == token. The K page uses the
// upstream ADJACENT nibble order (byte b holds channel 2b in the low nibble, 2b+1 in the high), which
// the writer is forced into: NUM_HEAD_SIZE_PARTITIONS splits the channel range across WORKGROUPS,
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
#  if ADJUSTED_K_HEAD_SIZE != K_HEAD_SIZE
// BY_CHANNEL's comp is sized by CHANNEL, so it grows the page's row COUNT
// (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE), not its row pitch. A host that added the BY_TOKEN +4 here
// would put every page base 4 * block_size bytes too far apart.
#    error "sdpa_ocl.cl: BY_CHANNEL must leave ADJUSTED_K_HEAD_SIZE == K_HEAD_SIZE"
#  endif
#endif

#if IS_PA_K_U4
#  if !IS_PA_K_BY_CHANNEL
// A u4 PA key cache is always BY_CHANNEL -- execution_config.cpp asserts against 4-bit BY_TOKEN keys
// -- so there is no u4 BY_TOKEN path to write and none is implemented.
#    error "sdpa_ocl.cl: IS_PA_K_U4 requires the token-major BY_CHANNEL page"
#  endif
#  if (K_HEAD_SIZE % 2) != 0
// PA_K_ROW_ELEMS is K_HEAD_SIZE/2 exactly, with no rounding anywhere.
#    error "sdpa_ocl.cl: u4 needs an even K_HEAD_SIZE"
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
// A u4 page's row is K_HEAD_SIZE/2 bytes, so at head 64 it is 32 -- below the 64-byte block2d
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

#define SDPA_OCL_CONFIG_INL 1
