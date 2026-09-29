// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compile-time configuration of sdpa_ocl.cl: derived tiling, dtype and layout macros, and the
// host/kernel invariants as #errors. Included once, by sdpa_ocl.cl only.

// 16-bit element type of Q/K/V/output (f16 or bf16). The DT_* bodies are not parenthesised (they
// replaced per-dtype code token for token), so pass only simple operands.
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

// Runtime scale input: SCALE_DATA_T is the input's own storage type (half, float, or ushort for bf16). A bf16
// value is the upper half of the f32 with the same bits, so it widens with a shift (no bf16 helper needed,
// which the include of bf16_utils.cl in sdpa_ocl.cl only provides for a bf16 Q).
#if SCALE_IS_BF16
#  define SCALE_TO_FLOAT(x) as_float(((uint)(x)) << 16)
#else
#  define SCALE_TO_FLOAT(x) convert_float(x)
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

// Depth tiles that hold data, for the KQ contraction only. DKS covers D_MAX (K_HEAD_SIZE rounded up
// to a power of two), so at head 72 only 5 of its 8 tiles are real; the rest would load nothing and
// add zero. Bounds the KQ depth loop and the Q staging. D_MAX itself must stay: the S*V split and
// the alpha[] nesting are derived from it (see choose_config()).
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

// Kc/Vc exist only in the MIXED signature, so PA_CUR_KV_F16 is forced off everywhere else: a stray
// host define is a no-op rather than a build failure.
#if !IS_PA_MIXED
#  undef PA_CUR_KV_F16
#  define PA_CUR_KV_F16 0
#endif

// Mask-kind predicates: MASK_KIND is the host's compile-time proof of the mask shape (2 = full 2D
// [q>1,k>1], 1 = per-key [q==1,k>1], 0 = scalar/broadcast) and folds the dead branches away; -1
// keeps the runtime MSK_D2/MSK_D3 checks.
#if MASK_KIND == -1
#  define MASK_IS_PER_KEY  (MSK_D2 == 1 && MSK_D3 > 1)
#  define MASK_IS_FULL_2D  (MSK_D2 > 1 && MSK_D3 > 1)
#else
#  define MASK_IS_PER_KEY  (MASK_KIND == 1)
#  define MASK_IS_FULL_2D  (MASK_KIND == 2)
#endif

// Bottom-right-aligned causal mask (plain SDPA only; paged attention shifts by past_len instead):
// every query position moves by causal_offset = max(0, k - q). Unparenthesised: only ever used as a
// comparison operand.
#if !IS_PAGED_ATTENTION && CAUSAL_MASK_LOWER_RIGHT
#  define LOWER_RIGHT_SHIFT(pos) pos + causal_offset
#else
#  define LOWER_RIGHT_SHIFT(pos) pos
#endif

// Bidirectional attention over image-token groups (gemma-4 and friends): a maximal run of
// token_type_ids[t] == 1 is a group whose members attend to each other in both directions, on top
// of the causal + sliding-window region. HAS_TOKEN_TYPE_IDS declares the parameter; USE_BIDIR_MASK
// (SDPA_OCL_BIDIR=0) removes only the logic, so the toggle cannot desync the signature from
// get_arguments_desc(). token_type_ids covers the NEW tokens only, so it is indexed in local
// coordinates (key = query_position_offset + local). GENERATE never reaches this kernel.
#define BIDIR_MASK (HAS_TOKEN_TYPE_IDS && USE_BIDIR_MASK && IS_PAGED_ATTENTION)

// A runtime-empty token_type_ids ("[B_token | 0]", no image tokens) is legal and the impl is
// shape-agnostic, so every read is gated on the runtime count. USE_BIDIR_GATE=0
// (SDPA_OCL_BIDIR_GATE) drops the gate as a negative control. Needs a default because it is used in
// C expressions.
#ifndef USE_BIDIR_GATE
#  define USE_BIDIR_GATE 1
#endif

// Row pitch of a K / V page's DATA region, in cache-dtype elements: the head size for f16 and i8. A
// u4 page packs two values per byte in a u8 layout, so the host supplies it: exactly K_HEAD_SIZE/2
// for K (so the token-major page fits the upstream d-major allocation) and Align(V_HEAD_SIZE/2,
// SUBGROUP_SIZE) for V. Matches sdpa_ocl_decode.cl's K/V_ROW_ELEMS and the writer.
#ifndef PA_K_ROW_ELEMS
#  define PA_K_ROW_ELEMS K_HEAD_SIZE
#endif
#ifndef PA_V_ROW_ELEMS
#  define PA_V_ROW_ELEMS V_HEAD_SIZE
#endif

// In-page addressing of a K cache page, in K elements: d-major pages are [head, token], token-major
// ones [token, head] like V. The scalar-gather fallbacks (used wherever the host's block2d page
// rule fails) address through this pair, so one path serves both layouts.
#if IS_PA_K_TOKEN_MAJOR
#  define PA_K_TOKEN_STRIDE  PA_K_ROW_ELEMS
#  define PA_K_HIDDEN_STRIDE 1
#else
#  define PA_K_TOKEN_STRIDE  1
#  define PA_K_HIDDEN_STRIDE PAGED_ATTENTION_BLOCK_SIZE
#endif

// Distance in K elements between consecutive (block, kv_head) pages. The comp region grows
// whichever factor it is indexed by, so every layout is the same product (table in
// docs/sdpa_ocl.md): BY_TOKEN widens the row (ADJUSTED_K_HEAD_SIZE), BY_CHANNEL adds rows
// (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE). The data row pitch never includes the comp.
#if IS_PAGED_ATTENTION
#  define PA_K_PAGE_STRIDE (ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE)
// Uncompressed: both factors are the plain sizes. Asserted, because the f16 page read spells the
// product out literally.
#  if !IS_PA_KV_COMPRESSED
#    if (ADJUSTED_K_HEAD_SIZE != K_HEAD_SIZE) || (ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE != PAGED_ATTENTION_BLOCK_SIZE)
#      error "sdpa_ocl.cl: an uncompressed PA K page must have ADJUSTED_* equal to the plain sizes"
#    endif
#  endif
#endif

// Offset from a page base to its comp region, in cache-dtype elements (bytes when compressed).
// BY_TOKEN: per-token f16 scale at [token], zp at [PAGED_ATTENTION_BLOCK_SIZE + token]. BY_CHANNEL:
// one interleaved (scale, zp) f16 pair -- one DWORD -- per channel, at [c]. Matches
// pa_kv_cache_update_ref.cl (BC_COMP_OFF / quantize_and_save_per_token).
#if IS_PAGED_ATTENTION
#  define PA_K_COMP_OFF ((size_t)PA_K_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#  define PA_V_COMP_OFF ((size_t)PA_V_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#endif

#if IS_PAGED_ATTENTION
// Element offset of the (page, kv_head) K / V cache page. The V form keeps its call sites' original
// (distributed) product.
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

// Linkage of every helper in the sdpa_ocl_*.cl headers. With a plain `inline` helper IGC still
// inlines it, but optimizes the whole kernel differently (the ISA moves far from the call site);
// with always_inline the ISA is the same as with the code written in place. Helpers also take the
// lane id twice where needed, `size_t lane` for size_t arithmetic and `int lane_i` for int: a
// `(int)lane` inside a helper folds against the kernel's widened lane when it is inlined and moves
// register allocation, so the caller does that conversion. Rationale: docs/sdpa_ocl.md.
#define SDPA_OCL_INLINE __attribute__((always_inline)) inline

// u4 (INT4) token-major BY_CHANNEL K cache: the DPAS depth axis is PERMUTED. The KQ A operand is K
// with lane == head dim, but a K page byte holds channels (2b, 2b+1) -- the writer's adjacent
// nibble order, which keeps a byte inside one writer workgroup -- so a byte-column read hands lane
// L a channel PAIR, never the contiguous (base + L) a DPAS tile wants. Depth is a contraction axis,
// so K and Q both adopt this labelling instead; Q pays for it once, in the SLM staging:
//     PA_K_U4_CHANNEL(db, L) = win + 2L + (db & 1),  win = (db >> 1) * 32,  byte = win/2 + L
// Derivation and per-path consequences: docs/sdpa_ocl.md.
#if IS_PA_K_U4
#  define PA_K_U4_WIN(db)        (((db) >> 1) * (2 * DPAS_K))
#  define PA_K_U4_PAR(db)        ((db) & 1)
#  define PA_K_U4_CHANNEL(db, l) (PA_K_U4_WIN(db) + 2 * (int)(l) + PA_K_U4_PAR(db))
#endif

// u4 V uses the SPLIT convention instead (byte b holds dims b and b + PA_V_ROW_ELEMS): V is the S*V
// B operand, where lane == head dim too, so a read at the folded byte column hands lane c dim base
// + c from either nibble. Both macros are the identity for i8/f16.
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
// The permutation pairs tiles (2g, 2g+1). Always true (D_MAX is a power of two >= 32 and DKS_ACTIVE
// is rounded up to even); checked so a drift in either is loud.
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

// 1D subgroup block read of a whole u4 page, for rows block2d cannot take (head 32/64: 16/32-byte
// rows, under the 64 B minimum). intel_sub_group_block_read_uc16 lands component i of lane L on
// byte SUBGROUP_SIZE * i + L of the page's contiguous data region, so with COLS = ROW /
// SUBGROUP_SIZE
//     SUBGROUP_SIZE * (PA_PAGE_UC16 * r + i) + L == t * ROW + SUBGROUP_SIZE * c + L
//                                           <=>  PA_PAGE_UC16 * r + i == t * COLS + c
// places (token t, 16-byte column group c) with no shuffle. r and i must stay compile-time
// constants (a runtime i turns the uchar16 subscript into indirect addressing), so a caller with a
// runtime c biases the base by SUBGROUP_SIZE * c and passes c = 0. Reads may overhang into the
// page's own comp arrays, never past the allocation. Derivation and bounds: docs/sdpa_ocl.md.
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

// Row geometry the mapping needs, asserted per tensor because the host gates K and V independently
// (at head 48 the u4 V row is 32 bytes and qualifies, the K row is 24 and does not). A drift would
// otherwise read the wrong bytes rather than fail to build.
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
