// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// PagedAttention GENERATE (decode) stage on DPAS + 2D block IO.
//
// Why a separate kernel instead of a mode in sdpa_ocl.cl: decode has exactly ONE query per
// sequence, which flips the DPAS operand roles. sdpa_ocl.cl computes S^T with A = K (M = 8 keys,
// lane = head dim) and B = Q, so its scores land lane = query and its softmax is an SLM tile plus
// an alpha[] rescale. Here A = Q with M = 1, B = K, so S lands one float per lane with lane = key
// and the softmax collapses to two sub_group reduces. Almost nothing is shared.
//
// Operand mapping (cl_intel_subgroup_matrix_multiply_accumulate: for every operand,
// lane == that matrix's COLUMN index, so A is M x K with lane = K(depth), B is K x N with
// lane = N, and C is M x N with lane = N):
//
//   KQ   S[key] = sum_d Q[d] * K[key][d]      M=1, N=key, depth=d
//        A = short  : lane = d      -> Q[t*16 + lane]
//        B = int8   : lane = key    -> K[key = lane][t*16 .. t*16+15]
//        C = float  : lane = key
//   SV   O[d] = sum_key P[key] * V[key][d]    M=1, N=d, depth=key
//        A = short  : lane = key    -> exactly the KQ result layout, no shuffle
//        B = int8   : lane = d, VNNI-packed over 16 keys
//        C = float  : lane = d
//
// The B operands come straight out of the cache pages because both are token-major
// ([PAGED_ATTENTION_BLOCK_SIZE tokens, head_size] row-major):
//   K -> transpose_32b_16r8x1c  (lane = row = key, 8 dwords = 16 consecutive head dims).
//        This is the same builtin, with the same fallback, that sdpa_ocl.cl uses to build its
//        Q B-operand out of a row-major [query, head] tensor.
//   V -> transform_16b_16r16x1c (lane = column = head dim, VNNI over the 16 key rows).
//        Same read sdpa_ocl.cl already uses for the paged V cache.
//
// int8 (BY_TOKEN) cache: the page's data region is the same [tokens, head_size] row-major tile, only
// one byte per element, with a head_size BYTE row pitch and two trailing per-token f16 arrays (scale
// then zp) at head_size * PAGED_ATTENTION_BLOCK_SIZE. ADJUSTED_*_HEAD_SIZE (head_size + 4) is
// therefore the PAGE stride while head_size stays the ROW pitch -- the +4 sits at the end of the
// page, not inside each row.
//
// Both dequants collapse, because scale/zp are per TOKEN and this kernel's scores are per LANE:
//
//   KQ:  S[key] = sc[key] * ( sum_d Q[d]*q_int[key][d]  -  zp[key] * sum_d Q[d] )
//                             \_________ the DPAS _________/
//        lane == key, so sc/zp are plain per-lane scalars: they leave the tile loop entirely, need
//        NO sub_group_broadcast (unlike sdpa_ocl.cl's mixed stage, where lane == head dim forces one
//        broadcast per key), and the B operand is a bare int8->half widen with no per-element
//        arithmetic at all. The correction runs once per (key group, head) in float, so unlike the
//        `zp + 1152.0h` bias trick in sdpa_ocl.cl it costs nothing AND keeps zp exact -- f16 has a
//        1.0 ulp at 1152, and the writer's zp (-min*scale - 128) is not an integer.
//   SV:  O[d] = sum_key (P[key]*sc[key]) * (q_int[key][d] - zp[key])
//        scale folds into the probabilities (already lane == key, so again no broadcast); zp varies
//        along the DPAS depth instead of across lanes, so it is the one value that still needs a
//        broadcast, hoisted to one per 16-key chunk. The softmax denominator stays sum(P), NOT
//        sum(P*sc) -- the V scale belongs to the value, not to the weight.
//
// There is no 8-bit transpose in cl_intel_subgroup_2d_block_io (transpose is 32b only), so the i8 K
// read views the page as a DWORD surface: legal because the row pitch is a multiple of 16 bytes and
// the page base is 64 B-aligned whenever head_size % 4 == 0. One read then delivers 32 bytes = 32
// head dims per lane, i.e. TWO DPAS tiles, so i8 issues half as many K messages per page as f16.
//
// GQA: Q_PER_WG q-heads share one K/V read. All heads of a kv group attend to the SAME K/V pages,
// so a workgroup takes Q_PER_WG of them and issues one DPAS per (tile, key group) with an A operand
// Q_PER_WG rows tall -- the B operand (the K or V tile) is loaded once and reused. This is the whole
// reason M > 1 exists here: decode is bandwidth bound, and at M=1 each kv page would be re-read
// kv_group_size times. pa_gqa_single_token does the same amortization with scalar mads
// (HEADS_PER_WI), but its candidate list is {4,3,2} so it caps at 4 pages-shared; DPAS carries M in
// the instruction's repeat count, so M=8 costs the same MACs per cycle as M=1.
// Q_PER_WG is 1/2/4/8 only -- the DPAS A operand comes in no other lengths (rep count <= 8).
//
// Work split: q_len == 1 leaves the key axis as the only source of parallelism, so this kernel
// reuses paged_attention_opt.cl's SDPA_STAGE_1 (pa_sdpa_finalization_stage) verbatim by writing
// the same per-partition intermediates. That kernel never touches the K/V cache, so nothing in
// paged_attention_opt.cl has to change. The contract is:
//   partition p covers keys [swa_start_token + p*SEQ_LEN_PARTITION_SIZE, ... + SEQ_LEN_PARTITION_SIZE)
//     -- with a sliding window the host drops the fully-masked prefix from the partition COUNT, so
//        partition 0 does not start at token 0. See the swa_start_block derivation below.
//   total_partitions_num == get_num_groups(2)
//   seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1
//        -> exp_sums / max_logits / tmp_out, where exp_sums is the sum against the partition's OWN
//           max and tmp_out is already divided by it
//   otherwise -> write output directly, because the host dispatches the finalization only when
//        num_of_partitions > 1. Both halves of that test matter: a long sequence with a small window
//        has seq_len > 256 and yet ONE partition.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io                : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                          : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                     : enable

#define DPAS_K 16  // intel_sub_group_f16_f16_matrix_mad_k16 fixes the depth at 16

#define KEYS_PER_SG (SEQ_LEN_PARTITION_SIZE / SG_PER_WG)
#define KEY_GROUPS  (KEYS_PER_SG / SUBGROUP_SIZE)
#define K_TILES     (K_HEAD_SIZE / DPAS_K)
#define V_TILES     (V_HEAD_SIZE / SUBGROUP_SIZE)

// Byte offset from a page base to its two trailing per-token f16 arrays: scale at [token], zp at
// [PAGED_ATTENTION_BLOCK_SIZE + token]. Same place kv_cache_update's quantize_and_save_per_token
// writes them and pa_sdpa_opt reads them from.
#define K_COMP_OFF (K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE)
#define V_COMP_OFF (V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE)

// How many DPAS tiles one transposed K read covers. The builtin always hands each lane 8 dwords of
// its own key's row; that is DPAS_K halves (one tile) for f16 but 32 bytes (two tiles) for i8. Only
// the block-read path can pair them -- the per-lane fallback loads one tile's worth by construction.
#if IS_KV_COMPRESSED && USE_2D_BLOCK_IO_K
#    define K_TILES_PER_READ 2
#else
#    define K_TILES_PER_READ 1
#endif
#define K_READS (K_TILES / K_TILES_PER_READ)

// int8 -> f16 widen for the KQ B operand, shared by both K load paths. Rests on the identity
//     as_half(0x6480 ^ b) == b + K_WIDEN_BIAS   exactly, for every signed byte b
// (0x6480 is 1152.0h; XOR-ing 0x80 into its mantissa maps the byte's two's-complement range onto
// consecutive halves). Its value is that it is pure DWORD arithmetic, so 4 source bytes become the 2
// VNNI dwords the DPAS wants with no sub-register data movement. The obvious
// convert_half16(as_char16(...)) needs THREE moves per element instead -- a stride-4 byte gather to
// materialise the char16, then b->w, then w->hf, because Xe2 has no direct b->hf convert -- measured
// at instCount 4431 / 2852 mov against 4045 / 2022 for this form at head 128, M=4.
//
// The bias is undone once per (key group, head) by the score correction, in FLOAT, so it is free AND
// leaves zp exact. Folding 1152 into an f16 zp instead -- what sdpa_ocl.cl's V path does -- would
// quantize it, since f16 has a 1.0 ulp at 1152 and the writer's zp (-min*scale - 128) is not an
// integer. That is also why the V operand keeps a plain widen: its zp must be subtracted per element,
// so it has no float correction to hide the bias in.
#define K_WIDEN_BIAS 1152.0f
#define K_WIDEN_DWORDS(dst, src, tiles)                                                              \
    unroll_for(uint u = 0; u < (tiles); ++u) {                                                       \
        unroll_for(uint j = 0; j < DPAS_K / 4; ++j) {                                                \
            const uint w_ = (src);                                                                   \
            (dst)[j * 2 + 0] = as_int((( w_        & 0x000000FFu) | ((w_ & 0x0000FF00u) << 8)) ^ 0x64806480u); \
            (dst)[j * 2 + 1] = as_int((((w_ >> 16) & 0x000000FFu) | ((w_ >> 8) & 0x00FF0000u)) ^ 0x64806480u); \
        }                                                                                            \
    }

// How many workgroups cover one kv group, and hence the dim-1 head axis the host dispatches.
#define HEAD_GROUPS (KV_HEADS_NUM * HEAD_ITERS)

// Work split for S*V. KQ splits the key axis across all SG_PER_WG subgroups; S*V instead gives
// SV_DIM_SGS subgroups the head-dim axis, because a subgroup that owns a head-dim tile can reduce
// over every key by itself and so produces a FINAL result, not a partial -- no output reduction.
// (Splitting keys in S*V too, as this kernel first did, left every subgroup holding a partial over
// all V_HEAD_SIZE dims and cost an SG_PER_WG * Q_PER_WG * V_HEAD_SIZE SLM round trip: 16 KB at
// head 128, which capped Xe-core occupancy at 7 workgroups instead of 8.)
// When V_TILES < SG_PER_WG there are not enough tiles to keep everyone busy, so the leftover
// subgroups split the keys behind each tile and only THAT many partials need reducing -- 1 for the
// head-128 target, where the reduction compiles away entirely.
#define SV_KEY_SGS        (SG_PER_WG / SV_DIM_SGS)
#define CHUNKS            (SEQ_LEN_PARTITION_SIZE / SUBGROUP_SIZE)
#define CHUNKS_PER_KEY_SG (CHUNKS / SV_KEY_SGS)
#if SV_KEY_SGS > 1
#    define SV_NEEDS_OUTPUT_REDUCTION 1
#endif
#if (KV_HEADS_GROUP_SIZE % Q_PER_WG) != 0
// The last workgroup of a kv group has fewer than Q_PER_WG real heads (e.g. kv group 7 with M 4).
#    define HAS_HEAD_LEFTOVERS 1
#endif

// M-wide operand/accumulator types. MAKE_VECTOR_TYPE(T, 1) is the scalar T, so element access needs
// the same accessor indirection paged_attention_opt.cl:60-67 uses for QUERIES_PER_WI.
#define A_VEC_TYPE MAKE_VECTOR_TYPE(short, Q_PER_WG)
#define H_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, Q_PER_WG)
#define S_VEC_TYPE MAKE_VECTOR_TYPE(SOFTMAX_ACCUMULATOR_TYPE, Q_PER_WG)

#define _QV_1(vec, idx) vec
#define _QV_2(vec, idx) vec[idx]
#define _QV_4(vec, idx) vec[idx]
#define _QV_8(vec, idx) vec[idx]
#define QV(vec, idx) CAT(_QV_, Q_PER_WG)(vec, idx)

// 2D block prefetch, PREFETCH_DIST iterations ahead. The point is memory-level parallelism that
// occupancy cannot buy: a normal block read needs a destination register, so at 128 GRF only two or
// three of the sixteen K (or V) tiles can be in flight at once, whereas a prefetch has no
// destination and just warms the cache. The measured elasticities say this is where the time goes --
// doubling K/V traffic costs 23% while cutting instructions 6.8% and SLM 86% bought only 1.2%, and
// every occupancy knob (SG_PER_WG 2/4/16, 256 GRF) was neutral or worse.
// sdpa_ocl.cl uses the same builtin for its prefill V blocks but explicitly excludes PA decode
// (`!(IS_PAGED_ATTENTION && !IS_PREFILL)`), so this path has never been exercised here.
// Gated per side because the two turned out to be opposite bets. Each prefetch costs ~8 instructions
// of a64 address and descriptor setup (16 K + 16 V prefetches took instCount 1202 -> 1469, mostly
// mov), and on llama-3.1-8b head 128 / M=4 that bought:
//   K only : 3.6% SLOWER  -- the KQ loop already runs KEY_GROUPS independent DPAS chains over a
//                            single 4 KB page, so its loads were pipelined already. Default OFF.
//   V only : 2.1% FASTER  -- S*V walks 16 different pages with one accumulator chain, so its loads
//                            were the ones exposing latency. Default ON.
//   both   : 0.8% slower  -- K's loss swamps V's win.
// Kept as toggles rather than deleting the K path, so the negative result stays recorded.
// PREFETCH_DIST 0 disables both.
#define USE_PREFETCH_K (PREFETCH_DIST > 0 && PREFETCH_K)
#define USE_PREFETCH_V (PREFETCH_DIST > 0 && PREFETCH_V)

// Indexed by READ, not by tile: x is 8 dwords per read for both precisions (see the K read below),
// so the descriptor arithmetic is dtype-independent and sizeof(INPUT1_TYPE) covers the rest.
#define PREFETCH_K_TILE(page_off, read)                                                   \
    intel_sub_group_2d_block_prefetch_32b_16r8x1c((__global void*)(key_cache + (page_off)), \
                                                  K_HEAD_SIZE * (int)sizeof(INPUT1_TYPE),   \
                                                  PAGED_ATTENTION_BLOCK_SIZE,               \
                                                  K_HEAD_SIZE * (int)sizeof(INPUT1_TYPE),   \
                                                  (int2)((read) * 8, 0))

#if IS_KV_COMPRESSED
// Matches the 8-bit VNNI-transform read below. x is in elements (bytes here), and a multiple of
// SUBGROUP_SIZE satisfies the spec's "multiple of four for 8-bit data" rule on coord.x.
#    define PREFETCH_V_TILE(page_off, value_tile)                                                  \
        intel_sub_group_2d_block_prefetch_8b_32r16x1c((__global void*)(value_cache + (page_off)),   \
                                                      V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),      \
                                                      PAGED_ATTENTION_BLOCK_SIZE,                  \
                                                      V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),      \
                                                      (int2)((value_tile) * SUBGROUP_SIZE, 0))
#else
#    define PREFETCH_V_TILE(page_off, value_tile)                                                  \
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((__global void*)(value_cache + (page_off)), \
                                                       V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),     \
                                                       PAGED_ATTENTION_BLOCK_SIZE,                 \
                                                       V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),     \
                                                       (int2)((value_tile) * SUBGROUP_SIZE, 0))
#endif

#if Q_PER_WG == 1
#    define AS_A(x) as_short(x)
#elif Q_PER_WG == 2
#    define AS_A(x) as_short2(x)
#elif Q_PER_WG == 4
#    define AS_A(x) as_short4(x)
#elif Q_PER_WG == 8
#    define AS_A(x) as_short8(x)
#endif

#if SUBGROUP_SIZE != 16
#    error "sdpa_ocl_decode.cl: the DPAS N dimension is the subgroup size, which must be 16"
#endif
#if Q_PER_WG != 1 && Q_PER_WG != 2 && Q_PER_WG != 4 && Q_PER_WG != 8
// M is the DPAS repeat count, which the ISA encodes only as 1/2/4/8.
#    error "sdpa_ocl_decode.cl: Q_PER_WG must be 1, 2, 4 or 8"
#endif
#if Q_PER_WG > KV_HEADS_GROUP_SIZE
// Heads beyond the kv group belong to a different K/V page, so they cannot share this read.
#    error "sdpa_ocl_decode.cl: Q_PER_WG must not exceed KV_HEADS_GROUP_SIZE"
#endif
#if HEAD_ITERS != ((KV_HEADS_GROUP_SIZE + Q_PER_WG - 1) / Q_PER_WG)
// Host and kernel must agree, or the dim-1 decode below lands on the wrong head.
#    error "sdpa_ocl_decode.cl: HEAD_ITERS must be ceil_div(KV_HEADS_GROUP_SIZE, Q_PER_WG)"
#endif
#if (SG_PER_WG % SV_DIM_SGS) != 0
// Every head-dim subgroup must have the same number of key subgroups behind it.
#    error "sdpa_ocl_decode.cl: SV_DIM_SGS must divide SG_PER_WG"
#endif
#if (CHUNKS % SV_KEY_SGS) != 0
#    error "sdpa_ocl_decode.cl: SV_KEY_SGS must divide the partition's chunk count"
#endif
#if PAGED_ATTENTION_BLOCK_SIZE != SUBGROUP_SIZE
// S*V walks the partition one chunk at a time and treats each chunk as exactly one cache page.
#    error "sdpa_ocl_decode.cl: an S*V chunk must be one page"
#endif
#if CHUNKS > SUBGROUP_SIZE
// The partition's page table is held one chunk per lane so it can be read in a single message.
#    error "sdpa_ocl_decode.cl: the partition must not span more chunks than a subgroup has lanes"
#endif
#if PAGED_ATTENTION_BLOCK_SIZE != SUBGROUP_SIZE
// A key group is one cache page: that is what lets the page index be hoisted out of the d loop
// and lets a 16-row block read cover exactly the group.
#    error "sdpa_ocl_decode.cl: PAGED_ATTENTION_BLOCK_SIZE must equal SUBGROUP_SIZE"
#endif
#if SEQ_LEN_PARTITION_SIZE % (SG_PER_WG * SUBGROUP_SIZE) != 0
#    error "sdpa_ocl_decode.cl: the partition must split evenly into whole key groups per subgroup"
#endif
#if K_HEAD_SIZE % DPAS_K != 0
#    error "sdpa_ocl_decode.cl: K_HEAD_SIZE must be a multiple of the DPAS depth"
#endif
#if (K_TILES % K_TILES_PER_READ) != 0
// The i8 transposed read covers two tiles, so the head dim must divide into whole pairs. Implied by
// the block2d pitch rule the host gates on (K_HEAD_SIZE % 64 == 0 for a 1-byte cache), but a drift
// between host and kernel must be loud rather than silently dropping the last tile.
#    error "sdpa_ocl_decode.cl: K_TILES must be a multiple of K_TILES_PER_READ"
#endif
#if V_HEAD_SIZE % SUBGROUP_SIZE != 0
#    error "sdpa_ocl_decode.cl: V_HEAD_SIZE must be a multiple of the DPAS N dimension"
#endif
#if SG_PER_WG > SUBGROUP_SIZE
// The cross-subgroup combine reduces one value per subgroup across the lanes of one subgroup.
#    error "sdpa_ocl_decode.cl: SG_PER_WG must not exceed SUBGROUP_SIZE"
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, SG_PER_WG, 1)))
KERNEL(sdpa_ocl_decode)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* block_indices,
    const __global INPUT5_TYPE* block_indices_begins,
#if HAS_SCALE_INPUT
    const __global SCALE_INPUT_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out) {
    // Inputs / outputs (same shapes pa_sdpa_opt documents):
    //   query:       [sequences_num, HEADS_NUM * K_HEAD_SIZE]  (+ optional feature padding)
    //   key_cache:   [num_blocks, KV_HEADS_NUM, PAGED_ATTENTION_BLOCK_SIZE, K_HEAD_SIZE]
    //   value_cache: [num_blocks, KV_HEADS_NUM, PAGED_ATTENTION_BLOCK_SIZE, V_HEAD_SIZE]
    //   output:      [sequences_num, HEADS_NUM * V_HEAD_SIZE]
    //   exp_sums / max_logits: [sequences_num, HEADS_NUM, total_partitions_num]
    //   tmp_out:     [sequences_num, HEADS_NUM, total_partitions_num, V_HEAD_SIZE]
    const uint lane = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();

    // Dim 1 carries (sequence, head group) because dim 0 is the subgroup lanes and dim 2 has to be
    // the partition, so that get_num_groups(2) is the total_partitions_num stage 1 is told about.
    // A head group is Q_PER_WG consecutive q-heads of one kv head, so kv_head_idx now falls out of
    // the group id rather than being divided out of a q-head index.
    const uint head_group_idx = get_group_id(1) % HEAD_GROUPS;
    const uint seq_idx = get_group_id(1) / HEAD_GROUPS;
    const uint partition_idx = get_group_id(2);
    const uint total_partitions_num = get_num_groups(2);

    const uint seq_len = past_lens[seq_idx] + 1;
    const uint total_blocks_num = CEIL_DIV(seq_len, PAGED_ATTENTION_BLOCK_SIZE);

    // ---- Sliding-window block skip. With a window the host does NOT dispatch a partition per 256
    // keys of the sequence: it drops the whole fully-masked prefix first, so
    // num_of_partitions = ceil(effective_blocks * 16 / 256) and partition 0 begins at the first block
    // the window can reach, not at token 0 (paged_attention_opt.cpp, effective_context_len). Getting
    // this wrong is silent: every key a partition covers is then masked, so the kernel emits a
    // perfectly well-formed all-zero result. Must match pa_sdpa_opt's swa_start_block /
    // swa_start_token / effective_blocks_num exactly -- both kernels are dispatched from that same
    // partition count and feed the same finalization.
    // swa_start_block rounds DOWN to a block boundary, so the first block still contains masked
    // tokens and the per-key mask below is still required.
#if SLIDING_WINDOW_SIZE != 0
    const uint swa_start_block =
        (seq_len > SLIDING_WINDOW_SIZE) ? ((seq_len - SLIDING_WINDOW_SIZE) / PAGED_ATTENTION_BLOCK_SIZE) : 0;
    const uint effective_blocks_num = total_blocks_num - swa_start_block;
#else
    const uint swa_start_block = 0;
    const uint effective_blocks_num = total_blocks_num;
#endif
    const uint swa_start_token = swa_start_block * PAGED_ATTENTION_BLOCK_SIZE;

    // Workgroup-uniform, so the whole workgroup leaves together and the barrier below is safe.
    if (partition_idx * SEQ_LEN_PARTITION_SIZE >= effective_blocks_num * PAGED_ATTENTION_BLOCK_SIZE) {
        return;
    }

    const uint kv_head_idx = head_group_idx / HEAD_ITERS;
    const uint head_iter = head_group_idx % HEAD_ITERS;
    const uint head_base = kv_head_idx * KV_HEADS_GROUP_SIZE + head_iter * Q_PER_WG;
#ifdef HAS_HEAD_LEFTOVERS
    // head_iter * Q_PER_WG < KV_HEADS_GROUP_SIZE by construction, so this cannot underflow.
    const uint heads_this_wg = min((uint)(KV_HEADS_GROUP_SIZE - head_iter * Q_PER_WG), (uint)Q_PER_WG);
#endif

    const uint base_block_index = block_indices_begins[seq_idx];

    // ---- Q: Q_PER_WG heads x K_HEAD_SIZE values, held in registers with lane == head dim. Kept in
    // registers rather than SLM -- pa_sdpa_opt is forced into slm_query whenever HEADS_PER_WI > 1
    // (paged_attention_opt.cl:18-20) and then re-reads it per (qk_idx, q_idx).
    // The scale is folded in here, matching pa_sdpa_opt (better accuracy than scaling the scores,
    // and free since Q is read once).
    INPUT0_TYPE q_reg[Q_PER_WG][K_TILES];
    {
        const uint q_row = INPUT0_OFFSET +
                           seq_idx * (K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM);
#ifdef SCALE_VAL
        const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(SCALE_VAL);
#else
        const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(*scale);
#endif
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#ifdef HAS_HEAD_LEFTOVERS
            // A leftover slot would read the next kv group's query, or past the tensor at the last
            // kv head. Re-read head_base instead -- always valid, and its result is never stored.
            const uint head = head_base + (m < heads_this_wg ? m : 0);
#else
            const uint head = head_base + m;
#endif
            const uint q_base = q_row + head * K_HEAD_SIZE;
            unroll_for(uint t = 0; t < K_TILES; ++t) {
                q_reg[m][t] = BLOCK_READN(INPUT0_TYPE, 1, query, q_base + t * SUBGROUP_SIZE) * scale_val;
            }
        }
    }

#if IS_KV_COMPRESSED
    // sum_d Q[d], the only thing the K zero point needs: the whole per-key zp contribution to a
    // score is zp[key] * sum_d Q[d] (see the identity at the top), so one reduce per head here
    // replaces every per-element subtraction in the KQ loop. q_reg holds lane == head dim, so the
    // reduce is over lanes. Kept in float: it multiplies an f32 accumulator, and f16 would cap the
    // correction's precision for no saving.
    SOFTMAX_ACCUMULATOR_TYPE q_sum[Q_PER_WG];
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE acc = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        unroll_for(uint t = 0; t < K_TILES; ++t) {
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(q_reg[m][t]);
        }
        q_sum[m] = sub_group_reduce_add(acc);
    }
#endif

    // ---- This subgroup's slice of the partition: KEY_GROUPS whole cache pages. Absolute token
    // indices, so they start at the window's first block rather than at the sequence start.
    const uint sg_key0 = swa_start_token + partition_idx * SEQ_LEN_PARTITION_SIZE + sgid * KEYS_PER_SG;

    // The partition spans exactly CHUNKS pages, and one chunk is one page, so the whole page table
    // for this partition fits in a single lane-per-chunk register: ONE coalesced read, then
    // sub_group_broadcast wherever a page is needed. K needs 2 of them (this subgroup's own key
    // groups) and S*V needs all of them, and doing it per use cost 18 separate scalar loads.
    // A chunk past the subsequence's last block would index outside block_indices[], so clamp to
    // page 0, which is always allocated; the scores of such a chunk are masked below.
    // Block index within the SEQUENCE, so the window's skipped prefix has to be added back.
    const uint my_block = swa_start_block + partition_idx * CHUNKS + lane;
    const uint my_page =
        (lane < CHUNKS && my_block < total_blocks_num) ? (uint)block_indices[base_block_index + my_block] : 0u;

    // Whether a key group is in range is pure arithmetic, so it needs no broadcast.
    // The page stride is ADJUSTED_K_HEAD_SIZE, which IS K_HEAD_SIZE for an uncompressed cache (so f16
    // is unchanged) and K_HEAD_SIZE + 4 for an i8 one, the extra 4 being the trailing scale/zp arrays.
    size_t k_page_off[KEY_GROUPS];
    bool group_in_range[KEY_GROUPS];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const uint chunk = sgid * KEY_GROUPS + g;  // == (sg_key0 + g*SUBGROUP_SIZE)/BLOCK_SIZE - partition*CHUNKS
        group_in_range[g] = (swa_start_block + partition_idx * CHUNKS + chunk) < total_blocks_num;
        const size_t page = (size_t)sub_group_broadcast(my_page, chunk);
        k_page_off[g] = (page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
    }

#if IS_KV_COMPRESSED
    // Per-key scale and zero point, one coalesced 16-lane f16 load each per page. lane == the key's
    // token within its page (a key group IS a page), so these are exactly the per-lane scalars the
    // correction after the KQ loop wants -- nothing here is broadcast, and nothing is re-read inside
    // the tile loop. The writer stores 1/scale, so what comes back is already the multiplier.
    //
    // No validity guard, unlike the V side below: a token at/past seq_len was never written by
    // kv_cache_update, so its comp bytes can decode to NaN, but the mask further down OVERWRITES
    // s[g] with SOFTMAX_ACCUMULATOR_VAL_MIN on exactly those lanes (token_idx >= seq_len ||
    // !group_in_range[g]), which no NaN survives. V needs the guard because there the masking is
    // "the probability is 0", i.e. a multiply, and 0 * NaN is NaN.
    INPUT0_TYPE k_sc[KEY_GROUPS];
    INPUT0_TYPE k_zp[KEY_GROUPS];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(key_cache + k_page_off[g] + K_COMP_OFF);
        k_sc[g] = comp[lane];
        k_zp[g] = comp[PAGED_ATTENTION_BLOCK_SIZE + lane];
    }
#endif

#if USE_PREFETCH_K && USE_2D_BLOCK_IO_K
    // Prime the pipeline before the Q read below, so the Q load's own latency is spent usefully.
    unroll_for(uint r = 0; r < (PREFETCH_DIST < K_READS ? PREFETCH_DIST : K_READS); ++r) {
        unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
            PREFETCH_K_TILE(k_page_off[g], r);
        }
    }
#endif

    // ---- KQ. The K_TILES accumulations into one s[g] are serially dependent, so the g loop is
    // innermost: KEY_GROUPS independent DPAS chains interleave and hide each other's latency.
    // One DPAS per (tile, g) regardless of Q_PER_WG -- M rides in the repeat count, and the loaded K
    // tile `kb` is the shared B operand for all Q_PER_WG heads. That sharing is the amortization.
    S_VEC_TYPE s[KEY_GROUPS];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        s[g] = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
    }

    unroll_for(uint r = 0; r < K_READS; ++r) {
#if USE_PREFETCH_K && USE_2D_BLOCK_IO_K
        // Compile-time bound, so the guard costs nothing and the tail simply stops prefetching.
        if (r + PREFETCH_DIST < K_READS) {
            unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
                PREFETCH_K_TILE(k_page_off[g], r + PREFETCH_DIST);
            }
        }
#endif
        // One A operand per tile the read covers, built outside the g loop so the Q gather is not
        // repeated per key group. K_TILES_PER_READ is 1 everywhere except the i8 block-read path.
        A_VEC_TYPE a[K_TILES_PER_READ];
        unroll_for(uint u = 0; u < K_TILES_PER_READ; ++u) {
            H_VEC_TYPE qv;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(qv, m) = q_reg[m][r * K_TILES_PER_READ + u];
            }
            a[u] = AS_A(qv);
        }
        unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
            int8 kb[K_TILES_PER_READ];
#if USE_2D_BLOCK_IO_K
            // Surface = the page: [PAGED_ATTENTION_BLOCK_SIZE keys, K_HEAD_SIZE] row-major, so the
            // pitch is K_HEAD_SIZE (NOT the whole-cache stride) in BYTES. x is in dwords for a 32b
            // read, and one read is 8 of them for both precisions -- DPAS_K halves for f16,
            // K_TILES_PER_READ * DPAS_K bytes for i8 -- which is why this call is dtype-independent
            // and only the unpack below branches.
            uint8 kt;
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)(key_cache + k_page_off[g]),
                                                               K_HEAD_SIZE * (int)sizeof(INPUT1_TYPE),
                                                               PAGED_ATTENTION_BLOCK_SIZE,
                                                               K_HEAD_SIZE * (int)sizeof(INPUT1_TYPE),
                                                               (int2)(r * 8, 0),
                                                               (private uint*)&kt);
    #if IS_KV_COMPRESSED
            // 32 signed bytes in ADDRESS order, so byte i is head dim (r*32 + i) and the operand is
            // just those bytes widened to halves: two per dword, ascending depth, no shuffle.
            K_WIDEN_DWORDS(kb[u], kt[u * (DPAS_K / 4) + j], K_TILES_PER_READ);
    #else
            kb[0] = as_int8(kt);
    #endif
#else
            // Same operand, one per-lane load: lane owns key `lane` of the page and the DPAS_K head
            // dims it needs are contiguous there. Always inside the page (lane < block size), so no
            // bounds check is needed.
    #if IS_KV_COMPRESSED
            // DPAS_K bytes == 4 dwords, so the same dword widen applies; only the source differs.
            const uint4 kw = as_uint4(vload16(0, key_cache + k_page_off[g] + lane * K_HEAD_SIZE + r * DPAS_K));
            K_WIDEN_DWORDS(kb[u], kw[j], 1);
    #else
            const ushort16 kv = vload16(0, (const __global ushort*)(key_cache + k_page_off[g] + lane * K_HEAD_SIZE + r * DPAS_K));
            kb[0] = as_int8(as_short16(kv));
    #endif
#endif
            unroll_for(uint u = 0; u < K_TILES_PER_READ; ++u) {
                s[g] = intel_sub_group_f16_f16_matrix_mad_k16(a[u], kb[u], s[g]);
            }
        }
    }

#if IS_KV_COMPRESSED
    // Undo the quantization on the SCORE instead of on every K element: the dequant is affine in the
    // key, so sc[key] and zp[key] * sum_d Q[d] factor straight out of the dot product. Per (key
    // group, head) that is one fma and one multiply, against the K_TILES * DPAS_K per-element
    // subtract-and-scale it replaces. Both factors are per-lane (lane == key), so no broadcast.
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const SOFTMAX_ACCUMULATOR_TYPE sc = TO_SOFTMAX_ACCUMULATOR_TYPE(k_sc[g]);
        // K_WIDEN_BIAS rides along in zp: the B operand held (q_int + bias), so subtracting
        // (zp + bias) * q_sum removes both in one fma. In float, so zp is not rounded.
        const SOFTMAX_ACCUMULATOR_TYPE zp = TO_SOFTMAX_ACCUMULATOR_TYPE(k_zp[g]) + K_WIDEN_BIAS;
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            QV(s[g], m) = sc * (QV(s[g], m) - zp * q_sum[m]);
        }
    }
#endif

    // ---- Mask. SOFTMAX_ACCUMULATOR_VAL_MIN rather than -INFINITY, for the same reason
    // pa_sdpa_opt uses it: a partition can be masked out entirely (sliding window well behind the
    // partition), and MIN - MIN == 0 keeps exp() finite there while exp(MIN - finite) still
    // underflows to 0 everywhere else. Stage 1 then zeroes such a partition through
    // exp(max_logit - global_max).
    // The mask depends only on the key, so it applies to all Q_PER_WG rows identically.
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const uint token_idx = sg_key0 + g * SUBGROUP_SIZE + lane;
        bool masked = !group_in_range[g] || token_idx >= seq_len;
#if SLIDING_WINDOW_SIZE != 0
        masked = masked || (seq_len > SLIDING_WINDOW_SIZE && token_idx < (seq_len - SLIDING_WINDOW_SIZE));
#endif
        if (masked) {
            s[g] = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_MIN);
        }
    }

    // ---- Softmax over this subgroup's keys. Scores are one per lane with lane == key, so the
    // reductions are plain subgroup reduces: no SLM staging and no barrier, unlike pa_sdpa_opt's
    // slm_qk_vals[SEQ_LEN_PARTITION_SIZE].
    // Every reduction is per-head: the M rows are independent softmaxes over the same key set.
    S_VEC_TYPE m_sg = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_MIN);
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            QV(m_sg, m) = SOFTMAX_ACCUMULATOR_MAX_FUNC(QV(m_sg, m), QV(s[g], m));
        }
    }
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        QV(m_sg, m) = sub_group_reduce_max(QV(m_sg, m));
    }

    // S*V reads keys this subgroup did not score, so the probabilities go through SLM. Staging them
    // rather than the outputs is the whole point of the head-dim split: Q_PER_WG * partition halves
    // (2 KB at M=4) instead of SG_PER_WG * Q_PER_WG * V_HEAD_SIZE floats (16 KB).
    // Indexed by key, with the Q_PER_WG heads as the vector element -- i.e. head is the INNERMOST
    // axis. That is what makes each access one wide SLM message instead of Q_PER_WG narrow ones:
    // S*V wants lane == key, so lane L reads slm_p[chunk*16 + L], and across the subgroup those are
    // Q_PER_WG * 16 contiguous halves. Storing head-major instead cost 64 separate 32-byte reads per
    // subgroup (measured: SLM loads 44 -> 72, instCount +12%). Declaring it as the vector type also
    // gets the alignment for free, which a cast on a half array would not.
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max[SG_PER_WG * Q_PER_WG];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_sum[SG_PER_WG * Q_PER_WG];
    __local H_VEC_TYPE slm_p[SEQ_LEN_PARTITION_SIZE];

    // Exchange the max FIRST, so every subgroup can normalise against the same one. The alternative
    // -- normalise locally and rescale each chunk in S*V -- would need a per-chunk multiply on the
    // DPAS A operand, and the scores are still in registers here, so exp() costs the same either way.
    if (lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_max[sgid * Q_PER_WG + m] = QV(m_sg, m);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    S_VEC_TYPE m_wg;
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE my_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (lane < SG_PER_WG) {
            my_max = slm_max[lane * Q_PER_WG + m];
        }
        QV(m_wg, m) = sub_group_reduce_max(my_max);
    }

    // Probabilities against the partition max, so S*V needs no rescale at all. A fully masked key
    // gives exp(MIN - m_wg) = 0 unless the WHOLE partition is masked, in which case exp(MIN - MIN)
    // = 1 everywhere and stage 1 discards the partition through max_logits -- same as before.
    S_VEC_TYPE l_sg = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const S_VEC_TYPE e = native_exp(s[g] - m_wg);
        l_sg += e;
        // f16 probabilities, same as the sdpa_ocl/sdpa_micro S*V operand
        H_VEC_TYPE pv;
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            QV(pv, m) = TO_INPUT0_TYPE(QV(e, m));
        }
        slm_p[sgid * KEYS_PER_SG + g * SUBGROUP_SIZE + lane] = pv;
    }
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        QV(l_sg, m) = sub_group_reduce_add(QV(l_sg, m));
    }
    if (lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_sum[sgid * Q_PER_WG + m] = QV(l_sg, m);
        }
    }

    // Which head-dim tiles and which chunks this subgroup will take in S*V. Hoisted above the
    // barrier so the wait can be spent prefetching the first V tiles instead of idling -- the same
    // arrive/prefetch/wait split sdpa_ocl.cl:817-828 uses for its prefill V blocks.
    const uint dim_slot = sgid % SV_DIM_SGS;
    const uint key_slot = sgid / SV_DIM_SGS;

#if USE_PREFETCH_V && USE_2D_BLOCK_IO_V && PREFETCH_AT_BARRIER
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    unroll_for(uint i = 0; i < (PREFETCH_DIST < CHUNKS_PER_KEY_SG ? PREFETCH_DIST : CHUNKS_PER_KEY_SG); ++i) {
        const size_t pf_page = (size_t)sub_group_broadcast(my_page, key_slot * CHUNKS_PER_KEY_SG + i);
        PREFETCH_V_TILE((pf_page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE, dim_slot);
    }
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#else
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    S_VEC_TYPE l_wg;
    S_VEC_TYPE inv_l;
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE my_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        if (lane < SG_PER_WG) {
            my_sum = slm_sum[lane * Q_PER_WG + m];
        }
        const SOFTMAX_ACCUMULATOR_TYPE lw = sub_group_reduce_add(my_sum);
        QV(l_wg, m) = lw;
        QV(inv_l, m) = SOFTMAX_ACCUMULATOR_VAL_ONE / lw;
    }

    // ---- S*V, split by head dim rather than by key. This subgroup owns head-dim tiles
    // {dim_slot, dim_slot + SV_DIM_SGS, ...} and reduces them over CHUNKS_PER_KEY_SG chunks of the
    // partition. With SV_KEY_SGS == 1 (V_TILES >= SG_PER_WG, e.g. head 128 with 8 subgroups) that is
    // every chunk, so each tile is FINAL when the loop ends and can be written straight out -- no
    // output staging, no reduction, no second barrier.
    // The DPAS and V-load counts are unchanged from the key-split version: it was
    // KEY_GROUPS x V_TILES per subgroup, this is CHUNKS_PER_KEY_SG x (V_TILES / SV_DIM_SGS), and both
    // come to 16 at head 128. Only the SLM traffic differs.
#ifdef SV_NEEDS_OUTPUT_REDUCTION
    __local SOFTMAX_ACCUMULATOR_TYPE slm_out[SV_KEY_SGS * Q_PER_WG * V_HEAD_SIZE];
#endif

    for (uint cd = dim_slot; cd < V_TILES; cd += SV_DIM_SGS) {
        S_VEC_TYPE acc = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
        const uint value = cd * SUBGROUP_SIZE + lane;

        unroll_for(uint i = 0; i < CHUNKS_PER_KEY_SG; ++i) {
            const uint chunk = key_slot * CHUNKS_PER_KEY_SG + i;
#if USE_PREFETCH_V && USE_2D_BLOCK_IO_V
            if (i + PREFETCH_DIST < CHUNKS_PER_KEY_SG) {
                const size_t pf_page = (size_t)sub_group_broadcast(my_page, chunk + PREFETCH_DIST);
                PREFETCH_V_TILE((pf_page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE, cd);
            }
#endif
            // One chunk is one page, already clamped to page 0 when out of range. Such a chunk's
            // probabilities are zero (or the whole partition is masked and stage 1 drops it), so the
            // values it contributes never reach the output.
            const size_t page = (size_t)sub_group_broadcast(my_page, chunk);
            const size_t v_page_off = (page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;

            // A operand: lane == key within the chunk, which is how slm_p was written.
            H_VEC_TYPE pv = slm_p[chunk * SUBGROUP_SIZE + lane];
#if IS_KV_COMPRESSED
            // Per-key scale and zp of this chunk's page; lane == the key's token, so one coalesced
            // f16 load each. The scale is folded into the PROBABILITIES rather than into V: pv is
            // already lane == key, so that is a per-lane multiply with no broadcast, whereas scaling
            // V would need the value broadcast across the head-dim lanes. It does NOT touch the
            // softmax denominator -- inv_l divides by sum(P), and the V scale belongs to the value.
            // A key at/past seq_len was never written, so its comp bytes are arbitrary and could be
            // NaN: force scale AND zp to 0 there. Then the widened byte stays finite and the
            // probability (already 0 for a masked key) zeroes the contribution. Unlike the K side
            // there is no later overwrite to hide a NaN behind.
            const uint v_key = swa_start_token + partition_idx * SEQ_LEN_PARTITION_SIZE + chunk * SUBGROUP_SIZE + lane;
            const __global INPUT0_TYPE* v_comp =
                (const __global INPUT0_TYPE*)(value_cache + v_page_off + V_COMP_OFF);
            // v_key < seq_len also implies the chunk is in range, so one test covers both.
            const bool v_valid = v_key < seq_len;
            const INPUT0_TYPE v_sc = v_valid ? v_comp[lane] : (INPUT0_TYPE)0;
            const INPUT0_TYPE v_zp = v_valid ? v_comp[PAGED_ATTENTION_BLOCK_SIZE + lane] : (INPUT0_TYPE)0;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(pv, m) *= v_sc;
            }
            // zp varies along the DPAS depth (the key), not across the lanes, so it is the one factor
            // that still needs broadcasting. Every lane index here is a full-unroll constant, so each
            // broadcast folds into the subtract's source region instead of emitting a shuffle.
            MAKE_VECTOR_TYPE(INPUT0_TYPE, DPAS_K) vzp;
            unroll_for(uint kk = 0; kk < DPAS_K; ++kk) {
                vzp[kk] = sub_group_broadcast(v_zp, kk);
            }
#endif

            int8 vb;
#if USE_2D_BLOCK_IO_V
    #if IS_KV_COMPRESSED
            // 8-bit VNNI transform: lane == head dim, each uint packing 4 consecutive keys as bytes.
            // On Xe2 the builtin exists only in 32-row form (there is no _8b_16r variant), while a
            // page holds PAGED_ATTENTION_BLOCK_SIZE tokens -- so the height stops it at the page and
            // rows past it read as 0, leaving uints 0..3 as the 16 real keys. as_char16 of those puts
            // the keys in ascending order, which IS the VNNI order the operand wants: after the widen
            // and the zp subtract, dword i already holds keys (2i, 2i+1).
            uint8 vt;
            intel_sub_group_2d_block_read_transform_8b_32r16x1c((__global void*)(value_cache + v_page_off),
                                                               V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),
                                                               PAGED_ATTENTION_BLOCK_SIZE,
                                                               V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),
                                                               (int2)(cd * SUBGROUP_SIZE, 0),
                                                               (private uint*)&vt);
            vb = as_int8(convert_half16(as_char16(vt.lo)) - vzp);
    #else
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void*)(value_cache + v_page_off),
                                                                V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),
                                                                PAGED_ATTENTION_BLOCK_SIZE,
                                                                V_HEAD_SIZE * (int)sizeof(INPUT2_TYPE),
                                                                (int2)(cd * SUBGROUP_SIZE, 0),
                                                                (private uint*)&vb);
    #endif
#else
            // Hand-built VNNI operand: dword `key_pair` packs keys (2*key_pair, 2*key_pair+1) of
            // this lane's head dim. The row pitch is V_HEAD_SIZE for both precisions -- the page's
            // trailing scale/zp arrays sit after the data, not inside each row.
            unroll_for(uint key_pair = 0; key_pair < DPAS_K / 2; ++key_pair) {
                const INPUT2_TYPE v0 = value_cache[v_page_off + (size_t)(2 * key_pair) * V_HEAD_SIZE + value];
                const INPUT2_TYPE v1 = value_cache[v_page_off + (size_t)(2 * key_pair + 1) * V_HEAD_SIZE + value];
    #if IS_KV_COMPRESSED
                vb[key_pair] = as_int((MAKE_VECTOR_TYPE(INPUT0_TYPE, 2))((INPUT0_TYPE)v0 - vzp[2 * key_pair],
                                                                        (INPUT0_TYPE)v1 - vzp[2 * key_pair + 1]));
    #else
                vb[key_pair] = as_int((MAKE_VECTOR_TYPE(INPUT0_TYPE, 2))(v0, v1));
    #endif
            }
#endif
            acc = intel_sub_group_f16_f16_matrix_mad_k16(AS_A(pv), vb, acc);
        }

#ifdef SV_NEEDS_OUTPUT_REDUCTION
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_out[(key_slot * Q_PER_WG + m) * V_HEAD_SIZE + value] = QV(acc, m);
        }
#else
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#    ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#    endif
            const uint head = head_base + m;
            const SOFTMAX_ACCUMULATOR_TYPE o = QV(acc, m) * QV(inv_l, m);
            if (seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1) {
                const size_t tmp_out_offset = (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE * total_partitions_num +
                                              (size_t)head * V_HEAD_SIZE * total_partitions_num +
                                              (size_t)partition_idx * V_HEAD_SIZE + value;
                tmp_out[tmp_out_offset] = TO_OUTPUT_TYPE(o);
            } else {
                const size_t output_offset =
                    (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE + (size_t)head * V_HEAD_SIZE + value;
                output[output_offset] = TO_OUTPUT_TYPE(o);
            }
        }
#endif
    }

#ifdef SV_NEEDS_OUTPUT_REDUCTION
    // Only reached when there were fewer head-dim tiles than subgroups, so SV_KEY_SGS subgroups
    // split the keys behind each tile and their partials still have to be summed. No rescale: every
    // probability was already normalised against m_wg.
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint cd = sgid; cd < V_TILES; cd += SG_PER_WG) {
        const uint value = cd * SUBGROUP_SIZE + lane;
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#    ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#    endif
            SOFTMAX_ACCUMULATOR_TYPE o = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            unroll_for(uint ks = 0; ks < SV_KEY_SGS; ++ks) {
                o += slm_out[(ks * Q_PER_WG + m) * V_HEAD_SIZE + value];
            }
            o *= QV(inv_l, m);
            const uint head = head_base + m;
            if (seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1) {
                const size_t tmp_out_offset = (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE * total_partitions_num +
                                              (size_t)head * V_HEAD_SIZE * total_partitions_num +
                                              (size_t)partition_idx * V_HEAD_SIZE + value;
                tmp_out[tmp_out_offset] = TO_OUTPUT_TYPE(o);
            } else {
                const size_t output_offset =
                    (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE + (size_t)head * V_HEAD_SIZE + value;
                output[output_offset] = TO_OUTPUT_TYPE(o);
            }
        }
    }
#endif

    if (seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1 && sgid == 0 && lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#endif
            const size_t offset = (size_t)seq_idx * HEADS_NUM * total_partitions_num +
                                  (size_t)(head_base + m) * total_partitions_num + partition_idx;
            exp_sums[offset] = QV(l_wg, m);
            max_logits[offset] = QV(m_wg, m);
        }
    }
}
