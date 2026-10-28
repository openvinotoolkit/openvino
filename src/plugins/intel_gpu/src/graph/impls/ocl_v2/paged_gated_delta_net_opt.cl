// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "paged_gated_delta_net_type_utils.cl"

// Gated delta rule, S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T, o_t = S_t q_t.
//
// Terms:
//  - S[d_v][d] is one head's V_HEAD_DIM x K_HEAD_DIM state; d_v indexes the v
//    axis, d the k axis.
//  - A q or k row is query[token][head][:] or key[token][head][:], the
//    K_HEAD_DIM elements of one token and head; a v row is S[d_v][:], the
//    K_HEAD_DIM elements of row d_v of the state.
//  - A tile is the V_BLOCK_SIZE v rows one subgroup holds.
//  - A lane is a work item of the subgroup, numbered by get_sub_group_local_id()
//    from 0 to SUBGROUP_SIZE - 1.
//  - A lane's interleaved share of a row is its elements d = lane,
//    lane + SUBGROUP_SIZE, lane + 2 * SUBGROUP_SIZE, ..., which is what a
//    subgroup block read of the row gives that lane.
//  - The input state is the state read from the subsequence's first block; a
//    snapshot is the state written to its next block after every
//    cache_interval tokens and after its last token.
//
// The state is held in registers and distributed over the subgroup in one of
// two tilings, named after the index of S that runs across the lanes: under
// d_v-on-lanes lane i holds row i of the tile, whole; under d-on-lanes every
// lane holds its interleaved share of each row of the tile. The host picks the
// tiling and V_BLOCK_SIZE; see get_v_rows_per_thread() and get_d_on_lanes_v_rows().
//
// Outside PLAIN_TOKEN_LOOP, operands for token t+1 are issued before the state
// work for token t, so that the loads can overlap the arithmetic. The state
// update and the output dot product run in one loop rather than two.

#ifndef FUSE_QK_L2NORM
#    define FUSE_QK_L2NORM 0
#endif
#ifndef Q_L2_NORM_EPS
#    define Q_L2_NORM_EPS 1e-6f
#endif
#ifndef K_L2_NORM_EPS
#    define K_L2_NORM_EPS 1e-6f
#endif

#if ((K_HEAD_DIM % 16) != 0) || ((V_HEAD_DIM % 16) != 0)
#    error "paged_gated_delta_net_opt requires K_HEAD_DIM and V_HEAD_DIM divisible by 16"
#endif
#if ((K_HEAD_DIM % SUBGROUP_SIZE) != 0)
#    error "K_HEAD_DIM must be a multiple of SUBGROUP_SIZE"
#endif

// Elements of the k axis per lane: the size of an interleaved share.
#define K_LANE_ELEMS (K_HEAD_DIM / SUBGROUP_SIZE)
#define K_VEC_COUNT  (K_LANE_ELEMS / K_VEC_SIZE)

#if ((K_LANE_ELEMS % K_VEC_SIZE) != 0)
#    error "K_VEC_SIZE must divide K_LANE_ELEMS"
#endif
#if ((V_HEAD_DIM % V_BLOCK_SIZE) != 0)
#    error "V_BLOCK_SIZE must tile V_HEAD_DIM"
#endif
#if ((K_HEAD_DIM % ACC_WAYS) != 0)
#    error "ACC_WAYS must divide K_HEAD_DIM"
#endif

// --------------------------------------------------------------------------
// Tiling
//
// D_LANES selects the tiling (see the top of the file). d_v-on-lanes keeps
// K_HEAD_DIM floats of state per lane, d-on-lanes V_BLOCK_SIZE * K_LANE_ELEMS.
// --------------------------------------------------------------------------
#if (!D_LANES && (V_BLOCK_SIZE != SUBGROUP_SIZE))
#    error "under d_v-on-lanes each lane holds one v row, so V_BLOCK_SIZE must equal SUBGROUP_SIZE"
#endif

// PLAIN_TOKEN_LOOP replaces the software pipeline and the prefetch with a plain
// token loop.
#if (PLAIN_TOKEN_LOOP && !D_LANES)
#    error "PLAIN_TOKEN_LOOP requires D_LANES"
#endif

// --------------------------------------------------------------------------
// The distributed q/k/state read
//
// A subgroup block read of N elements per lane returns a vector of N in each
// lane: component j of lane l's vector is the element at offset
// j * SUBGROUP_SIZE + l. Chunk c reads K_VEC_SIZE elements per lane at offset
// c * K_VEC_SIZE * SUBGROUP_SIZE, so array element e of lane l holds row element
// e * SUBGROUP_SIZE + l: the interleaved share, in the order the unrolled loops
// below index it, so no shuffle is needed.
//
// Results go into a float[] rather than a vector because a vector's components
// can only be named .sN; an array indexed by a constant is register-promoted
// after unrolling.
// --------------------------------------------------------------------------
#if (K_VEC_SIZE == 8)
#    define PGDN_SPREAD(dst, f, c)                                             \
        dst[(c) * 8 + 0] = f.s0; dst[(c) * 8 + 1] = f.s1;                      \
        dst[(c) * 8 + 2] = f.s2; dst[(c) * 8 + 3] = f.s3;                      \
        dst[(c) * 8 + 4] = f.s4; dst[(c) * 8 + 5] = f.s5;                      \
        dst[(c) * 8 + 6] = f.s6; dst[(c) * 8 + 7] = f.s7
#    define PGDN_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 8)
#elif (K_VEC_SIZE == 4)
#    define PGDN_SPREAD(dst, f, c)                                             \
        dst[(c) * 4 + 0] = f.s0; dst[(c) * 4 + 1] = f.s1;                      \
        dst[(c) * 4 + 2] = f.s2; dst[(c) * 4 + 3] = f.s3
#    define PGDN_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 4)
#elif (K_VEC_SIZE == 2)
#    define PGDN_SPREAD(dst, f, c)                                             \
        dst[(c) * 2 + 0] = f.s0; dst[(c) * 2 + 1] = f.s1
#    define PGDN_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 2)
#else
#    define PGDN_SPREAD(dst, f, c) dst[(c)] = f
#    define PGDN_CHUNK_DECV(KT, v) PGDN_SDEC(KT, v)
#endif

#define PGDN_READ_CHUNK(TYPE, KT, dst, p, base, c)                             \
    do {                                                                       \
        const MAKE_VECTOR_TYPE(float, K_VEC_SIZE) _f = PGDN_CHUNK_DECV(        \
            KT, BLOCK_READN(TYPE, K_VEC_SIZE, (p),                             \
                            (base) + (c) * K_VEC_SIZE * SUBGROUP_SIZE));       \
        PGDN_SPREAD(dst, _f, c);                                               \
    } while (0)

#if (K_VEC_COUNT == 1)
#    define PGDN_READ_LANE(TYPE, KT, dst, p, base)                             \
        do {                                                                   \
            PGDN_READ_CHUNK(TYPE, KT, dst, p, base, 0);                        \
        } while (0)
#elif (K_VEC_COUNT == 2)
#    define PGDN_READ_LANE(TYPE, KT, dst, p, base)                             \
        do {                                                                   \
            PGDN_READ_CHUNK(TYPE, KT, dst, p, base, 0);                        \
            PGDN_READ_CHUNK(TYPE, KT, dst, p, base, 1);                        \
        } while (0)
#else
#    define PGDN_READ_LANE(TYPE, KT, dst, p, base)                             \
        do {                                                                   \
            unroll_for(int _c = 0; _c < K_VEC_COUNT; ++_c)                     \
                PGDN_READ_CHUNK(TYPE, KT, dst, p, base, _c);                   \
        } while (0)
#endif

// The inverse, for the d-on-lanes state: the same chunks, written back with
// block writes.
#if (K_VEC_SIZE == 8)
#    define PGDN_GATHER(src, c)                                                \
        (float8)(src[(c) * 8 + 0], src[(c) * 8 + 1], src[(c) * 8 + 2],         \
                 src[(c) * 8 + 3], src[(c) * 8 + 4], src[(c) * 8 + 5],         \
                 src[(c) * 8 + 6], src[(c) * 8 + 7])
#    define PGDN_CHUNK_ENCV(KT, f) PGDN_ENCV(KT, f, 8)
#elif (K_VEC_SIZE == 4)
#    define PGDN_GATHER(src, c)                                                \
        (float4)(src[(c) * 4 + 0], src[(c) * 4 + 1], src[(c) * 4 + 2], src[(c) * 4 + 3])
#    define PGDN_CHUNK_ENCV(KT, f) PGDN_ENCV(KT, f, 4)
#elif (K_VEC_SIZE == 2)
#    define PGDN_GATHER(src, c)    (float2)(src[(c) * 2 + 0], src[(c) * 2 + 1])
#    define PGDN_CHUNK_ENCV(KT, f) PGDN_ENCV(KT, f, 2)
#else
#    define PGDN_GATHER(src, c)    src[(c)]
#    define PGDN_CHUNK_ENCV(KT, f) PGDN_SENC(KT, f)
#endif

#define S_WRITE(p, base, src)                                                  \
    do {                                                                       \
        unroll_for(int _c = 0; _c < K_VEC_COUNT; ++_c)                         \
            BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, (p),                         \
                         (base) + _c * K_VEC_SIZE * SUBGROUP_SIZE,             \
                         PGDN_CHUNK_ENCV(S_KT, PGDN_GATHER(src, _c)));         \
    } while (0)

#define Q_READ(dst, p, base) PGDN_READ_LANE(INPUT0_TYPE, Q_KT, dst, p, base)
#define K_READ(dst, p, base) PGDN_READ_LANE(INPUT1_TYPE, K_KT, dst, p, base)
#define S_READ(dst, p, base) PGDN_READ_LANE(INPUT3_TYPE, S_KT, dst, p, base)

// One token's operands: q and k, normalized when FUSE_QK_L2NORM is set; v,
// V_BLOCK_SIZE elements under d-on-lanes and one per lane under d_v-on-lanes;
// exp(gate); beta.
#if D_LANES
#    define PGDN_V_READ(dst, p) VLOAD_VB(dst, p)
#else
#    define PGDN_V_READ(dst, p) dst = PGDN_SDEC(V_KT, BLOCK_READN(INPUT2_TYPE, 1, p, 0))
#endif
#define PGDN_READ_TOKEN(q, k, v, g, b)                                         \
    do {                                                                       \
        Q_READ(q, qp, 0);                                                      \
        K_READ(k, kp, 0);                                                      \
        FUNC_CALL(l2_normalize_qk)(q, k, scale_factor);                        \
        PGDN_V_READ(v, vp);                                                    \
        g = exp(PGDN_SDEC(G_KT, *gp));                                         \
        b = PGDN_SDEC(B_KT, *bp);                                              \
    } while (0)

// Under d_v-on-lanes the dot products over the k axis run inside each lane, which
// needs every element of the q and k rows: element d is array element
// d / SUBGROUP_SIZE of lane d % SUBGROUP_SIZE, so KD broadcasts it from there.
#define KD(v, d) sub_group_broadcast((v)[(d) / SUBGROUP_SIZE], (d) % SUBGROUP_SIZE)

// With V_BLOCK_SIZE 4 or 8, the V_BLOCK_SIZE contiguous elements are read in one
// access rather than V_BLOCK_SIZE separate loads. start_iv and the output strides
// are multiples of V_BLOCK_SIZE, and so are the value strides unless the v axis
// of the value is padded, so the access is naturally aligned.
#if (V_BLOCK_SIZE == 8)
#    define VLOAD_VB(dst, p)                                                   \
        do {                                                                   \
            const float8 _v = PGDN_DECV(                                       \
                V_KT,                                                          \
                *(const __global MAKE_VECTOR_TYPE(INPUT2_TYPE, 8)*)(p), 8);    \
            dst[0] = _v.s0; dst[1] = _v.s1; dst[2] = _v.s2; dst[3] = _v.s3;    \
            dst[4] = _v.s4; dst[5] = _v.s5; dst[6] = _v.s6; dst[7] = _v.s7;    \
        } while (0)
#    define VSTORE_VB(p, src)                                                  \
        do {                                                                   \
            *(__global MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)*)(p) =                 \
                PGDN_ENCV(O_KT, ((float8)(src[0], src[1], src[2], src[3],      \
                                          src[4], src[5], src[6], src[7])),    \
                          8);                                                  \
        } while (0)
#elif (V_BLOCK_SIZE == 4)
#    define VLOAD_VB(dst, p)                                                   \
        do {                                                                   \
            const float4 _v = PGDN_DECV(                                       \
                V_KT,                                                          \
                *(const __global MAKE_VECTOR_TYPE(INPUT2_TYPE, 4)*)(p), 4);    \
            dst[0] = _v.x; dst[1] = _v.y; dst[2] = _v.z; dst[3] = _v.w;        \
        } while (0)
#    define VSTORE_VB(p, src)                                                  \
        do {                                                                   \
            *(__global MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)*)(p) =                 \
                PGDN_ENCV(O_KT, ((float4)(src[0], src[1], src[2], src[3])),    \
                          4);                                                  \
        } while (0)
#else
#    define VLOAD_VB(dst, p)                                                   \
        do {                                                                   \
            unroll_for(int _p = 0; _p < V_BLOCK_SIZE; ++_p)                    \
                dst[_p] = PGDN_SDEC(V_KT, (p)[_p]);                            \
        } while (0)
#    define VSTORE_VB(p, src)                                                  \
        do {                                                                   \
            unroll_for(int _p = 0; _p < V_BLOCK_SIZE; ++_p)                    \
                (p)[_p] = PGDN_SENC(O_KT, src[_p]);                            \
        } while (0)
#endif

// Under d_v-on-lanes a lane's state is one contiguous v row of K_HEAD_DIM
// elements, loaded in 16-element vectors: one access per lane per vector rather
// than one per element. K_HEAD_DIM is a multiple of 16, so each vector is
// naturally aligned.
//
// NOTE: The state is stored element by element rather than in such vectors.
// Measured on Panther Lake at (128, 128), that is faster in f32 and on longer
// f16 subsequences.
#define S_VEC16 MAKE_VECTOR_TYPE(INPUT3_TYPE, 16)
#define S_ROW_LOAD(dst, p)                                                     \
    do {                                                                       \
        unroll_for(int _c = 0; _c < K_HEAD_DIM / 16; ++_c) {                   \
            const float16 _f =                                                 \
                PGDN_DECV(S_KT, ((const __global S_VEC16*)(p))[_c], 16);       \
            dst[_c * 16 + 0] = _f.s0;  dst[_c * 16 + 1] = _f.s1;               \
            dst[_c * 16 + 2] = _f.s2;  dst[_c * 16 + 3] = _f.s3;               \
            dst[_c * 16 + 4] = _f.s4;  dst[_c * 16 + 5] = _f.s5;               \
            dst[_c * 16 + 6] = _f.s6;  dst[_c * 16 + 7] = _f.s7;               \
            dst[_c * 16 + 8] = _f.s8;  dst[_c * 16 + 9] = _f.s9;               \
            dst[_c * 16 + 10] = _f.sa; dst[_c * 16 + 11] = _f.sb;              \
            dst[_c * 16 + 12] = _f.sc; dst[_c * 16 + 13] = _f.sd;              \
            dst[_c * 16 + 14] = _f.se; dst[_c * 16 + 15] = _f.sf;              \
        }                                                                      \
    } while (0)

// Accumulator chains cannot exceed what there is to chain: under d-on-lanes a
// lane holds K_LANE_ELEMS of the k axis, not K_HEAD_DIM.
#define WAYS_D (ACC_WAYS < K_LANE_ELEMS ? ACC_WAYS : K_LANE_ELEMS)

inline void FUNC(l2_normalize_qk)(float* qq, float* kk, float scale_factor) {
#if FUSE_QK_L2NORM
    float qs = 0.0f, ks = 0.0f;
    unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
        qs += qq[c] * qq[c];
        ks += kk[c] * kk[c];
    }
    const float q_scale = rsqrt(sub_group_reduce_add(qs) + Q_L2_NORM_EPS) * scale_factor;
    const float k_scale = rsqrt(sub_group_reduce_add(ks) + K_L2_NORM_EPS);
#else
    const float q_scale = scale_factor;
    const float k_scale = 1.0f;
#endif
    unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
        qq[c] *= q_scale;
        kk[c] *= k_scale;
    }
}

#if D_LANES
// One token under d-on-lanes, in two passes over the lane's state elements.
// Pass 1 takes h_k against the undecayed state and multiplies the result by the
// decay a = exp(gate): a * (S . k) == (a * S) . k, so the decay reaches the
// state only in pass 2 and the decayed state is computed once. Pass 2 fuses the
// state update with the output dot product. Every lane returns the totals in
// o_val. Under d_v-on-lanes the state is decayed in pass 1 instead.
//
// NOTE: Measured on Panther Lake at (256, 256), decaying the state in pass 1
// here is slower in f16 and f32 and spills in f32.
inline void FUNC(d_on_lanes_token)(float state[V_BLOCK_SIZE][K_LANE_ELEMS],
                                const float* qv,
                                const float* kv,
                                const float* v_val,
                                float b_g,
                                float b_beta,
                                float* o_val) {
    float update[V_BLOCK_SIZE];
    unroll_for(int p = 0; p < V_BLOCK_SIZE; ++p) {
        float acc[WAYS_D];
        unroll_for(int a = 0; a < WAYS_D; ++a) acc[a] = state[p][a] * kv[a];
        unroll_for(int c = WAYS_D; c < K_LANE_ELEMS; ++c) acc[c % WAYS_D] += state[p][c] * kv[c];
        float part = 0.0f;
        unroll_for(int a = 0; a < WAYS_D; ++a) part += acc[a];
        const float h_k = sub_group_reduce_add(part) * b_g;
        update[p] = (v_val[p] - h_k) * b_beta;
    }
    unroll_for(int p = 0; p < V_BLOCK_SIZE; ++p) {
        float acc[WAYS_D];
        unroll_for(int a = 0; a < WAYS_D; ++a) {
            state[p][a] = state[p][a] * b_g + update[p] * kv[a];
            acc[a] = state[p][a] * qv[a];
        }
        unroll_for(int c = WAYS_D; c < K_LANE_ELEMS; ++c) {
            state[p][c] = state[p][c] * b_g + update[p] * kv[c];
            acc[c % WAYS_D] += state[p][c] * qv[c];
        }
        float part = 0.0f;
        unroll_for(int a = 0; a < WAYS_D; ++a) part += acc[a];
        o_val[p] = sub_group_reduce_add(part);
    }
}

// Each lane holds its interleaved share of each v row, so a snapshot is
// written row by row with block writes.
inline void FUNC(d_on_lanes_snapshot)(__global INPUT3_TYPE* sp, float state[V_BLOCK_SIZE][K_LANE_ELEMS]) {
    unroll_for(int p = 0; p < V_BLOCK_SIZE; ++p) S_WRITE(sp, p * K_HEAD_DIM, state[p]);
}
#endif

// reqd_work_group_size lets the compiler prove the launch shape and therefore
// prove gate, beta and the prefetch addresses uniform. The host sets it under
// d-on-lanes.
#if OCL_REQD_WG
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
#endif
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(paged_gated_delta_net_opt)
(__global INPUT0_TYPE* query,
 __global INPUT1_TYPE* key,
 __global INPUT2_TYPE* value,
 __global INPUT3_TYPE* recurrent_state_table,
 __global INPUT4_TYPE* gate,
 __global INPUT5_TYPE* beta,
 __global INPUT6_TYPE* subsequence_begins,
 __global INPUT7_TYPE* block_indices,
 __global INPUT8_TYPE* block_indices_begins,
 __global INPUT9_TYPE* past_lens,
 __global INPUT10_TYPE* cache_interval,
 __global OUTPUT_TYPE* output,
 int num_sequences,
 int query_head_offset,
 int key_head_offset,
 int value_head_offset,
 int q_token_stride,
 int q_head_stride,
 int k_token_stride,
 int k_head_stride,
 int v_token_stride,
 int v_head_stride) {
    const int v_tile = (int)get_group_id(0);
    const int h = (int)get_global_id(1);
    const int seq = (int)get_global_id(2);
    const int lane = (int)get_sub_group_local_id();
    const int start_iv = v_tile * V_BLOCK_SIZE;

    const float scale_factor = SCALE_FACTOR;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int past_len = past_lens[seq];
    const int interval = cache_interval[seq];
    // An empty subsequence has nothing to update, and the reads below would go
    // past its end in query, key, value, gate and beta and take a snapshot block
    // it need not have. Uniform across the work group.
    if (token_begin >= token_end) {
        return;
    }
    const int prev_nums = interval > 0 ? past_len % interval : 0;

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const int q_head_base = (hk + query_head_offset) * q_head_stride;
    const int k_head_base = (hk + key_head_offset) * k_head_stride;
    const int v_head_base = (h + value_head_offset) * v_head_stride;
    const int state_stride = K_HEAD_DIM * V_HEAD_DIM;

    const int initial_block_id = block_indices[block_begin];
    // The first snapshot's block, which a non-empty subsequence always writes,
    // read ahead so that the snapshot does not wait for it behind the output
    // stores.
    int next_block_id = block_indices[block_begin + 1];
    const long initial_base =
        (long)(initial_block_id * V_HEAD_NUM + h) * state_stride + (long)start_iv * K_HEAD_DIM;

    // The input state: under d-on-lanes each lane's interleaved share of each
    // v row, read with block reads; under d_v-on-lanes each lane's whole v row,
    // loaded in 16-element vectors.
#if D_LANES
    float state[V_BLOCK_SIZE][K_LANE_ELEMS];
    {
        const __global INPUT3_TYPE* sp = recurrent_state_table + initial_base;
        unroll_for(int p = 0; p < V_BLOCK_SIZE; ++p) {
            float row[K_LANE_ELEMS];
            S_READ(row, sp, p * K_HEAD_DIM);
            unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) state[p][c] = row[c];
        }
    }
#else
    float state[K_HEAD_DIM];
    {
        const __global INPUT3_TYPE* sp = recurrent_state_table + initial_base + (long)lane * K_HEAD_DIM;
        S_ROW_LOAD(state, sp);
    }
#endif

    int token = token_begin;
    int slot = 1;
    int tokens_to_next_boundary =
        interval > 0 ? (prev_nums > 0 ? (interval - prev_nums) : interval) : (token_end - token_begin);

    const __global INPUT0_TYPE* qp = query + (long)token * q_token_stride + q_head_base;
    const __global INPUT1_TYPE* kp = key + (long)token * k_token_stride + k_head_base;
    const __global INPUT2_TYPE* vp = value + (long)token * v_token_stride + v_head_base + start_iv;
    const __global INPUT4_TYPE* gp = gate + (long)token * V_HEAD_NUM + h;
    const __global INPUT5_TYPE* bp = beta + (long)token * V_HEAD_NUM + h;
    __global OUTPUT_TYPE* op = output + (long)(token * V_HEAD_NUM + h) * V_HEAD_DIM + start_iv;

#if !PLAIN_TOKEN_LOOP
#if PF_AHEAD > 0
    // Clamped: the prefetch runs PF_AHEAD tokens ahead, which is past the end
    // for a subsequence shorter than that.
    const int pf_first = min(PF_AHEAD, max(token_end - 1 - token, 0));
    const __global INPUT0_TYPE* pfq = qp + (long)pf_first * q_token_stride;
    const __global INPUT1_TYPE* pfk = kp + (long)pf_first * k_token_stride;
    const __global INPUT2_TYPE* pfv = vp + (long)pf_first * v_token_stride;
#endif

    // Prologue of the software pipeline: the first token's operands.
    float q_pre[K_LANE_ELEMS], k_pre[K_LANE_ELEMS];
#if D_LANES
    float v_pre[V_BLOCK_SIZE];
#else
    float v_pre;
#endif
    float g_pre, b_pre;
    PGDN_READ_TOKEN(q_pre, k_pre, v_pre, g_pre, b_pre);
#endif

    while (token < token_end) {
        const int chunk_end = min(token + tokens_to_next_boundary, token_end);

#if PLAIN_TOKEN_LOOP
        for (; token < chunk_end; ++token) {
            float qv[K_LANE_ELEMS], kv[K_LANE_ELEMS], v_val[V_BLOCK_SIZE], o_val[V_BLOCK_SIZE], b_g, b_beta;
            PGDN_READ_TOKEN(qv, kv, v_val, b_g, b_beta);
            FUNC_CALL(d_on_lanes_token)(state, qv, kv, v_val, b_g, b_beta, o_val);
            if (lane == 0)
                VSTORE_VB(op, o_val);
            qp += q_token_stride;
            kp += k_token_stride;
            vp += v_token_stride;
            gp += V_HEAD_NUM;
            bp += V_HEAD_NUM;
            op += V_HEAD_NUM * V_HEAD_DIM;
        }
#else
        // The pipeline stages token+1 and the prefetch reads PF_AHEAD further,
        // so both would leave the buffer at the end of a subsequence. Rather
        // than check the bounds in every iteration, the token range is split
        // into segments and the advance strides are zeroed where advancing
        // would overrun: segment 0 advances both, segment 1 freezes the
        // prefetch, segment 2 freezes both. A frozen pointer re-reads the last
        // valid token and the staged value is discarded by the loop exit, so no
        // arithmetic changes.
        //
        // Unrolled under d_v-on-lanes only, so that each segment there sees its
        // strides as constants.
#if !D_LANES
        __attribute__((opencl_unroll_hint))
#endif
        for (int seg = 0; seg < 3; ++seg) {
            int seg_end;
            if (seg == 0)
                seg_end = min(chunk_end, max(token_end - PF_AHEAD - 1, token));
            else if (seg == 1)
                seg_end = min(chunk_end, max(token_end - 1, token));
            else
                seg_end = chunk_end;

            const int qs = (seg < 2) ? q_token_stride : 0;
            const int ks = (seg < 2) ? k_token_stride : 0;
            const int vs = (seg < 2) ? v_token_stride : 0;
            const int gs = (seg < 2) ? V_HEAD_NUM : 0;
            const int bs = (seg < 2) ? V_HEAD_NUM : 0;
            const int pq = (seg < 1) ? q_token_stride : 0;
            const int pk = (seg < 1) ? k_token_stride : 0;
            const int pv = (seg < 1) ? v_token_stride : 0;

            for (; token < seg_end; ++token) {
                float qv[K_LANE_ELEMS], kv[K_LANE_ELEMS];
                unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
                    qv[c] = q_pre[c];
                    kv[c] = k_pre[c];
                }
#if D_LANES
                float v_val[V_BLOCK_SIZE];
                unroll_for(int p = 0; p < V_BLOCK_SIZE; ++p) v_val[p] = v_pre[p];
#else
                const float v_val = v_pre;
#endif
                const float b_g = g_pre;
                const float b_beta = b_pre;

                {
                    qp += qs;
                    kp += ks;
                    vp += vs;
                    gp += gs;
                    bp += bs;
                    PGDN_READ_TOKEN(q_pre, k_pre, v_pre, g_pre, b_pre);
                }

#if PF_AHEAD > 0
                // PF_BYTES is the size of each prefetch request; it is not meant
                // to cover the row. The prefetch issues one request each for q,
                // k and v per token, at the start of each one's row.
                {
                    pfq += pq;
                    pfk += pk;
                    pfv += pv;
                    prefetch(pfq, PF_BYTES / sizeof(INPUT0_TYPE));
                    prefetch(pfk, PF_BYTES / sizeof(INPUT1_TYPE));
                    prefetch(pfv, PF_BYTES / sizeof(INPUT2_TYPE));
                }
#endif

#if D_LANES
                float o_val[V_BLOCK_SIZE];
                FUNC_CALL(d_on_lanes_token)(state, qv, kv, v_val, b_g, b_beta, o_val);
                // Every lane holds the totals, so lane 0 alone writes them, with
                // an ordinary store: a block write has to be executed by the whole
                // subgroup and cannot go under this if.
                if (lane == 0)
                    VSTORE_VB(op, o_val);
#else
                // Pass 1 decays the state and takes h_k against the decayed state;
                // pass 2 adds the update and takes the output dot product. Both
                // dot products over the k axis run inside the lane. In
                // d_on_lanes_token() the state is decayed in pass 2 instead.
                //
                // NOTE: Measured on Panther Lake at (128, 128), decaying the state
                // in pass 1 here is faster in f32 on long subsequences.
                float hk_acc[ACC_WAYS];
                unroll_for(int a = 0; a < ACC_WAYS; ++a) {
                    state[a] *= b_g;
                    hk_acc[a] = state[a] * KD(kv, a);
                }
                unroll_for(int d = ACC_WAYS; d < K_HEAD_DIM; ++d) {
                    state[d] *= b_g;
                    hk_acc[d % ACC_WAYS] += state[d] * KD(kv, d);
                }
                float h_k = 0.0f;
                unroll_for(int a = 0; a < ACC_WAYS; ++a) h_k += hk_acc[a];
                const float update = (v_val - h_k) * b_beta;

                float out_acc[ACC_WAYS];
                unroll_for(int a = 0; a < ACC_WAYS; ++a) {
                    state[a] = fma(KD(kv, a), update, state[a]);
                    out_acc[a] = state[a] * KD(qv, a);
                }
                unroll_for(int d = ACC_WAYS; d < K_HEAD_DIM; ++d) {
                    state[d] = fma(KD(kv, d), update, state[d]);
                    out_acc[d % ACC_WAYS] += state[d] * KD(qv, d);
                }
                float o = 0.0f;
                unroll_for(int a = 0; a < ACC_WAYS; ++a) o += out_acc[a];

                BLOCK_WRITEN(OUTPUT_TYPE, 1, op, 0, PGDN_SENC(O_KT, o));
#endif
                // The output is a contiguous [token][v_head][v] tensor with no
                // padding, so op advances by V_HEAD_NUM * V_HEAD_DIM per token,
                // not by v_token_stride: the value input may be a padded view of
                // a fused projection, whose token stride is larger.
                op += V_HEAD_NUM * V_HEAD_DIM;
            }
        }
#endif

        const int block_id = next_block_id;
        slot++;
        if (token < token_end)
            next_block_id = block_indices[block_begin + slot];
        const long block_base =
            (long)(block_id * V_HEAD_NUM + h) * state_stride + (long)start_iv * K_HEAD_DIM;
#if D_LANES
        FUNC_CALL(d_on_lanes_snapshot)(recurrent_state_table + block_base, state);
#else
        unroll_for(int d = 0; d < K_HEAD_DIM; ++d)
            (recurrent_state_table + block_base + (long)lane * K_HEAD_DIM)[d] = PGDN_SENC(S_KT, state[d]);
#endif

        if (interval > 0)
            tokens_to_next_boundary = interval;
    }
}
