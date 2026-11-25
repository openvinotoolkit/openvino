// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#if REPLAY_STAGE == 1
#    include "include/batch_headers/sub_group_block_read.cl"
#    include "include/batch_headers/sub_group_block_write.cl"
#    include "include/batch_headers/sub_group_shuffle.cl"

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    sum = sub_group_reduce_add(sum);
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(dot8_fma)(float8 a, float8 b) {
    float acc = a.s0 * b.s0;
    acc = fma(a.s1, b.s1, acc);
    acc = fma(a.s2, b.s2, acc);
    acc = fma(a.s3, b.s3, acc);
    acc = fma(a.s4, b.s4, acc);
    acc = fma(a.s5, b.s5, acc);
    acc = fma(a.s6, b.s6, acc);
    acc = fma(a.s7, b.s7, acc);
    return acc;
}

#    if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
inline float8 FUNC(load_q8_as_float8)(const __global INPUT0_TYPE* p) {
#        if INPUT0_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#        elif INPUT0_TYPE_SIZE == 4
    return convert_float8(vload8(0, (const __global float*)p));
#        else
    float8 v = (float8)(0.0f);
#            pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline float8 FUNC(load_k8_as_float8)(const __global INPUT1_TYPE* p) {
#        if INPUT1_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#        elif INPUT1_TYPE_SIZE == 4
    return convert_float8(vload8(0, (const __global float*)p));
#        else
    float8 v = (float8)(0.0f);
#            pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline float4 FUNC(load_v4_as_float4)(const __global INPUT2_TYPE* p) {
#        if INPUT2_TYPE_SIZE == 2
    return convert_float4(vload4(0, (const __global half*)p));
#        elif INPUT2_TYPE_SIZE == 4
    return convert_float4(vload4(0, (const __global float*)p));
#        else
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#            pragma unroll
    for (int i = 0; i < 4; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline void FUNC(
    prepare_qk)(const __global INPUT0_TYPE* q, const __global INPUT1_TYPE* k, size_t q_offset, size_t k_offset, int lane_k_base, float8* b_q, float8* b_k) {
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
#        if FUSE_QK_L2NORM
    // normalize k and q (l2norm + q scale)
    float k_sum = 0.0f;
    float q_sum = 0.0f;
#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        int idx = lane_k_base + c * 8;
        float8 lane_k = FUNC(load_k8_as_float8)(k + k_offset + idx);
        float8 lane_q = FUNC(load_q8_as_float8)(q + q_offset + idx);
        b_k[c] = lane_k;
        b_q[c] = lane_q;
        k_sum += FUNC(dot8_fma)(lane_k, lane_k);
        q_sum += FUNC(dot8_fma)(lane_q, lane_q);
    }
    float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
    float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);

#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        b_k[c] *= (float8)(k_scale);
        b_q[c] *= (float8)(q_scale);
    }
#        else
#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        int idx = lane_k_base + c * 8;
        float8 lane_k = FUNC(load_k8_as_float8)(k + k_offset + idx);
        float8 lane_q = FUNC(load_q8_as_float8)(q + q_offset + idx);
        b_k[c] = lane_k;
        b_q[c] = lane_q * (float8)(SCALE_FACTOR);
    }
#        endif
}
#    else
inline void FUNC(prepare_qk)(__global INPUT0_TYPE* q, __global INPUT1_TYPE* k, size_t q_offset, size_t k_offset, float* k_scale, float* q_scale) {
    *k_scale = 1.0f;
    *q_scale = SCALE_FACTOR;
#        if FUSE_QK_L2NORM
    float k_sum = 0.0f;
    float q_sum = 0.0f;
    for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
        float k_val = convert_float(k[k_offset + k_idx]);
        float q_val = convert_float(q[q_offset + k_idx]);
        k_sum += k_val * k_val;
        q_sum += q_val * q_val;
    }
    *k_scale = rsqrt(k_sum + K_L2_NORM_EPS);
    *q_scale = rsqrt(q_sum + Q_L2_NORM_EPS) * SCALE_FACTOR;
#        endif
}
#    endif


#endif

#define SF ((size_t)V_HEAD_NUM * K_HEAD_DIM * V_HEAD_DIM)
#define KF ((size_t)P * K_HEAD_NUM * K_HEAD_DIM)
#define VF ((size_t)P * V_HEAD_NUM * V_HEAD_DIM)
#define GF ((size_t)2 * P * V_HEAD_NUM * GATE_DIM)
#define TOTAL_FEATURES (SF + KF + VF + GF)
#define READ_BANK(bank, row, f) ((bank) == 0 ? state_bank[OUTPUT1_GET_INDEX(0,0,row,f)] : \
    (bank) == 1 ? key_bank[OUTPUT2_GET_INDEX(0,0,row,f)] : \
    (bank) == 2 ? value_bank[OUTPUT3_GET_INDEX(0,0,row,f)] : gate_bank[OUTPUT4_GET_INDEX(0,0,row,f)])
#define WRITE_BANK(bank, row, f, value) do { \
    if ((bank) == 0) state_bank[OUTPUT1_GET_INDEX(0,0,row,f)] = (value); \
    else if ((bank) == 1) key_bank[OUTPUT2_GET_INDEX(0,0,row,f)] = (value); \
    else if ((bank) == 2) value_bank[OUTPUT3_GET_INDEX(0,0,row,f)] = (value); \
    else gate_bank[OUTPUT4_GET_INDEX(0,0,row,f)] = (value); \
} while (0)

// Gather accepts negative indices. Keep all memory accesses inside the bank
// even when a caller supplies an index outside the ScatterNDUpdate domain.
inline int FUNC(row_index)(long i, int size) {
    return (int)clamp(i < 0 ? i + size : i, (long)0, (long)(size - 1));
}

inline bool FUNC(direct_state)(const __global INPUT9_TYPE* rows, int n, int mapping, int source, int destination, int slots) {
    if (source != destination) return false;
    for (int i=0; i<n; ++i)
        if (i != mapping && FUNC(row_index)(rows[2*i], slots) == source) return false;
    return true;
}

inline bool FUNC(reset_source)(const __global INPUT11_TYPE* resets, int count, int row, int slots) {
    for (int i = 0; i < count; ++i) if (FUNC(row_index)(resets[i],slots) == row) return true;
    return false;
}
inline bool FUNC(active_claim)(int bank, size_t feature, int T, int P) {
    if (bank == 0) return true;
    const int stored = min(T, P);
    if (bank == 1) return feature < (size_t)stored * K_HEAD_NUM * K_HEAD_DIM;
    if (bank == 2) return feature < (size_t)stored * V_HEAD_NUM * V_HEAD_DIM;
    const size_t token = feature / (V_HEAD_NUM * GATE_DIM);
    return token < stored || (token >= P && token < P + stored && feature % GATE_DIM == 0);
}

#if REPLAY_STAGE == 1
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
#endif
KERNEL(recurrent_gdn_cache_replay)(OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* q, const __global INPUT1_TYPE* k, const __global INPUT2_TYPE* v,
    __global INPUT3_TYPE* state_bank, const __global INPUT4_TYPE* g, const __global INPUT5_TYPE* beta,
    __global INPUT6_TYPE* key_bank, __global INPUT7_TYPE* value_bank, __global INPUT8_TYPE* gate_bank,
    const __global INPUT9_TYPE* row_map, const __global INPUT10_TYPE* control, const __global INPUT11_TYPE* reset_rows,
    __global OUTPUT_TYPE* output, int B, int T, int N, int J, int P,
    __global float* staged) {
    row_map += INPUT9_OFFSET;
    control += INPUT10_OFFSET;
    reset_rows += INPUT11_OFFSET;
    const size_t widths[4] = {SF, KF, VF, GF};
#if REPLAY_STAGE == 0
    const size_t index = get_global_id(0);
    if (index >= (size_t)N * TOTAL_FEATURES) return;
    size_t offset = index;
    int bank = 0;
    while (offset >= (size_t)N * widths[bank]) offset -= (size_t)N * widths[bank++];
    const int mapping = offset / widths[bank];
    const size_t feature = offset % widths[bank];
    const int destination = FUNC(row_index)(row_map[2 * mapping],OUTPUT1_SIZE_Y), source = FUNC(row_index)(row_map[2 * mapping + 1],OUTPUT1_SIZE_Y);
    if (mapping < B && ((bank == 0 && FUNC(direct_state)(row_map,N,mapping,source,destination,OUTPUT1_SIZE_Y)) || (bank != 0 && control[mapping] == 0))) return;
    staged[index] = FUNC(reset_source)(reset_rows, J, source, OUTPUT1_SIZE_Y) ? 0.0f : READ_BANK(bank, source, feature);
#else
    const size_t group = get_group_id(0), lane = get_sub_group_local_id();
    const int value_blocks = (V_HEAD_DIM + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    const size_t recurrence_groups = (size_t)B * V_HEAD_NUM * value_blocks;
    const bool keep = control[B] != 0;
    if (group >= recurrence_groups) {
        const size_t tile = SUBGROUP_SIZE * 4;
        const size_t tiles = (TOTAL_FEATURES + tile - 1) / tile;
        const size_t work = group - recurrence_groups;
        const int work_row = work / tiles;
        int mapping = -1, destination;
        if (work_row < N - B) {
            mapping = B + work_row;
            destination = FUNC(row_index)(row_map[2 * mapping],OUTPUT1_SIZE_Y);
        } else {
            const int reset_index = work_row - (N - B);
            if (reset_index >= J) return;
            destination = FUNC(row_index)(reset_rows[reset_index],OUTPUT1_SIZE_Y);
            if (lane == 0) for (int i = 0; i < N; ++i) if (FUNC(row_index)(row_map[2 * i],OUTPUT1_SIZE_Y) == destination) { mapping = i; break; }
            mapping = sub_group_broadcast(mapping, 0);
            if (mapping >= B) return;
        }
        for (int part = 0; part < 4; ++part) {
            size_t feature = (work % tiles) * tile + lane + part * SUBGROUP_SIZE;
            if (feature >= TOTAL_FEATURES) continue;
            int bank = 0;
            size_t prefix = 0;
            while (feature >= widths[bank]) { feature -= widths[bank]; prefix += widths[bank++]; }
            if (work_row < N - B) {
                WRITE_BANK(bank, destination, feature, staged[N * prefix + mapping * widths[bank] + feature]);
            } else if (mapping < 0 || !FUNC(active_claim)(bank, feature, T, P)) {
                WRITE_BANK(bank, destination, feature, 0.0f);
            }
        }
        return;
    }
    const int block = group % value_blocks;
    const int h = (group / value_blocks) % V_HEAD_NUM;
    const int b = group / (value_blocks * V_HEAD_NUM);
    const int source = FUNC(row_index)(row_map[2 * b + 1],OUTPUT1_SIZE_Y), destination = FUNC(row_index)(row_map[2 * b],OUTPUT1_SIZE_Y);
    const int hk = GROUPED_HEADS ? h / (V_HEAD_NUM / K_HEAD_NUM) : h % K_HEAD_NUM;
    const int hq = GROUPED_HEADS ? h / (V_HEAD_NUM / Q_HEAD_NUM) : h % Q_HEAD_NUM;
    const int first_v = block * V_BLOCK_SIZE;
    const int replay = (int)clamp((long)control[b], (long)0, (long)P);
    const bool direct = FUNC(direct_state)(row_map,N,b,source,destination,OUTPUT1_SIZE_Y);
    const bool source_reset = FUNC(reset_source)(reset_rows, J, source, OUTPUT1_SIZE_Y);
#define LOAD_STATE(key, value) (source_reset ? 0.0f : direct ? \
    state_bank[OUTPUT1_GET_INDEX(0,0,source,(h * V_HEAD_DIM + (value)) * K_HEAD_DIM + (key))] : \
    staged[b * SF + (h * V_HEAD_DIM + (value)) * K_HEAD_DIM + (key)])
#define STORE_STATE_VECTOR() do { \
    for (int column = 0; column < V_BLOCK_SIZE; ++column) \
        for (int chunk = 0; chunk < K_CHUNKS; ++chunk) \
            for (int element = 0; element < 8; ++element) { \
                const int key_index = lane_k_base + chunk * 8 + element; \
                state_bank[OUTPUT1_GET_INDEX(0,0,destination,(h * V_HEAD_DIM + first_v + column) * K_HEAD_DIM + key_index)] = state[column][chunk][element]; \
            } \
} while (0)
#if K_HEAD_DIM == 128 && V_HEAD_DIM % V_BLOCK_SIZE == 0
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
    const int lane_k_base = lane * (K_HEAD_DIM / SUBGROUP_SIZE);
    float8 state[V_BLOCK_SIZE][K_CHUNKS];
    for (int column = 0; column < V_BLOCK_SIZE; ++column)
        for (int chunk = 0; chunk < K_CHUNKS; ++chunk)
            for (int element = 0; element < 8; ++element)
                state[column][chunk][element] = LOAD_STATE(lane_k_base + chunk * 8 + element, first_v + column);
    for (int step = 0; step < replay + T; ++step) {
        const bool pending = step < replay;
        const int t = pending ? step : step - replay;
        if (!pending && t == 0 && keep) STORE_STATE_VECTOR();
        float8 key[K_CHUNKS], query[K_CHUNKS], decay[K_CHUNKS];
        float4 value;
        float beta_value;
        if (pending) {
            for (int chunk = 0; chunk < K_CHUNKS; ++chunk) {
                const size_t key_offset = N * SF + b * KF + (t * K_HEAD_NUM + hk) * K_HEAD_DIM + lane_k_base + chunk * 8;
                key[chunk] = vload8(0, staged + key_offset);
                query[chunk] = (float8)(0.0f);
                for (int element = 0; element < 8; ++element) {
                    const int gi = GATE_DIM == 1 ? 0 : lane_k_base + chunk * 8 + element;
                    decay[chunk][element] = exp(staged[N * (SF + KF + VF) + b * GF + (t * V_HEAD_NUM + h) * GATE_DIM + gi]);
                }
            }
            value = vload4(0, staged + N * (SF + KF) + b * VF + (t * V_HEAD_NUM + h) * V_HEAD_DIM + first_v);
            beta_value = staged[N * (SF + KF + VF) + b * GF + ((P + t) * V_HEAD_NUM + h) * GATE_DIM];
        } else {
            const size_t query_offset = INPUT0_GET_INDEX(b,t,hq,0), key_offset = INPUT1_GET_INDEX(b,t,hk,0);
            FUNC(prepare_qk)(q, k, query_offset, key_offset, lane_k_base, query, key);
            value = FUNC(load_v4_as_float4)(v + INPUT2_GET_INDEX(b,t,h,first_v));
            beta_value = convert_float(beta[INPUT5_GET_INDEX(b,t,h,0)]);
            for (int chunk = 0; chunk < K_CHUNKS; ++chunk)
                for (int element = 0; element < 8; ++element) {
                    const int gi = GATE_DIM == 1 ? 0 : lane_k_base + chunk * 8 + element;
                    decay[chunk][element] = exp(convert_float(g[INPUT4_GET_INDEX(b,t,h,gi)]));
                }
        }
        float4 dot = (float4)(0.0f);
        for (int column = 0; column < V_BLOCK_SIZE; ++column)
            for (int chunk = 0; chunk < K_CHUNKS; ++chunk) {
                state[column][chunk] *= decay[chunk];
                dot[column] += FUNC(dot8_fma)(state[column][chunk], key[chunk]);
            }
        for (int column = 0; column < V_BLOCK_SIZE; ++column) dot[column] = sub_group_reduce_add(dot[column]);
        const float4 delta = (value - dot) * beta_value;
        float4 result = (float4)(0.0f);
        for (int column = 0; column < V_BLOCK_SIZE; ++column)
            for (int chunk = 0; chunk < K_CHUNKS; ++chunk) {
                state[column][chunk] = fma(key[chunk], (float8)(delta[column]), state[column][chunk]);
                if (!pending) result[column] += FUNC(dot8_fma)(state[column][chunk], query[chunk]);
            }
        if (!pending) {
            for (int column = 0; column < V_BLOCK_SIZE; ++column) result[column] = sub_group_reduce_add(result[column]);
            if (lane == 0) for (int column = 0; column < V_BLOCK_SIZE; ++column)
                output[OUTPUT_GET_INDEX(b,t,h,first_v + column)] = TO_OUTPUT_TYPE(result[column]);
            if (t < P) {
                const int key_owner = GROUPED_HEADS ? hk * (V_HEAD_NUM / K_HEAD_NUM) : hk;
                if (block == 0 && h == key_owner)
                    for (int chunk = 0; chunk < K_CHUNKS; ++chunk)
                        for (int element = 0; element < 8; ++element)
                            key_bank[OUTPUT2_GET_INDEX(0,0,destination,(t * K_HEAD_NUM + hk) * K_HEAD_DIM + lane_k_base + chunk * 8 + element)] = key[chunk][element];
                if (lane == 0) for (int column = 0; column < V_BLOCK_SIZE; ++column)
                    value_bank[OUTPUT3_GET_INDEX(0,0,destination,(t * V_HEAD_NUM + h) * V_HEAD_DIM + first_v + column)] = value[column];
                if (block == 0) {
                    if (GATE_DIM == 1) {
                        if (lane == 0) gate_bank[OUTPUT4_GET_INDEX(0,0,destination,t * V_HEAD_NUM + h)] = convert_float(g[INPUT4_GET_INDEX(b,t,h,0)]);
                    } else {
                        for (int gi = lane; gi < GATE_DIM; gi += SUBGROUP_SIZE)
                            gate_bank[OUTPUT4_GET_INDEX(0,0,destination,(t * V_HEAD_NUM + h) * GATE_DIM + gi)] = convert_float(g[INPUT4_GET_INDEX(b,t,h,gi)]);
                    }
                    if (lane == 0) gate_bank[OUTPUT4_GET_INDEX(0,0,destination,((P + t) * V_HEAD_NUM + h) * GATE_DIM)] = beta_value;
                }
            }
        }
    }
    if (!keep) STORE_STATE_VECTOR();
#else
    if (lane != 0) return;
    float state[V_BLOCK_SIZE][K_HEAD_DIM];
    for (int column = 0; column < V_BLOCK_SIZE; ++column)
        for (int ki = 0; ki < K_HEAD_DIM; ++ki)
            state[column][ki] = first_v + column < V_HEAD_DIM ? LOAD_STATE(ki,first_v + column) : 0.0f;
    for (int step = 0; step < replay + T; ++step) {
        const bool pending = step < replay;
        const int t = pending ? step : step - replay;
        if (!pending && t == 0 && keep)
            for (int column = 0; column < V_BLOCK_SIZE && first_v + column < V_HEAD_DIM; ++column)
                for (int ki = 0; ki < K_HEAD_DIM; ++ki)
                    state_bank[OUTPUT1_GET_INDEX(0,0,destination,(h * V_HEAD_DIM + first_v + column) * K_HEAD_DIM + ki)] = state[column][ki];
        float key[K_HEAD_DIM], query[K_HEAD_DIM], decay[K_HEAD_DIM];
        float key_scale = 1.0f, query_scale = SCALE_FACTOR;
        if (!pending) FUNC(prepare_qk)((__global INPUT0_TYPE*)q, (__global INPUT1_TYPE*)k,
            INPUT0_GET_INDEX(b,t,hq,0), INPUT1_GET_INDEX(b,t,hk,0), &key_scale, &query_scale);
        for (int ki = 0; ki < K_HEAD_DIM; ++ki) {
            key[ki] = pending ? staged[N * SF + b * KF + (t * K_HEAD_NUM + hk) * K_HEAD_DIM + ki] : convert_float(k[INPUT1_GET_INDEX(b,t,hk,ki)]) * key_scale;
            query[ki] = pending ? 0.0f : convert_float(q[INPUT0_GET_INDEX(b,t,hq,ki)]) * query_scale;
            const int gi = GATE_DIM == 1 ? 0 : ki;
            const float gate = pending ? staged[N * (SF + KF + VF) + b * GF + (t * V_HEAD_NUM + h) * GATE_DIM + gi] : convert_float(g[INPUT4_GET_INDEX(b,t,h,gi)]);
            decay[ki] = exp(gate);
            const int key_owner = GROUPED_HEADS ? hk * (V_HEAD_NUM / K_HEAD_NUM) : hk;
            if (!pending && t < P && block == 0 && h == key_owner)
                key_bank[OUTPUT2_GET_INDEX(0,0,destination,(t * K_HEAD_NUM + hk) * K_HEAD_DIM + ki)] = key[ki];
            if (!pending && t < P && block == 0 && (GATE_DIM != 1 || ki == 0))
                gate_bank[OUTPUT4_GET_INDEX(0,0,destination,(t * V_HEAD_NUM + h) * GATE_DIM + gi)] = gate;
        }
        const float beta_value = pending ? staged[N * (SF + KF + VF) + b * GF + ((P + t) * V_HEAD_NUM + h) * GATE_DIM] : convert_float(beta[INPUT5_GET_INDEX(b,t,h,0)]);
        if (!pending && t < P && block == 0) gate_bank[OUTPUT4_GET_INDEX(0,0,destination,((P + t) * V_HEAD_NUM + h) * GATE_DIM)] = beta_value;
        for (int column = 0; column < V_BLOCK_SIZE && first_v + column < V_HEAD_DIM; ++column) {
            const int vi = first_v + column;
            const float value = pending ? staged[N * (SF + KF) + b * VF + (t * V_HEAD_NUM + h) * V_HEAD_DIM + vi] : convert_float(v[INPUT2_GET_INDEX(b,t,h,vi)]);
            float dot = 0.0f;
            for (int ki = 0; ki < K_HEAD_DIM; ++ki) { state[column][ki] *= decay[ki]; dot = fma(state[column][ki],key[ki],dot); }
            const float delta = (value - dot) * beta_value;
            float result = 0.0f;
            for (int ki = 0; ki < K_HEAD_DIM; ++ki) { state[column][ki] = fma(key[ki],delta,state[column][ki]); if (!pending) result = fma(state[column][ki],query[ki],result); }
            if (!pending) {
                output[OUTPUT_GET_INDEX(b,t,h,vi)] = TO_OUTPUT_TYPE(result);
                if (t < P) value_bank[OUTPUT3_GET_INDEX(0,0,destination,(t * V_HEAD_NUM + h) * V_HEAD_DIM + vi)] = value;
            }
        }
    }
    if (!keep)
        for (int column = 0; column < V_BLOCK_SIZE && first_v + column < V_HEAD_DIM; ++column)
            for (int ki = 0; ki < K_HEAD_DIM; ++ki)
                state_bank[OUTPUT1_GET_INDEX(0,0,destination,(h * V_HEAD_DIM + first_v + column) * K_HEAD_DIM + ki)] = state[column][ki];
#endif
#endif
}
