// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cm/cm.h>
#include <cm/cmtl.h>

#include "sort.hpp"
#define MYMIN(x, y) ((x) < (y) ? (x) : (y))

#define MYCONCAT(x, y) x ## y
#define IS_float 1
#define IS_half 2
#define CUR_TYPE_(a) MYCONCAT(IS_, a)
#define CUR_TYPE CUR_TYPE_(SOFTMAX_TYPE)

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<SOFTMAX_TYPE, M, N> out,
                          svmptr_t base, uint offset, uint pitch, uint valid_m) {
    #pragma unroll
    for (int i = 0; i < out.n_rows(); i++) {
        if (i < (int)valid_m) {
            out.row(i).format<uint>() =
                cm_ptr_load<uint, N, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(
                    (uint*)base, offset + i * pitch);
        } else {
            out.row(i) = SOFTMAX_TYPE(0);
        }
    }
}

template <int M, int N>
CM_INLINE void cm_store_2d(matrix_ref<SOFTMAX_TYPE, M, N> out, svmptr_t base, uint offset, uint pitch,  uint valid_m) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        if(offset + i * pitch < valid_m*pitch)
            cm_ptr_store<uint, N>((uint*)base, offset + i * pitch, out.row(i).format<uint>());
    }
}

// kq_max_wg:          [b, hq, n_groups, q_stride_pad]
// kq_exp_partial_sum: [b, hq, q_stride_pad, k_block_pad]
// kq_sum:             [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
CM_INLINE void find(uint slm, int m_block,
    svmptr_t kq_max_wg,
    //#ifdef CM_HAS_LSC_UNTYPED_2D
    svmptr_t kq_exp_partial_sum,
    // #else
    // SurfaceIndex kq_exp_partial_sum [[type("buffer_t")]],
    // #endif
    svmptr_t block_mask, uint q_len, uint q_stride, uint q_stride_pad, uint k_block_pad, float thresh, uint causal_start_index
#if DEBUG_ACC == 1
    , svmptr_t kq_sum
#endif
) {
#ifndef BLOCK_SHARE_MAX
    #define BLOCK_SG_M  64
    #define BLOCK_SG_N  32
    #define SG_M  2
    #define SG_N  4
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
    #define BLOCK_SIZE 128
    #define BLOCK_SHARE_MAX 256
#endif

    constexpr int TOKEN_IN_BLOCK = (BLOCK_SIZE / STRIDE);   // 8 -> 16
    int m = m_block * TOKEN_IN_BLOCK;
    vector<SOFTMAX_TYPE, TOKEN_IN_BLOCK> max_m;

    constexpr int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;   // 32 -> 16
    kq_exp_partial_sum += m * k_block_pad * (int)sizeof(SOFTMAX_TYPE);

    kq_max_wg += m * (int)sizeof(SOFTMAX_TYPE);
    constexpr SOFTMAX_TYPE log2e = 1.4426950408889634f;
    matrix<float, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> sum_m = 0;
    matrix<SOFTMAX_TYPE, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> data;   // (8, 32) -> (16, 16)
    int m_start = MYMIN(m, q_stride);
    int m_end = MYMIN(m_start + TOKEN_SHARE_MAX, q_stride);
    int valid_m = m_end - m_start;
    block_mask += m_block * k_block_pad;
    if (valid_m == 0) {
        // case for tails: q is not inside mask, aka q % BLOCK_SIZE < STRIDE
        if (m * STRIDE < q_len) {
            vector<uchar, TOKEN_SHARE_MAX> one = 1;
            for (int j = 0; j < k_block_pad; j += TOKEN_SHARE_MAX) {
                cm_ptr_store<int, TOKEN_SHARE_MAX / 4>((int*)block_mask, j, one.format<int>());
            }
        }
        return;
    }
    #ifdef CM_HAS_LSC_UNTYPED_2D
    #if BLOCK_SIZE == 128
    lsc::block_2d_desc<SOFTMAX_TYPE, 1, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX / (sizeof(SOFTMAX_TYPE) / sizeof(half))> desc_sum{ kq_exp_partial_sum, (uint)valid_m - 1, (uint)(k_block_pad * sizeof(SOFTMAX_TYPE) - 1), (uint)(k_block_pad * sizeof(SOFTMAX_TYPE) - 1),
        0, 0 };
    #else
    lsc::block_2d_desc<SOFTMAX_TYPE, 1, TOKEN_IN_BLOCK / 2, TOKEN_SHARE_MAX> desc_sum{ kq_exp_partial_sum, (uint)valid_m - 1, (uint)(k_block_pad * sizeof(SOFTMAX_TYPE) - 1), (uint)(k_block_pad * sizeof(SOFTMAX_TYPE) - 1),
        0, 0 };
    #endif
    #else
    const uint pitch_sum = k_block_pad * sizeof(SOFTMAX_TYPE);
    uint off_sum = 0;// m * k_block_pad * (int)sizeof(SOFTMAX_TYPE);
    #endif
    {
        // find max: (k_block_pad / TOKEN_SHARE_MAX) * q_stride_pad
        max_m = SOFTMAX_TYPE{-60000};

        for (int idx = 0; idx < k_block_pad / TOKEN_SHARE_MAX; idx++) {
            vector<SOFTMAX_TYPE, TOKEN_IN_BLOCK> max_m_in_group;
            max_m_in_group.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / (sizeof(int) / sizeof(SOFTMAX_TYPE))>((int*)kq_max_wg, q_stride_pad * idx * (int)sizeof(SOFTMAX_TYPE));
            max_m = cm_max<SOFTMAX_TYPE>(max_m, max_m_in_group);
        }
    }
    // compensation: val*exp(local - global)
    #ifdef CM_HAS_LSC_UNTYPED_2D
    desc_sum.set_block_x(0);
    #else
    off_sum = 0;
    #endif
    for (int j = 0, idx = 0; j < k_block_pad; j += TOKEN_SHARE_MAX, idx++) {
        vector<SOFTMAX_TYPE, TOKEN_IN_BLOCK> max_m_in_group;
        max_m_in_group.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / (sizeof(int) / sizeof(SOFTMAX_TYPE))>((int*)kq_max_wg, q_stride_pad * idx * (int)sizeof(SOFTMAX_TYPE));
#ifdef CM_HAS_LSC_UNTYPED_2D
#if CUR_TYPE == IS_float && BLOCK_SIZE == 128
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 0>(data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, 0).format<SOFTMAX_TYPE>(), desc_sum);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 16, 0>(data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, TOKEN_SHARE_MAX / 2).format<SOFTMAX_TYPE>(), desc_sum);
#elif CUR_TYPE == IS_float && BLOCK_SIZE == 256
        // 2x(8, 16) -> (16, 16)
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 0>(data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(0, 0).format<SOFTMAX_TYPE>(), desc_sum);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 8>(data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(TOKEN_IN_BLOCK / 2, 0).format<SOFTMAX_TYPE>(), desc_sum);
#else
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<SOFTMAX_TYPE>(), desc_sum);
#endif
#else
//xe1
        cm_load_2d(data, kq_exp_partial_sum, off_sum, pitch_sum, valid_m);
#endif
        for (int i = 0; i < TOKEN_IN_BLOCK; i++) {
            if (i < valid_m) {
                data.row(i) *= cm_exp((max_m_in_group[i] - max_m[i]) * log2e);
                sum_m.row(i) += data.row(i);
            }
        }
#ifdef CM_HAS_LSC_UNTYPED_2D
#if CUR_TYPE == IS_float && BLOCK_SIZE == 128
        cm_store<CacheHint::Uncached, CacheHint::WriteBack,  0, 0>(desc_sum, data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, 0).format<SOFTMAX_TYPE>());
        cm_store<CacheHint::Uncached, CacheHint::WriteBack, 16, 0>(desc_sum, data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, TOKEN_SHARE_MAX / 2).format<SOFTMAX_TYPE>());
#elif CUR_TYPE == IS_float && BLOCK_SIZE == 256
        cm_store<CacheHint::Uncached, CacheHint::WriteBack,  0, 0>(desc_sum, data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(0, 0).format<SOFTMAX_TYPE>());
        cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8>(desc_sum, data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(TOKEN_IN_BLOCK / 2,  0).format<SOFTMAX_TYPE>());
#else
        cm_store(desc_sum, data.format<SOFTMAX_TYPE>());
#endif
#else
//xe1
        cm_store_2d(data, kq_exp_partial_sum, off_sum, pitch_sum, valid_m);
#endif
#ifdef CM_HAS_LSC_UNTYPED_2D
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
#else
        off_sum += TOKEN_SHARE_MAX * sizeof(SOFTMAX_TYPE);
#endif
    }

    // exp/sum
    vector<float, TOKEN_IN_BLOCK> inv_sum_v;
    for (int i = 0; i < TOKEN_IN_BLOCK; i++) {
        if (i < valid_m)
            inv_sum_v[i] = 1.0f / cm_sum<float>(sum_m.row(i));
        else
            inv_sum_v[i] = 0;
    }
    // compensation: sum(val*inv_sum_v)
    vector<float, TOKEN_SHARE_MAX> sum_m_after_add = 0;
#ifdef CM_HAS_LSC_UNTYPED_2D
    desc_sum.set_block_x(0);
#else
    off_sum = 0;
#endif

#if DEBUG_ACC == 1
    kq_sum += m_block * k_block_pad * (int)sizeof(SOFTMAX_TYPE);
#endif
    vector<uchar, TOKEN_SHARE_MAX> zero = 0;
    for (int j = 0; j < k_block_pad; j += TOKEN_SHARE_MAX) {
#ifdef CM_HAS_LSC_UNTYPED_2D
#if CUR_TYPE == IS_float && BLOCK_SIZE == 128
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 0>(data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, 0).format<SOFTMAX_TYPE>(), desc_sum);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 16, 0>(data.select<TOKEN_IN_BLOCK, 1, TOKEN_SHARE_MAX / 2, 1>(0, TOKEN_SHARE_MAX / 2).format<SOFTMAX_TYPE>(), desc_sum);
#elif CUR_TYPE == IS_float && BLOCK_SIZE == 256
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 0>(data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(0, 0).format<SOFTMAX_TYPE>(), desc_sum);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached,  0, 8>(data.select<TOKEN_IN_BLOCK / 2, 1, TOKEN_SHARE_MAX, 1>(TOKEN_IN_BLOCK / 2, 0).format<SOFTMAX_TYPE>(), desc_sum);
#else
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<SOFTMAX_TYPE>(), desc_sum);
#endif
#else

//xe1
        cm_load_2d(data, kq_exp_partial_sum, off_sum, pitch_sum, valid_m);

#endif
        data.row(0) *= inv_sum_v[0];
        for (int i = 1; i < TOKEN_IN_BLOCK; i++) {
            data.row(0) += data.row(i) * inv_sum_v[i];
        }

#ifdef CM_HAS_LSC_UNTYPED_2D
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
#else
        off_sum += TOKEN_SHARE_MAX * sizeof(SOFTMAX_TYPE);
#endif
        // The paper reference keeps this accumulation in half, but Xe1 long
        // sparse prefill over-masks if we follow that literally here. Keep the
        // runtime scores in float for thresholding and causal block selection.
        sum_m_after_add += data.row(0);
        cm_ptr_store<int, TOKEN_SHARE_MAX>((int*)kq_exp_partial_sum, j * (int)sizeof(float), data.row(0).format<int>());
#if DEBUG_ACC == 1
        cm_ptr_store<int, TOKEN_SHARE_MAX>((int*)kq_sum, j * (int)sizeof(float), data.row(0).format<int>());
#endif
        cm_ptr_store<int, TOKEN_SHARE_MAX / 4>((int*)block_mask, j, zero.format<int>());
    }
    auto thresh_act = cm_sum<float>(sum_m_after_add) * thresh;

    // Repack the per-block scratch space by actual byte size instead of the
    // historical half-based line layout. Score/value keep SOFTMAX_TYPE storage;
    // index/tmp remain ushort arrays even when scores are float.
    const int score_bytes = k_block_pad * (int)sizeof(SOFTMAX_TYPE);
    const int sorted_value_bytes = k_block_pad * (int)sizeof(SOFTMAX_TYPE);
    const int sorted_index_bytes = k_block_pad * (int)sizeof(ushort);
    const int sorted_tmp_bytes = k_block_pad * (int)sizeof(ushort);

    auto score        = kq_exp_partial_sum;
    auto sorted_value = score + score_bytes;
    auto sorted_index = sorted_value + sorted_value_bytes;
    auto sorted_tmp   = sorted_index + sorted_index_bytes;
    auto acc_score    = sorted_tmp + sorted_tmp_bytes;

#if IS_CAUSAL == 1
    auto score_p = (float*)score;
    float s_0 = score_p[0];
    float s_causal = score_p[causal_start_index + m_block];
    float s_sum = s_0;
    if (causal_start_index + m_block) s_sum += s_causal;
    score_p[0] = -1.f;
    score_p[causal_start_index + m_block] = -1.f;
    uchar* block_mask_p = (uchar*)block_mask;
    auto sorted_value_p = (float*)sorted_value;
    auto sorted_index_p = (ushort*)sorted_index;
    auto acc_score_p = (float*)acc_score;
    // Preserve the historical debug/acc buffer contract even though the causal
    // path now performs float-based selection without calling sort<half>.
    sorted_value_p[0] = 0;
    sorted_value_p[1] = s_sum;
    sorted_index_p[0] = 0;
    sorted_index_p[1] = causal_start_index + m_block;
    block_mask_p[0] = 1;
    block_mask_p[causal_start_index + m_block] = 1;
    float sum_cur = s_sum;
#if DEBUG_ACC == 1
    acc_score_p[0] = 0;
    acc_score_p[1] = 0;
#endif
    int j;
    for (j = 2; j < k_block_pad; j++) {
#if DEBUG_ACC == 1
        acc_score_p[j] = sum_cur;
#endif
        if (sum_cur >= thresh_act) {
            break;
        }
        float best_score = -1.f;
        int best_idx = -1;
        for (int k_idx = 1; k_idx <= (int)(causal_start_index + m_block); k_idx++) {
            if (score_p[k_idx] > best_score) {
                best_score = score_p[k_idx];
                best_idx = k_idx;
            }
        }
        if (best_idx < 0) {
            break;
        }
        sorted_index_p[j] = static_cast<ushort>(best_idx);
        sorted_value_p[j] = best_score;
        block_mask_p[best_idx] = 1;
        score_p[best_idx] = -1.f;
        sum_cur += best_score;
    }
#if DEBUG_ACC == 1
    for (; j < k_block_pad; j++) {
        acc_score_p[j] = sum_cur;
    }
#endif

    // for (int j = causal_start_index + m_block + 1; j < k_block_pad; j++)
    //     block_mask_p[j] = 0;

#else
    sort<half>(slm, score, sorted_value, sorted_index, sorted_tmp, k_block_pad);
    uchar* block_mask_p = (uchar*)block_mask;
    auto sorted_value_p = (half*)sorted_value;
    auto sorted_index_p = (ushort*)sorted_index;
    auto acc_score_p = (half*)acc_score;
    block_mask_p[0] = 1;
    float sum_cur = 0;
#if DEBUG_ACC == 1
    acc_score_p[0] = 0;
#endif
    int j;
    for (j = 0; j < k_block_pad - 1; j++) {
        sum_cur += sorted_value_p[j];
#if DEBUG_ACC == 1
        acc_score_p[j + 1] = sum_cur;
#endif
        if (sum_cur < thresh_act) {
            block_mask_p[sorted_index_p[j]] = 1;
        } else {
            block_mask_p[sorted_index_p[j]] = 1;
            break;
        }
    }
#if DEBUG_ACC == 1
    for (j = j + 1; j < k_block_pad - 1; j++) {
        sum_cur += sorted_value_p[j];
        acc_score_p[j + 1] = sum_cur;
    }
#endif

#endif
}
