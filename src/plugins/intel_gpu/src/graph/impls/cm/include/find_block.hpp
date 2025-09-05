/*
 * Copyright (c) 2020-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>
#include <cm/cmtl.h>

#include "sort.hpp"

#define MYMIN(x, y) ((x) < (y) ? (x) : (y))

// kq_max_wg:          [b, hq, n_groups, q_stride_pad]
// kq_exp_partial_sum: [b, hq, q_stride_pad, k_block_pad]
// kq_sum:             [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
CM_INLINE void find(uint slm, int m_block, svmptr_t kq_max_wg, svmptr_t kq_exp_partial_sum, svmptr_t block_mask, uint q_stride, uint q_stride_pad, uint k_block_pad, float thresh, uint causal_start_index
#if DEBUG_ACC == 1
    , svmptr_t kq_sum
#endif
) {
    constexpr int SG_SIZE = 16;
#ifndef BLOCK_SG_M
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

    const int TOKEN_IN_BLOCK = (BLOCK_SIZE / STRIDE);
    int m = m_block * TOKEN_IN_BLOCK;
    vector<half, TOKEN_IN_BLOCK> max_m;

    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    kq_exp_partial_sum += m * k_block_pad * (int)sizeof(half);
    kq_max_wg += m * (int)sizeof(half);
    constexpr half log2e = 1.4426950408889634f;
    matrix<float, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> sum_m = 0;
    matrix<half, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> data;
    int m_start = MYMIN(m, q_stride);
    int m_end = MYMIN(m_start + TOKEN_SHARE_MAX, q_stride);
    int valid_m = m_end - m_start;
    if (valid_m == 0) return;
    lsc::block_2d_desc<half, 1, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> desc_sum{ kq_exp_partial_sum, (uint)valid_m - 1, (uint)(k_block_pad * sizeof(half) - 1), (uint)(k_block_pad * sizeof(half) - 1),
        0, 0 };
    {
        // find max: (k_block_pad / TOKEN_SHARE_MAX) * q_stride_pad
        max_m = half{-60000};

        for (int idx = 0; idx < k_block_pad / TOKEN_SHARE_MAX; idx++) {
            vector<half, TOKEN_IN_BLOCK> max_m_in_group;
            max_m_in_group.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / 2>((int*)kq_max_wg, q_stride_pad * idx * (int)sizeof(half));
            max_m = cm_max<half>(max_m, max_m_in_group);
        }
    }
    // compensation: val*exp(local - global)
    desc_sum.set_block_x(0);
    for (int j = 0, idx = 0; j < k_block_pad; j += TOKEN_SHARE_MAX, idx++) {
        vector<half, TOKEN_IN_BLOCK> max_m_in_group;
        max_m_in_group.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / 2>((int*)kq_max_wg, q_stride_pad * idx * (int)sizeof(half));
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<half>(), desc_sum);
        for (int i = 0; i < TOKEN_IN_BLOCK; i++) {
            if (i < valid_m) {
                data.row(i) *= cm_exp((max_m_in_group[i] - max_m[i]) * log2e);
                sum_m.row(i) += data.row(i);
            }
        }
        cm_store(desc_sum, data.format<half>());
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
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
    desc_sum.set_block_x(0);
#if DEBUG_ACC == 1
    kq_sum += m_block * k_block_pad * (int)sizeof(half);
#endif
    for (int j = 0; j < k_block_pad; j += TOKEN_SHARE_MAX) {
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<half>(), desc_sum);
        data.row(0) *= inv_sum_v[0];
        for (int i = 1; i < TOKEN_IN_BLOCK; i++) {
            data.row(0) += data.row(i) * inv_sum_v[i];
        }
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
        sum_m_after_add += data.row(0);
        cm_ptr_store<int, TOKEN_SHARE_MAX / 2>((int*)kq_exp_partial_sum, j * (int)sizeof(half), data.row(0).format<int>());
#if DEBUG_ACC == 1
        cm_ptr_store<int, TOKEN_SHARE_MAX / 2>((int*)kq_sum, j * (int)sizeof(half), data.row(0).format<int>());
#endif
    }
    auto thresh_act = cm_sum<float>(sum_m_after_add) * thresh;

    // content of 8(aka stride) lines:
    // line 0: score
    // line 1: sorted value
    // line 3: sorted index
    // line 5: sorted tmp
    // line 6: accumalative score
    block_mask += m_block * k_block_pad;
    auto score        = kq_exp_partial_sum + 0 * k_block_pad * (int)sizeof(half);
    auto sorted_value = kq_exp_partial_sum + 1 * k_block_pad * (int)sizeof(half);
    auto sorted_index = kq_exp_partial_sum + 3 * k_block_pad * (int)sizeof(half);
    auto sorted_tmp   = kq_exp_partial_sum + 5 * k_block_pad * (int)sizeof(half);
    auto acc_score    = kq_exp_partial_sum + 6 * k_block_pad * (int)sizeof(half);

#if IS_CAUSAL == 1
    auto score_p = (half*)score;
    half s_0 = score_p[0];
    half s_causal = score_p[causal_start_index + m_block];
    half s_sum = s_0;
    if (causal_start_index + m_block) s_sum += s_causal;
    score_p[0] = -1;
    score_p[causal_start_index + m_block] = -1;
    sort<half>(slm, score, sorted_value + 2 * sizeof(half), sorted_index + 2 * sizeof(half), sorted_tmp, k_block_pad);
    uchar* block_mask_p = (uchar*)block_mask;
    auto sorted_value_p = (half*)sorted_value;
    auto sorted_index_p = (ushort*)sorted_index;
    auto acc_score_p = (half*)acc_score;
    sorted_value_p[0] = 0;
    sorted_value_p[1] = s_sum;
    block_mask_p[0] = 1;
    block_mask_p[causal_start_index + m_block] = 1;
    float sum_cur = s_sum;
#if DEBUG_ACC == 1
    acc_score_p[0] = 0;
    acc_score_p[1] = 0;
#endif
    for (int j = 2; j < k_block_pad - 2; j++) {
#if DEBUG_ACC == 1
        acc_score_p[j] = sum_cur;
#endif
        if (sum_cur < thresh_act) {
            block_mask_p[sorted_index_p[j]] = 1;
        } else {
#if DEBUG_ACC != 1
            break;
#endif
        }
        sum_cur += sorted_value_p[j];
    }

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
    for (int j = 0; j < k_block_pad - 1; j++) {
        sum_cur += sorted_value_p[j];
#if DEBUG_ACC == 1
        acc_score_p[j + 1] = sum_cur;
#endif
        if (sum_cur < thresh_act) {
            block_mask_p[sorted_index_p[j]] = 1;
        } else {
            block_mask_p[sorted_index_p[j]] = 1;
#if DEBUG_ACC != 1
            break;
#endif
        }
    }
#endif
}
