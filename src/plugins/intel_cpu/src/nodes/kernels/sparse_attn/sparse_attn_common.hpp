// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <numeric>

namespace ov::Extensions::Cpu {

/// @brief A decode uses the brgemm path of the head pack, not per-head SIMD items, when its selection holds at least
/// this many K + V bytes (rows x (k + v head size) x element size): brgemm wins from about 2 MB, SIMD below 1 MB.
constexpr size_t kBrgemmDecodeMinBytes = 3 * 512 * 1024;

/// @brief Query heads that share both K/V rows and the selected blocks, so a work item can process them together:
/// q_head_pack_size = gcd(q_heads_per_k_head, q_heads_per_sel_head) with q_heads_per_k_head = q_head_num / k_head_num
/// and q_heads_per_sel_head = q_head_num / sel_head_num (query head h reads K head h / q_heads_per_k_head and uses
/// selection head h / q_heads_per_sel_head). sel_head_num of 1 or k_head_num gives the whole K-head group, q_head_num
/// gives 1.
inline size_t max_q_head_pack_size(size_t q_head_num, size_t k_head_num, size_t sel_head_num) {
    return std::gcd(q_head_num / k_head_num, q_head_num / sel_head_num);
}

}  // namespace ov::Extensions::Cpu
