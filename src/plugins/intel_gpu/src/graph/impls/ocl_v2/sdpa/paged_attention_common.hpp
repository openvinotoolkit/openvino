// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "common_utils/kernel_generator_base.hpp"

namespace ov::intel_gpu::ocl {
namespace {
enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t max_context_len;
    size_t paged_attention_aligned_seq_len;
    size_t sdpa_opt_seq_len_partition_size;

    size_t paged_attention_snap_kv_tokens;
    bool use_micro_sdpa = false;
    bool use_gqa_kernel = false;
    bool use_cm_kernel = false;
    size_t query_block_size = 16;
};

}  // namespace
}  // namespace ov::intel_gpu::ocl
