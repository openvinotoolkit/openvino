// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "moe_gemm_base.hpp"

namespace ov::intel_gpu::ocl {

#define MOE_INTERNAL_BUFFER_UP_OUTPUT                      0   // up output
#define MOE_INTERNAL_BUFFER_DOWN_OUTPUT                    1   // down output
#define MOE_INTERNAL_BUFFER_GATE_UP_INPUT                  2   // gather input tensor
#define MOE_INTERNAL_BUFFER_ROUTING_WEIGHTS                3   // routing_weights
#define MOE_INTERNAL_BUFFER_GATE_OUTPUT                    4   // gate output
#define MOE_INTERNAL_BUFFER_EXPERT_MASK_BATCH              5   // expert_mask_batch
#define MOE_INTERNAL_BUFFER_EXPERT_MASK_TOPK               6   // expert_mask_topk
#define MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS           7   // experts_ids for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT  8   // token start offset idx (input gather tokens) for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT 9   // token len (input gather tokens) for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT           10  // token idx per expert
#define MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM         11  // num_actual_used_experts
#define MOE_INTERNAL_BUFFER_GROUPED_OFFSETS                12  // int32_t cumulative end-offsets per expert for OneDNN grouped GEMM

#define ENABLE_MOE_MICRO_GEMM_POST_PROC_SILU_MUL 1

enum class MoE3GemmMicroKernelType : uint8_t { MLP_GATE = 0, MLP_UP = 1, MLP_DOWN = 2 };

enum class MOE3GemmInputIndex : uint8_t {
    HIDDEN_STATES = 0,
    TOPK_WEIGHTS = 1,
    TOPK_INDICES = 2,
    WEIGHT_0 = 3,
    SCALE_0 = 4,
    ZP_0 = 5,
    WEIGHT_1 = 6,
    SCALE_1 = 7,
    ZP_1 = 8,
    WEIGHT_2 = 9,
    SCALE_2 = 10,
    ZP_2 = 11,
    // Shared expert inputs (optional, when num_shared_expert > 0)
    // Always start at index 12 regardless of routing type.
    SHARED_GATE_WEIGHT = 12,
    SHARED_GATE_SCALE = 13,
    SHARED_GATE_ZP = 14,
    SHARED_UP_WEIGHT = 15,
    SHARED_UP_SCALE = 16,
    SHARED_UP_ZP = 17,
    SHARED_DOWN_WEIGHT = 18,
    SHARED_DOWN_SCALE = 19,
    SHARED_DOWN_ZP = 20,
    SHARED_GATE_GATE_WEIGHT = 21
};

struct moe_3gemm_config {
    int32_t weight_group_size = -1;
    bool has_batch_dim = false;  // 0 - pa, 1 - non-pa
};

struct MoE3GemmRuntimeParams : public MoEGemmRuntimeParams {};

}  // namespace ov::intel_gpu::ocl