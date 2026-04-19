// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "moe_gemm_base.hpp"

namespace ov::intel_gpu::ocl {

#define MOE_INTERNAL_BUFFER_TOPK_IDX                       0   // topk_idx
#define MOE_INTERNAL_BUFFER_TOPK_WEIGHTS                   1   // topk_weights
#define MOE_INTERNAL_BUFFER_UP_OUTPUT                      2   // up output
#define MOE_INTERNAL_BUFFER_DOWN_OUTPUT                    3   // down output
#define MOE_INTERNAL_BUFFER_GATE_UP_INPUT                  4   // gather input tensor
#define MOE_INTERNAL_BUFFER_ROUTING_WEIGHTS                5   // routing_weights
#define MOE_INTERNAL_BUFFER_GATE_OUTPUT                    6   // gate output
#define MOE_INTERNAL_BUFFER_EXPERT_MASK_BATCH              7   // expert_mask_batch
#define MOE_INTERNAL_BUFFER_EXPERT_MASK_TOPK               8   // expert_mask_topk
#define MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS           9   // experts_ids for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT  10  // token start offset idx (input gather tokens) for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT 11  // token len (input gather tokens) for each activated expert
#define MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT           12  // token idx per expert
#define MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM         13  // num_actual_used_experts

#define ENABLE_MOE_MICRO_GEMM_POST_PROC_SILU_MUL 1

enum class MoE3GemmMicroKernelType : uint8_t { MLP_GATE = 0, MLP_UP = 1, MLP_DOWN = 2 };

enum class MOE3GemmInputIndex : uint8_t {
    HIDDEN_STATES = 0,
    ROUTING_WEIGHTS = 1,
    WEIGHT_0 = 2,
    SCALE_0 = 3,
    ZP_0 = 4,
    WEIGHT_1 = 5,
    SCALE_1 = 6,
    ZP_1 = 7,
    WEIGHT_2 = 8,
    SCALE_2 = 9,
    ZP_2 = 10,
    // Sigmoid routing (optional, always at index 11-12)
    // For SOFTMAX routing without shared expert these are absent.
    // For SOFTMAX routing with shared expert, dummy placeholders fill these slots.
    ROUTING_BIAS = 11,
    ROUTING_EPS = 12,
    // Shared expert inputs (optional, when num_shared_expert > 0)
    // Always start at index 13 regardless of routing type.
    SHARED_GATE_WEIGHT = 13,
    SHARED_GATE_SCALE = 14,
    SHARED_GATE_ZP = 15,
    SHARED_UP_WEIGHT = 16,
    SHARED_UP_SCALE = 17,
    SHARED_UP_ZP = 18,
    SHARED_DOWN_WEIGHT = 19,
    SHARED_DOWN_SCALE = 20,
    SHARED_DOWN_ZP = 21,
    SHARED_GATE_GATE_WEIGHT = 22
};

struct moe_3gemm_config {
    int32_t weight_group_size = -1;
    bool has_batch_dim = false;  // 0 - pa, 1 - non-pa
};

struct MoE3GemmRuntimeParams : public MoEGemmRuntimeParams {};

}  // namespace ov::intel_gpu::ocl