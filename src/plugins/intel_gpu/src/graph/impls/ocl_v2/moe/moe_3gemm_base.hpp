// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "moe_gemm_base.hpp"

// #define ENABLE_ONEDNN_FOR_GPU

namespace ov::intel_gpu::ocl {

//  mlp_gate: 0
//  mlp_up: 1
//  mlp_down: 2

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
    ZP_2 = 10
};

struct moe_3gemm_config {
    int32_t weight_group_size = -1;
    bool has_batch_dim = false;  // 0 - pa, 1 - non-pa
};

struct MoE3GemmRuntimeParams : public MoEGemmRuntimeParams {
};

}  // namespace ov::intel_gpu::ocl