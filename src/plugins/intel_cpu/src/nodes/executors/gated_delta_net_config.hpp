// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

struct GatedDeltaNetAttrs {
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6F;
    float k_l2_norm_eps = 1e-6F;
    size_t jit_v_tile = 16;
};

using GatedDeltaNetConfig = executor::Config<GatedDeltaNetAttrs>;

enum GatedDeltaNetArgId : uint8_t {
    ARG_GDN_QUERY = ARG_SRC_0,
    ARG_GDN_KEY = ARG_SRC_1,
    ARG_GDN_VALUE = ARG_SRC_2,
    ARG_GDN_STATE = ARG_SRC_3,
    ARG_GDN_GATE = ARG_SRC_4,
    ARG_GDN_BETA = ARG_SRC_5,
    ARG_GDN_OUT_ATTN = ARG_DST_0,
    ARG_GDN_OUT_STATE = ARG_DST_0 + 1,
};

}  // namespace ov::intel_cpu
