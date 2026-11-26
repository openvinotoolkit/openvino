// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "executor_config.hpp"
#include "memory_arguments.hpp"

namespace ov::intel_cpu {

// Named memory arguments of GroupedMatMul. They alias the generic ARG_* slots, so the shared
// inner-product helpers (which address scales / zero points by ARG_SRC_3 / ARG_SRC_4) stay compatible.
inline constexpr int ARG_GMM_DATA = ARG_SRC;      // activations [T, K] or [G, M, K]
inline constexpr int ARG_GMM_WEIGHTS = ARG_WEI;   // weights [G, K, N]
inline constexpr int ARG_GMM_OFFSETS = ARG_SRC_1; // optional group offsets (2D x 3D form only)
inline constexpr int ARG_GMM_SCALES = ARG_SRC_3;  // decompression scales (compressed weights only)
inline constexpr int ARG_GMM_ZP = ARG_SRC_4;      // decompression zero points (optional)

struct GroupedMatMulAttrs {
    uint64_t dynamic_quantization_group_size = 0;
};

using GroupedMatMulConfig = executor::Config<GroupedMatMulAttrs>;

}  // namespace ov::intel_cpu
