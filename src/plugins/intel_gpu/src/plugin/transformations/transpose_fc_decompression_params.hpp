// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Stores grouped FullyConnectedCompressed decompression scale / zero point constants in the [groups, N] layout.
 *
 * The OCL and oneDNN FC kernels read grouped decompression parameters with N innermost. With transposed weights
 * [N, K] the parameters are [N, groups], so prepare_quantization adds a reorder to fbyx, which keeps a second,
 * reordered copy of each parameter in device memory. This pass transposes such constants at compile time and
 * clears transpose_b_scale / transpose_b_zp, so the parameter is used as is.
 *
 * Before:                                     After:
 *
 *   Data  Weights [N, K]  Scale [N, G]          Data  Weights [N, K]  Scale' [G, N]
 *     │        │             │                    │        │             │
 *  ┌──┴────────┴─────────────┴──┐              ┌──┴────────┴─────────────┴──┐
 *  │ FullyConnectedCompressed   │              │ FullyConnectedCompressed   │
 *  │ transpose_b_scale = true   │              │ transpose_b_scale = false  │
 *  └────────────────────────────┘              └────────────────────────────┘
 *
 * The same applies to the zero point (transpose_b_zp). A parameter is transposed only when:
 *  - the weights are 2D [N, K] with transpose_b;
 *  - it is a Constant [N, G] with G > 1 (a single group has the same memory order in both layouts);
 *  - its element type is at least 8 bits wide (sub-byte zero points keep the existing layout).
 * The new constant keeps the WeightlessCacheAttribute of the original one and records the transpose order in the
 * "gpu_weightless_transpose_order" rt_info entry, so a weightless cache load rebuilds its data from the weights file.
 */
class TransposeFCDecompressionParams : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeFCDecompressionParams");
    TransposeFCDecompressionParams();
};

}  // namespace ov::intel_gpu
