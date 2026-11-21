// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps the input of a gated MLP down projection inside the FP16 range.
 *
 * In FC(down)(Multiply(Swish(FC(gate)), FC(up))) the product is stored in FP16. A single element
 * can exceed the FP16 range while the rest of the tensor stays far below it, and the down
 * projection then turns the whole row into NaN. The pass divides the weight decompression scale
 * of the up projection by a power of two and multiplies that of the down projection by the same
 * factor, so no node is added and the output is unchanged. The up branch may pass through
 * Reshape, Squeeze, Unsqueeze, Convert and Add with a constant (scaled too), and the bias of the
 * up projection is scaled as well. For a gate/up pair merged by FullyConnectedHorizontalFusion
 * only the up channels are scaled. Scaling by a power of two is exact for normal FP16 values, so
 * the pass is skipped when a scaled FP16 constant would leave that range.
 */
class ScaleDownGatedMLPInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScaleDownGatedMLPInput");
    explicit ScaleDownGatedMLPInput(float scale_factor = 8.F);
};

}  // namespace ov::intel_gpu
