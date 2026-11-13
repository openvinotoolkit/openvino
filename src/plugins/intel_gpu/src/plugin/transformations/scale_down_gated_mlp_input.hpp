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
 * projection then turns the whole row into NaN. The pass scales the activated gate down by
 * a power of two and the down projection output up by the same factor (after the Convert that
 * keeps it in FP32, if there is one). Scaling by a power of two is exact for normal FP16 values.
 */
class ScaleDownGatedMLPInput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ScaleDownGatedMLPInput");
    explicit ScaleDownGatedMLPInput(float scale_factor = 8.f);
};

}  // namespace ov::intel_gpu
