// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass::pattern::op {

class TRANSFORMATIONS_API CompressedWeightsBlock;

}  // namespace ov::pass::pattern::op

class ov::pass::pattern::op::CompressedWeightsBlock : public ov::pass::pattern::op::Block {
public:
    /**
     * @param supported_weights_types element types accepted for the weights constant
     * @param supported_weights_ranks ranks accepted for the final decompressed weights
     * @param allow_reshaped_decompression_constants if true, scale / zero point constants (optionally behind a
     *        Convert) may be followed by a Reshape that restores dimensions squeezed out of the constant. The
     *        Reshape nodes are exposed through the "sub_reshape" / "mul_reshape" anchors; the consumer is
     *        responsible for applying the target shape to the matched constants. Disabled by default.
     */
    CompressedWeightsBlock(const std::vector<ov::element::Type>& supported_weights_types,
                           const std::set<size_t>& supported_weights_ranks,
                           bool allow_reshaped_decompression_constants = false);
};
