// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @brief Absorb explicit, dynamically computed SAME_UPPER padding into a forward convolution.
 *
 * Recognizes spatial padding computed from ShapeOf of the unpadded input, including the
 * scalar arithmetic and padding-vector rearrangements produced by PyTorch/timm exports.
 * Supports Convolution and GroupConvolution, preserving other users of the Pad and its
 * shape subgraph. Only constant zero padding and convolutions without existing padding
 * are eligible.
 */
class TRANSFORMATIONS_API DynamicSamePaddingFusion : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicSamePaddingFusion");
    DynamicSamePaddingFusion();
};

}  // namespace ov::pass
