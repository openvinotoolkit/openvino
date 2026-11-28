// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "transformations/pattern_blocks/pattern_anchor.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass::pattern {

/**
 * @brief GPT-NeoX-style rotate-half RoPE pattern with named pattern-node handles.
 *
 * The supplied single-output input is part of the enclosing pattern and should enforce the requested rank.
 * The cosine branch is intentionally matched with two independent inputs; its relationship
 * to the RoPE input must be checked by the consumer.
 * The half-dimension symbol is shared with the enclosing matcher. Use different names
 * when independent instances must accept different rotary dimensions.
 */
struct TRANSFORMATIONS_API GPTNeoXRoPEPattern {
    struct Anchors {
        PatternAnchor input;
        PatternAnchor cos_input_1;
        PatternAnchor cos_input_2;
        PatternAnchor sin;
        PatternAnchor minus_one_convert;
        PatternAnchor negated_half;
        PatternAnchor rotated_input;
        PatternAnchor mul_cos;
        PatternAnchor mul_sin;
        PatternAnchor root;

        std::array<const PatternAnchor*, 10> all() const&;
        std::array<const PatternAnchor*, 10> all() const&& = delete;
    };

    GPTNeoXRoPEPattern(const std::shared_ptr<Node>& input, int rank, std::string half_ndims_symbol = "half_ndims");

    Anchors anchors;
    std::string half_ndims_symbol;
};

}  // namespace ov::pass::pattern
