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
    GPTNeoXRoPEPattern(const std::shared_ptr<Node>& input, int rank, std::string half_ndims_symbol = "half_ndims");

    std::array<PatternAnchor, 10> anchors() const;

    std::shared_ptr<Node> input;
    std::string half_ndims_symbol;
    std::shared_ptr<Node> cos_input_1;
    std::shared_ptr<Node> cos_input_2;
    std::shared_ptr<Node> sin;
    std::shared_ptr<Node> minus_one_convert;
    std::shared_ptr<Node> negated_half;
    std::shared_ptr<Node> rotated_input;
    std::shared_ptr<Node> mul_cos;
    std::shared_ptr<Node> mul_sin;
    std::shared_ptr<Node> root;
};

}  // namespace ov::pass::pattern
