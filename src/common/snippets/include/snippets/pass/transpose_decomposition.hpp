// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface TransposeDecomposition
 * @brief Decompose Transpose to Load + Store wrapped in several loops.
 * @ingroup snippets
 */
class TransposeDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TransposeDecomposition");
    TransposeDecomposition();

    static bool is_supported_transpose(const Output<Node>& transpose_out);
    static bool is_supported_transpose_order(const std::vector<int32_t>& order);
};

}  // namespace ov::snippets::pass
