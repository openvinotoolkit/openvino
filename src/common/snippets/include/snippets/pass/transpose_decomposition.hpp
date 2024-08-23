// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TransposeDecomposition
 * @brief Decompose Transpose to Load + Store wrapped in several loops.
 * @ingroup snippets
 */
class TransposeDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeDecomposition", "0");
    TransposeDecomposition();

    static bool is_supported_transpose(const Output<Node>& transpose_out);
    static bool is_supported_transpose_order(const std::vector<int32_t>& order);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
