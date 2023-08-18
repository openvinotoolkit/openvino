// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Change all trivial concatenations (concatenation where output buffer is a buffer made by appending input
 * buffers) by reshaping its inputs to 1 x total_input_size and its output to 1 x total_concat_size and changing the
 * axis to 1. For example, let's say all inputs have the same shape equal to: {1, 1, 5, 3} then for axis 0, 1, 2 the
 * concat can be flattened and inputs will be reshaped to 1, 15. For shape {2, 1, 5, 3}, only concat with axis 0 can be
 * flattened; in such case inputs will be reshaped to 1, 30.
 */
class FlattenTrivialConcat : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FlattenTrivialConcat", "0");
    FlattenTrivialConcat();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
