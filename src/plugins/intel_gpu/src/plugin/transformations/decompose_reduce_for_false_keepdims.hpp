// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_gpu {

/**
 * @brief Add Reshape to modify output of Reduce and modify keep_dims to true : reduce-reshape
 *        A clDNN Reduce reorders un-reduced axes of its output tensor to b-f and spatial order when keep_dims is false.
 *        oneDNN reduction does not allow this. And clDNN execution shows a huge perf drop for blocked formats.
 */
class DecomposeReduceForFalseKeepDims : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DecomposeReduceForFalseKeepDims");

    // Decompose reduce if keep_dims is false and it reduces batch and spatial axes
    DecomposeReduceForFalseKeepDims();

    // Returns true if reduction axes includes one of blocked axis and all spatial axes
    bool need_transformation_for_reordered_axes(std::vector<int64_t> reduce_axes, size_t num_dim, size_t num_spatial);
};

}  // namespace ov::intel_gpu
