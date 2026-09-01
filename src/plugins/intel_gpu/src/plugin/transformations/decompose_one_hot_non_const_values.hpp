// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Decomposes OneHot (v1/v16) whose on_value/off_value are not Constants.
 *
 * The cldnn one_hot primitive bakes on/off into the OpenCL kernel as jit constants,
 * so it can only handle compile time values.
 * ONNX exports of `Tensor.repeat(...)` build the repeats vector with a OneHot,
 * and its on_value comes from a ShapeOf chain, which is a genuine runtime value.
 *
 * Such a OneHot is rewritten as a boolean mask plus a Select:
 *     OneHot(idx, depth, on, off)  ->  Select(OneHot(idx, depth, true, false), on, off)
 *
 * on_value/off_value are scalars by the op specification,
 * so the Select output shape and type match the original ones.
 * When indices and depth are constants the mask is folded away and only the Select remains,
 * which already has a CPU implementation for shape_of subgraphs.
 */
class DecomposeOneHotNonConstValues : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DecomposeOneHotNonConstValuesGPU");
    DecomposeOneHotNonConstValues();
};

}  // namespace ov::intel_gpu
