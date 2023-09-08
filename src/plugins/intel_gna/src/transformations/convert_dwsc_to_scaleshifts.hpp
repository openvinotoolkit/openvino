// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Convert a depthwise separable convolution (represented by a GroupConvolution) to a set of ScaleShift layers
 * (MatMul + Add) Additionally supported are bias and fake quantize layers.
 */
class ConvertDWSCToScaleShifts : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDWSCToScaleShifts", "0");
    ConvertDWSCToScaleShifts();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
