// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

// @brief Splits convolution with large input buffer
class SplitConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolution", "0");
    SplitConvolution();
};

// @brief Splits convolution with large input buffer, move add with bias to each convolution before concat
class SplitConvolutionWithBias : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolutionWithBias", "0");
    SplitConvolutionWithBias();
};

/* @brief Splits convolution with large input buffer,
 * move add with bias and/or fake quantize to each convolution before concat
 */
class SplitConvolutionWithFq : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolutionWithFq", "0");
    SplitConvolutionWithFq();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
