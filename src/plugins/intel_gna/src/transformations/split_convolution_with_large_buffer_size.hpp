// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

// @brief Splits convolution with large input buffer
class SplitConvolution : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolution", "0");
    SplitConvolution(size_t mem_alignment);
};

// @brief Splits convolution with large input buffer, move add with bias to each convolution before concat
class SplitConvolutionWithBias : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolutionWithBias", "0");
    SplitConvolutionWithBias(size_t mem_alignment);
};

/* @brief Splits convolution with large input buffer,
 * move add with bias and/or fake quantize to each convolution before concat
 */
class SplitConvolutionWithFq : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitConvolutionWithFq", "0");
    SplitConvolutionWithFq(size_t mem_alignment);
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
