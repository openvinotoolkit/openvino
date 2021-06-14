// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

// @brief Splits convolution with large input buffer
class SplitConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolution();
};

// @brief Splits convolution with large input buffer, move add with bias to each convolution before concat
class SplitConvolutionWithBias : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolutionWithBias();
};

/* @brief Splits convolution with large input buffer,
 * move add with bias and/or fake quantize to each convolution before concat
 */
class SplitConvolutionWithFq : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolutionWithFq();
};

} // namespace GNAPluginNS