// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/* @brief Splits convolution with large number of filters
 *
 *                                     Conv  Conv
 *  Conv with large no of filters        \    /
 *                                       Concat
 */
class SplitConvolutionFilter : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolutionFilter();
};

/* @brief Splits convolution with large number of filters,
 * moves add with bias to each convolution before concat
 *
 *                                     Conv  Conv
 *  Conv with large no of filters       |      |
 *                 |                   Bias  Bias
 *                Bias                   \    /
 *                                       Concat
 */
class SplitConvolutionFilterWithBias : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolutionFilterWithBias();
};

/* @brief Splits convolution with large number of filters,
 * moves add with bias and/or fake quantize to each convolution before concat
 *
 *                                     Conv  Conv
 *  Conv with large no of filters       |      |
 *                 |                    FQ    FQ
 *                 FQ                    \    /
 *                                       Concat
 *
 *                       OR
 *
 *                                     Conv  Conv
 *  Conv with large no of filters       |      |
 *                 |                   Bias  Bias
 *                Bias                  |      |
 *                 |                    FQ    FQ
 *                 FQ                    \    /
 *                                       Concat
 * 
 */
class SplitConvolutionFilterWithFq : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  SplitConvolutionFilterWithFq();
};

} // namespace GNAPluginNS
