// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Convert a padded convolution with bias, max pooling and activation function
 * wrapped with transposes, to a valid convolution with padding added before the leading transpose:
 *
 *                                                Padding
 *                                                   |
 *   Transpose (NHWC -> NCHW)               Transpose (NHWC -> NCHW)
 *              |                                      |
 *   Convolution with padding               Convolution with padding
 *              |                                      |
 *   Broadcast Bias (optional)              Broadcast Bias (optional)
 *              |                                      |
 *    Max Pooling (optional)                 Max Pooling (optional)
 *              |                                      |
 * Activation Function (optional)       Activation Function (optional)
 *              |                                      |
 *   Transpose (NCHW -> NHWC)               Transpose (NCHW -> NHWC)
 *
 */
class ConvertPadded2ValidConv : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertPadded2ValidConv();
};

/**
 * @brief Convert a padded convolution wrapped with transposes, with bias
 * and activation function after trailing transpose, to a valid convolution with padding added before the leading transpose:
 *
 *                                                   Padding
 *                                                      |
 *      Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *                 |                                    |
 *      Convolution with padding             Convolution with padding
 *                 |                                    |
 *      Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *                 |                                    |
 *           Broadcast Bias                       Broadcast Bias
 *                 |                                    |
 *   Activation Function (optional)       Activation Function (optional)
 *
 */
class ConvertPaddedTransposed2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedTransposed2ValidConv();
};

} // namespace GNAPluginNS
