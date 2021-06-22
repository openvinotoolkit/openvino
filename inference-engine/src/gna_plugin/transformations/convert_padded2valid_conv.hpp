// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

 /**
  * @brief Convert a padded convolution, wrapped with transposes,
  * to a valid convolution with padding added before the leading transpose:
  *
  *                                              Padding
  *                                                 |
  * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
  *            |                                    |
  * Convolution with padding             Convolution with padding
  *            |                                    |
  * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
  * 
  */
class ConvertPadded2ValidConv : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertPadded2ValidConv();
};

/**
 * @brief Convert a padded convolution with bias, wrapped with transposes,
 * to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *
 */
class ConvertPaddedWithBias2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedWithBias2ValidConv();
};

/**
 * @brief Convert a padded convolution with bias and an activation function,
 * wrapped with transposes, to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 *   Activation Function                  Activation Function
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *
 */
class ConvertPaddedWithBiasAF2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedWithBiasAF2ValidConv();
};

/**
 * @brief Convert a padded convolution with bias and max pooling,
 * wrapped with transposes, to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 *       Max Pooling                          Max Pooling
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *
 */
class ConvertPaddedWithBiasMaxPool2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedWithBiasMaxPool2ValidConv();
};

/**
 * @brief Convert a padded convolution with bias, max pooling and activation function
 * wrapped with transposes, to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 *       Max Pooling                          Max Pooling
 *            |                                    |
 *   Activation Function                  Activation Function
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *
 */
class ConvertPaddedWithBiasMaxPoolAF2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedWithBiasMaxPoolAF2ValidConv();
};

/**
 * @brief Convert a padded convolution wrapped with transposes, with bias after trailing transpose,
 * to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *
 */
class ConvertPaddedTransposedWithBias2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedTransposedWithBias2ValidConv();
};

/**
 * @brief Convert a padded convolution wrapped with transposes, with bias
 * and activation function after trailing transpose, to a valid convolution with padding added before the leading transpose:
 *
 *                                              Padding
 *                                                 |
 * Transpose (NHWC -> NCHW)             Transpose (NHWC -> NCHW)
 *            |                                    |
 * Convolution with padding             Convolution with padding
 *            |                                    |
 * Transpose (NCHW -> NHWC)             Transpose (NCHW -> NHWC)
 *            |                                    |
 *      Broadcast Bias                       Broadcast Bias
 *            |                                    |
 *   Activation Function                  Activation Function
 *
 */
class ConvertPaddedTransposedWithBiasAF2ValidConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPaddedTransposedWithBiasAF2ValidConv();
};

} // namespace GNAPluginNS
