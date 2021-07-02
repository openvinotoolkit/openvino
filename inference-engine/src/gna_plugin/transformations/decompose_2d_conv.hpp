// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
* @brief Decomopose a 2D convolution, wrapped with transposes,
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
class Decompose2DConv : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConv();
};

/**
* @brief Decomopose a 2D convolution with bias, wrapped with transposes,
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
class Decompose2DConvWithBias : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvWithBias();
};

/**
* @brief Decomopose a 2D convolution with bias and an activation function,
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
class Decompose2DConvWithBiasAF : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvWithBiasAF();
};

/**
* @brief Decomopose a 2D convolution with bias and max pooling,
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
class Decompose2DConvWithBiasMaxPool : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvWithBiasMaxPool();
};

/**
* @brief Decomopose a 2D convolution with bias, max pooling and activation function
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
class Decompose2DConvWithBiasMaxPoolAF : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvWithBiasMaxPoolAF();
};

/**
* @brief Decomopose a 2D convolution wrapped with transposes, with bias after trailing transpose,
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
class Decompose2DConvTransposedWithBias : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvTransposedWithBias();
};

/**
* @brief Decomopose a 2D convolution wrapped with transposes, with bias
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
class Decompose2DConvTransposedWithBiasAF : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Decompose2DConvTransposedWithBiasAF();
};

} // namespace GNAPluginNS
