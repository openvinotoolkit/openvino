// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Convert a MatMul with batch size unsupported by GNA to a point-wise convolution with NHWC layout
 * with transposes around it:
 *                                      Transose (NHWC -> NCHW)
 *                                                 |
 * Matmul                               Convolution in NHWC layout
 * Input1: [A, B] B > 8     ------->    Input: [1, 1, A, B]
 * Input2: [B, C]                       Kernel: [C, B, 1, 1]
 * Output: [A, C]                       Output: [1, 1, A, C]
 *                                                  |
 *                                      Transose (NCHW -> NHWC)
 */
class ConvertMatmulToPointWiseConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMatmulToPointWiseConvolution();
};

/**
 * @brief Convert a MatMul with batch size unsupported by GNA to a point-wise convolution with NHWC layout
 * with transposes around it, moved add with bias before the last transpose:
 *                                      Transose (NHWC -> NCHW)
 *                                                 |
 * Matmul                               Convolution in NHWC layout
 * Input1: [A, B] B > 8     ------->    Input: [1, 1, A, B]
 * Input2: [B, C]                       Kernel: [C, B, 1, 1]
 * Output: [A, C]                       Output: [1, 1, A, C]
 *       |                                         |
 *      Add (const)                            Add (const)
 *                                                 |
 *                                      Transose (NCHW -> NHWC)
 */
class ConvertMatmulWithBiasToPointWiseConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMatmulWithBiasToPointWiseConvolution();
};

/**
 * @brief Convert a MatMul with batch size unsupported by GNA to a point-wise convolution with NHWC layout
 * with transposes around it, moved add with bias and/or fake quantize before the last transpose:
 *                                      Transose (NHWC -> NCHW)
 *                                                 |
 * Matmul                               Convolution in NHWC layout
 * Input1: [A, B] B > 8     ------->    Input: [1, 1, A, B]
 * Input2: [B, C]                       Kernel: [C, B, 1, 1]
 * Output: [A, C]                       Output: [1, 1, A, C]
 *       |                                         |
 *      Add (const)                            Add (const)
 *       |                                         |
 *     FakeQuantize                            FakeQuantize
 *                                                 |
 *                                         Transose (NCHW -> NHWC)
 */
class ConvertMatmulWithFqToPointWiseConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMatmulWithFqToPointWiseConvolution();
};

} // namespace GNAPluginNS