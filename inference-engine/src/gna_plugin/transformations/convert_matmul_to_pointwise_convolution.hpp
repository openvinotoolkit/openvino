// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Convert a MatMul with batch size unsupported by GNA to a point-wise convolution:
 * Matmul                               Convolution in NHWC layout
 * Input1: [A, B] B > 8     ------->    Input: [1, 1, A, B]
 * Input2: [B, C]                       Kernel: [C, B, 1, 1]
 * Output: [A, C]                       Output: [1, 1, A, C]
 */
class ConvertMatmulToPointWiseConvolution : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  ConvertMatmulToPointWiseConvolution();
};

} // namespace GNAPluginNS