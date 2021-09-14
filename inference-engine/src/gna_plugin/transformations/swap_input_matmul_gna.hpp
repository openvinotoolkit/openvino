// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SWAP_INPUT_MATMUL_GNA_HPP
#define SWAP_INPUT_MATMUL_GNA_HPP

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {
// @brief Swaps and transposes inputs of MatMul if
// 1. its first input is const and its batch size isn't supported by GNA
// 2. its first input is non-const and its batch size isn't supported by GNA
class SwapInputMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMul();
};

class SwapInputMatMulWithBias: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulWithBias();
};

class SwapInputMatMulWithFq: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulWithFq();
};
} // namespace GNAPluginNS

#endif // SWAP_INPUT_MATMUL_GNA_HPP
