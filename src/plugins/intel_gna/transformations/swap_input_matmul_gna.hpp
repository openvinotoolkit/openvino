// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SWAP_INPUT_MATMUL_GNA_HPP
#define SWAP_INPUT_MATMUL_GNA_HPP

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {
/* @brief Swaps and transposes inputs of MatMul and transposes its output if
 * 1. its first input is const and its batch size isn't supported by GNA
 * 2. its first input is non-const and its batch size isn't supported by GNA
 * The following pattern is detected:
 *    Constant           Any input
 *       |                   |
 *   [FakeQuantize]      [Transpose]
 *        \                 /
 *               MatMul
 *                 |
 *               [Add]
 *                 |
 *           [FakeQuantize]
 *                 |
 *            [Activation]
 *                 |
 *            [Transpose]
 * 
 * The existed Transposes will be removed instead of inserting new ones during transposition.
 * Separate matchers are required for different last nodes. They should be registered in the order from the longest
 * to the shortest pattern.
 **/
class SwapInputMatMul: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwapInputMatMul", "0");
    SwapInputMatMul();
};

class SwapInputMatMulWithBias: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwapInputMatMulWithBias", "0");
    SwapInputMatMulWithBias();
};

class SwapInputMatMulWithFq: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwapInputMatMulWithFq", "0");
    SwapInputMatMulWithFq();
};

class SwapInputMatMulWithAct: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwapInputMatMulWithAct", "0");
    SwapInputMatMulWithAct();
};

class SwapInputMatMulWithTrailingTranspose: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwapInputMatMulWithTrailingTranspose", "0");
    SwapInputMatMulWithTrailingTranspose();
};
} // namespace GNAPluginNS

#endif // SWAP_INPUT_MATMUL_GNA_HPP
