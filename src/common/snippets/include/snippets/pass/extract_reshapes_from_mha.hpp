// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface ExtractPairsAfterMatmul
 * @brief This pass tries to extract unsupported reshape pairs around eltwise ops between MatMul and Softmax from MHA
 * Subgraph inside the MHA body:
 *
 *               matmul
 *                 |
 *              Reshape1                      input1      input2
 *                 |   input1                       \     /
 *                 |  /                           ExtractedAdd
 *                Add1            =>                   |
 *                 |   input2                   ExtractedReshape
 *                 |  /                matmul      /
 *                Add2                       \    /
 *                 |                          Add
 *              Reshape2
 *
 * @ingroup snippets
 */
class SNIPPETS_API ExtractPairsAfterMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ExtractPairsAfterMatmul");
    ExtractPairsAfterMatmul();
};

/**
 * @interface RankUpgradeToRankReduction
 * @brief This pass tries to transfer rank upgrade reshape from MHA branch to rank reduction reshape in MHA input
 * branch. This pass can be only applied to static shape, and Eltwise below means BinaryElementwiseArithmetic.
 * Subgraph inside the MHA body:
 *               Matmul                        Matmul
 *                 |   input1                    |  input1   input2
 *                 |  /                          |  /        /
 *               Eltwise1                      Eltwise1     /
 *                 |               ==>           |    reshape
 *                 |                             |   /
 *               Reshape                       Eltwise2
 *                 |   input2
 *                 |  /
 *               Eltwise2
 *                 |
 *                 |
 *               Reshape
 *
 * @ingroup snippets
 */
class SNIPPETS_API RankUpgradeToRankReduction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::RankUpgradeToRankReduction");
    RankUpgradeToRankReduction();
};

class SNIPPETS_API ExtractReshapesFromMHA : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ExtractReshapesFromMHA");
    ExtractReshapesFromMHA() {
        add_matcher<ExtractPairsAfterMatmul>();
        add_matcher<RankUpgradeToRankReduction>();
    }
};

}  // namespace ov::snippets::pass
