// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

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
class ExtractPairsAfterMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ExtractPairsAfterMatmul");
    ExtractPairsAfterMatmul();
};

/**
 * @interface RankUpgradeToRankReduction
 * @brief This pass tries to transfer rank upgrade reshape from MHA branch to rank reduction reshape in MHA input branch
 * Subgraph inside the MHA body:
 *
 *               Matmul                        Matmul
 *                 |   input1                    |  input1   input2
 *                 |  /                          |  /        /
 *                Add1                           Add1       /
 *                 |               ==>            |    reshape
 *                 |                              |   /
 *               Reshape                          Add2
 *                 |   input2
 *                 |  /
 *                Add2
 *                 |
 *                 |
 *               Reshape
 *
 * @ingroup snippets
 */
class RankUpgradeToRankReduction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::RankUpgradeToRankReduction");
    RankUpgradeToRankReduction();
};

class ExtractReshapesFromMHA : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ExtractReshapesFromMHA");
    ExtractReshapesFromMHA() {
        add_matcher<ExtractPairsAfterMatmul>();
        add_matcher<RankUpgradeToRankReduction>();
    }
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
