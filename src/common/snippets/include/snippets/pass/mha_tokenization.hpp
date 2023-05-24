// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMHASnippets
 * @brief The pass tokenizes MHA-pattern into Subgraph
 *        Pattern:           Transpose1
 *                               |
 *             Transpose0  Eltwise/Select
 *                     \     /
 *                     MatMul0
 *                        |
 *           Eltwise/Select/Reshape
 *                        |
 *                     Softmax
 *                        |
 *            Eltwise/Select/Reshape  Transpose2
 *                               \      /
 *                                MatMul1
 *                                  |
 *                  Eltwise/Select/Reshape/Transpose3
 *        Notes:
 *          - Transposes can be missed
 *          - Transpose0, Transpose2 and Transpose3 may have only [0,2,1,3] order
 *          - Transpose1 may have only [0,2,3,1] order
 * @ingroup snippets
 */
class TokenizeMHASnippets: public ov::pass::MatcherPass {
public:
    /**
     * @interface Config
     * @brief Allow to adjust tokenization
     * @ingroup snippets
     */
    struct Config {
        Config(bool enable_transpose_token = true) : enable_transpose(enable_transpose_token) {}

        // False if all Transposes aren't tokenized. Otherwise, they may be fused into Subgraph is possible
        bool enable_transpose = true;
    };

    OPENVINO_RTTI("TokenizeMHASnippets", "0");
    TokenizeMHASnippets(const Config& config = {});
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
