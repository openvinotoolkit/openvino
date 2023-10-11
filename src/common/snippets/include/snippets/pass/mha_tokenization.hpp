// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMHASnippets
 * @brief The pass tokenizes MHA-pattern into Subgraph
 *        Pattern:           Transpose1
 *                               |
 *             Transpose0 [Eltwise, Select]
 *                     \     /
 *                     MatMul0
 *                        |
 *           [Eltwise, Select, Reshape]
 *                        |
 *                     Softmax
 *                        |
 *            [Eltwise, Select, Reshape]  Transpose2
 *                               \      /
 *                                MatMul1
 *                                  |
 *                  [Eltwise, Select, Transpose3]
 *        Notes:
 *          - Transposes can be missed
 *          - Transpose0, Transpose2 and Transpose3 may have only [0,2,1,3] order
 *          - Transpose1 may have only [0,2,3,1] order
 *          - [...] means any count of different nodes from list. But:
 *              * Reshapes can be only explicitly around Softmax (Reshape -> Softmax -> Reshape)
 *              * After MatMul1 may be only Transpose3 or any count of Eltwise, Select ops.
 * @ingroup snippets
 */
class TokenizeMHASnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeMHASnippets", "0");
    TokenizeMHASnippets(const SnippetsTokenization::Config& config = {});
    static bool is_matmul0_supported(const std::shared_ptr<ov::opset1::MatMul>& matmul);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
