// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMLPSeqSnippets
 * @brief The pass tokenizes sequence of MLPs into Subgraph
 *        Pattern:
 * [Eltwise, FQ, Softmax]  Constant
 *              |         /
 *           MatMul0
 *              |
 * [Eltwise, FQ, Softmax]  Constant
 *              |        /
 *           MatMul1
 *              |
 *             ...
 * @ingroup snippets
 */
class TokenizeMLPSeqSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeMLPSeqSnippets");
    TokenizeMLPSeqSnippets(const SnippetsTokenization::Config& config);

private:
    static bool is_matmul_supported(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_softmax(const std::shared_ptr<ov::Node>& node);
    static bool is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node);
    static bool is_tensor_supported(const ov::descriptor::Tensor& t);

    static const size_t m_rank;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
