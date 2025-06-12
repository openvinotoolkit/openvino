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
 * @interface TokenizeGatedMLPSnippets
 * @brief The pass tokenizes Gated MLP pattern into Subgraph
 *        Pattern:
 *            Input
 *           /      \
 * FullyConnected  FullyConnected
 *          |         |
 *      Activation   /
 *          \       /
 *           Multiply
 *              |
 *         FullyConnected
 * @ingroup snippets
 */
class TokenizeGatedMLPSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeGatedMLPSnippets");
    TokenizeGatedMLPSnippets(const SnippetsTokenization::Config& config);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
