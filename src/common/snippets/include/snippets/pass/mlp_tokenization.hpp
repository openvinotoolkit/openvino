// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMLPSnippets
 * @brief The pass tokenizes MLP-like patterns:
 *             Input
 *           /      \
 * FullyConnected  FullyConnected
 *          \    Swish
 *           Multiply
 *         FullyConnected
 * @ingroup snippets
 */
class TokenizeMLPSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeMLPSnippets");
    TokenizeMLPSnippets(const SnippetsTokenization::Config& config);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
