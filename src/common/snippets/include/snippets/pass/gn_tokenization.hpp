// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface TokenizeGNSnippets
 * @brief Tokenize GroupNormalization to a subgraph
 * @ingroup snippets
 */
class TokenizeGNSnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeGNSnippets");
    TokenizeGNSnippets();
};

}  // namespace ov::snippets::pass
