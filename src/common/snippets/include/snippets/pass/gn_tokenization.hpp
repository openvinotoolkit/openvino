// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeGNSnippets
 * @brief Tokenize GroupNormalization to a subgraph
 * @ingroup snippets
 */
class TokenizeGNSnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeGNSnippets", "0");
    TokenizeGNSnippets();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov