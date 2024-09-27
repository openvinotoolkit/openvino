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
 * @interface TokenizeFCSnippets
 * @brief The pass tokenizes FullyConnected like (with constant path on B input) MatMuls
 * @ingroup snippets
 */
class TokenizeFCSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeFCSnippets", "0");
    TokenizeFCSnippets(const SnippetsTokenization::Config& config);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
