// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface TokenizeFCSnippets
 * @brief The pass tokenizes FullyConnected like (with constant path on B input) MatMuls
 * @ingroup snippets
 */
class SNIPPETS_API TokenizeFCSnippets : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::TokenizeFCSnippets");
    explicit TokenizeFCSnippets(const TokenizationConfig& config);
};

}  // namespace ov::snippets::pass
