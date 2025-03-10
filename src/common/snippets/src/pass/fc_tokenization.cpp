// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/fc_tokenization.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/tokenization_utils.hpp"

ov::snippets::pass::TokenizeFCSnippets::TokenizeFCSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeFCSnippets);
    // TODO: extend constant path coverage
    // Ticket: 153480
    auto constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_matmul = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), constant});

    auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeFCSnippets")
        const auto matmul = m.get_match_root();
        if (transformation_callback(matmul)) {
            return false;
        }
        return ov::snippets::utils::tokenize_node(matmul, config);
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(m_matmul, matcher_name);
    register_matcher(matcher, callback);
}
