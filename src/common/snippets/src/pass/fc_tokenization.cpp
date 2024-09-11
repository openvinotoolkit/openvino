// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/fc_tokenization.hpp"

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"

ov::snippets::pass::TokenizeFCSnippets::TokenizeFCSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeFCSnippets);

    // TODO: extend constant path coverage:
    // 1. Add u8/i8/bf16 precisions
    // 2. Add subgraphs (Transpose/Convert)
    // 3. Add Decompression subgraphs support (and all the possible compressed weights related precisions)
    auto constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::type_matches(ov::element::f32));
    auto m_matmul = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), constant});

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, matcher_name),
        [m_matmul](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeFCSnippets")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto matmul = pattern_map.at(m_matmul).get_node_shared_ptr();
        const auto subgraph = op::Subgraph::wrap_node_as_subgraph(matmul);
        subgraph->get_rt_info()["originalLayersNames"] = matmul->get_friendly_name();
        // MatMul weights are stored outside the subgraph
        subgraph->set_virtual_port_count(1);
        op::update_out_tensor_name(subgraph);
        ov::replace_node(matmul, subgraph);
        return true;
    });
}
