// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_disable_subgraph_transforms.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
namespace ov {
namespace intel_cpu {

SnippetsDisableSubgraphTransforms::SnippetsDisableSubgraphTransforms(Subgraphs & subgraphs) {
    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto subgraph = std::dynamic_pointer_cast<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (!subgraph) {
            return false;
        }
        subgraph->allow_transformations(false);
        subgraphs.emplace_back(subgraph);
        return false;
    };

    auto subgraph = ov::pass::pattern::wrap_type<ngraph::snippets::op::Subgraph>();
    auto m = std::make_shared<ov::pass::pattern::Matcher>(subgraph, "SnippetsDisableSubgraphTransforms");
    this->register_matcher(m, callback);
}

}   // namespace intel_cpu
}   // namespace ov
