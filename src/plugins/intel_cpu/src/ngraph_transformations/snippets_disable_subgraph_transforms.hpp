// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <snippets/op/subgraph.hpp>
#include <vector>

namespace ov {
namespace intel_cpu {

class SnippetsDisableSubgraphTransforms: public ngraph::pass::MatcherPass {
public:
    using Subgraphs = std::vector<std::shared_ptr<ngraph::snippets::op::Subgraph>>;
    OPENVINO_RTTI("SnippetsDisableSubgraphTransforms", "0");
    SnippetsDisableSubgraphTransforms(Subgraphs & subgraphs);
};

}   // namespace intel_cpu
}   // namespace ov
