// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/tokenization.hpp"

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "snippets/itt.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"
#include "snippets/pass/fc_tokenization.hpp"
#include "snippets/pass/gn_tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

void SetSnippetsNodeType(const std::shared_ptr<Node> &node, SnippetsNodeType nodeType) {
    auto& rt = node->get_rt_info();
    rt["SnippetsNodeType"] = nodeType;
}

void SetSnippetsSubgraphType(const std::shared_ptr<op::Subgraph> &node, SnippetsSubgraphType nodeType) {
    if (node) {
        auto &rt = node->get_rt_info();
        rt["SnippetsSubgraphType"] = nodeType;
    }
}

SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node> &node) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsNodeType")
    auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsNodeType");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    return rinfo->second.as<SnippetsNodeType>();
}

SnippetsSubgraphType GetSnippetsSubgraphType(const std::shared_ptr<const op::Subgraph> &node) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsSubgraphType")
    if (!node)
        return SnippetsSubgraphType::NotSet;
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsSubgraphType");
    if (rinfo == rt.end())
        return SnippetsSubgraphType::NotSet;
    return rinfo->second.as<SnippetsSubgraphType>();
}

void SetTopologicalOrder(const std::shared_ptr<Node> &node, int64_t order) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto& rt = node->get_rt_info();
    rt["TopologicalOrder"] = order;
}

int64_t GetTopologicalOrder(const std::shared_ptr<const Node> &node) {
    auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("TopologicalOrder");
    if (rinfo == rt.end())
        OPENVINO_THROW("Topological order is required, but not set.");
    return rinfo->second.as<int64_t>();
}

bool EnumerateNodes::run_on_model(const std::shared_ptr<ov::Model> &m) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::EnumerateNodes")
    int64_t order = 0;
    // Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
    for (auto& node : m->get_ordered_ops()) {
        SetTopologicalOrder(node, order++);
    }
    return true;
}


bool SnippetsTokenization::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SnippetsTokenization);
    ov::pass::Manager manager(get_pass_config(), "Snippets:Tokenization");
    manager.set_per_pass_validation(false);

    manager.register_pass<EnumerateNodes>();
    manager.register_pass<ExtractReshapesFromMHA>();
    // This pass mustn't be registered in GraphRewrite with other tokenization passes because of 2 reasons:
    // 1. It has higher priority than other tokenization passes
    // 2. It changes the nodes after the matched root node
    manager.register_pass<TokenizeMHASnippets>(m_config);

    auto tokenization_passes = manager.register_pass<ov::pass::GraphRewrite>();
    tokenization_passes->add_matcher<TokenizeGNSnippets>();
    tokenization_passes->add_matcher<TokenizeFCSnippets>(m_config);
    tokenization_passes->add_matcher<TokenizeSnippets>(m_config);

    manager.register_pass<CommonOptimizations>(m_config);
    manager.run_passes(m);

    // Returning value is false because pass::Manager always apply Validation pass if function was changed.
    // But we don't need to validate the model
    return false;
}

} // namespace pass
} // namespace snippets
} // namespace ov
