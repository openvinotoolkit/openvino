// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/common_optimizations.hpp"


namespace ngraph {
namespace snippets {
namespace pass {

void SetSnippetsNodeType(const std::shared_ptr<Node> &node, SnippetsNodeType nodeType) {
    auto &rt = node->get_rt_info();
    rt["SnippetsNodeType"] = nodeType;
}

SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node> &node) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsNodeType")
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsNodeType");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    return rinfo->second.as<SnippetsNodeType>();
}

void SetTopologicalOrder(const std::shared_ptr<Node> &node, int64_t order) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto &rt = node->get_rt_info();
    rt["TopologicalOrder"] = order;
}

int64_t GetTopologicalOrder(const std::shared_ptr<const Node> &node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("TopologicalOrder");
    if (rinfo == rt.end())
        throw ngraph_error("Topological order is required, but not set.");
    return rinfo->second.as<int64_t>();
}

bool EnumerateNodes::run_on_model(const std::shared_ptr<ov::Model> &m) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::EnumerateNodes")
    int64_t order = 0;
    // Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
    for (auto &node : m->get_ordered_ops()) {
        SetTopologicalOrder(node, order++);
    }
    return true;
}


bool SnippetsTokenization::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SnippetsTokenization);
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);

    manager.register_pass<EnumerateNodes>();
    manager.register_pass<TokenizeMHASnippets>();
    manager.register_pass<TokenizeSnippets>();
    manager.register_pass<CommonOptimizations>();
    manager.run_passes(m);

    // Returning value is false because pass::Manager always apply Validation pass if function was changed.
    // But we don't need to validate the model
    return false;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
