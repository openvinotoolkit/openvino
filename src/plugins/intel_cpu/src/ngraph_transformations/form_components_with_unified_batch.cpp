// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "form_components_with_unified_batch.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/graph_component.hpp"
#include "rt_info/optimal_batch_size.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

using namespace MKLDNNPlugin;

namespace {
std::shared_ptr<GraphComponent> propagate_graph_component(const std::shared_ptr<ov::Node>& node,
                                                          const std::shared_ptr<GraphComponent>& graph_attr,
                                                          const size_t opt_batch) {
    std::shared_ptr<GraphComponent> new_component_attr = graph_attr;
    // skip layers that don't have batch preferences and continue propagation
    if (!has_graph_component(node)) {
        for (const auto& input : node->input_values()) {
            new_component_attr = propagate_graph_component(input.get_node_shared_ptr(), new_component_attr, opt_batch);
        }
    } else {
        // break propagation if preferred batches are different
        if (get_optimal_bs(node) != opt_batch) {
            return new_component_attr;
        }

        new_component_attr = get_graph_component(node);
        // graph components intersection
        if (new_component_attr.get() != graph_attr.get()) {
            const auto& new_starts = graph_attr->get_starts();
            const auto& new_ends = graph_attr->get_ends();

            // starts of subgraph
            const bool component_is_elementary = new_starts.size() == 1 && new_ends.size() == 1 && new_starts[0] == new_ends[0];
            if (!component_is_elementary) {
                for (const auto& elem : new_starts) {
                    new_component_attr->add_start(elem);
                }
            }

            // ends of subgraph
            auto base_ends = new_component_attr->get_ends();
            auto& found_elem = std::find(base_ends.begin(), base_ends.end(), node);
            if (found_elem == base_ends.end()) {
                base_ends.push_back(new_ends[0]);
            } else {
                (*found_elem) = new_ends[0];
            }
            new_component_attr->set_ends(base_ends);
        }

        // continue propagation
        for (const auto& input : node->input_values()) {
            new_component_attr = propagate_graph_component(input.get_node_shared_ptr(), new_component_attr, opt_batch);
        }

        // update shared attribute with the new graph component
        update_graph_component(node, new_component_attr);
    }

    return new_component_attr;
}
} // namespace

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::FormComponentsWithUnifiedBatch, "FormComponentsWithUnifiedBatch", 0);

MKLDNNPlugin::FormComponentsWithUnifiedBatch::FormComponentsWithUnifiedBatch() {
    auto can_be_separated_by_batch = [](const ov::Output<ov::Node>& output) {
        return MKLDNNPlugin::has_optimal_bs(output.get_node_shared_ptr());
    };
    auto root = ngraph::pattern::any_input(can_be_separated_by_batch);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto node = m.get_match_root();
        const auto pshape = node->get_output_partial_shape(0);

        // start graph component from layer with batch preferences
        if (!has_graph_component(node)) {
            set_graph_component(node, std::make_shared<GraphComponent>(ov::NodeVector{node}, ov::NodeVector{node}));
        }

        std::shared_ptr<GraphComponent> graph_attr = get_graph_component(node);
        const size_t optimal_batch = MKLDNNPlugin::get_optimal_bs(node);
        // merge components with the same batch
        propagate_graph_component(node, graph_attr, optimal_batch);

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "FormComponentsWithUnifiedBatch");
    this->register_matcher(m, callback);
}
