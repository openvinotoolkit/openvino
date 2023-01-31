// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"
#include "markup_optimal_bs.hpp"
#include "switch_affinity.hpp"
#include "mixed_affinity.hpp"

#include <memory>
#include <vector>
#include <unordered_map>

#include <openvino/pass/manager.hpp>
#include <openvino/opsets/opset1.hpp>

#include "rt_info/mixed_affinity_props.hpp"

using namespace ov::intel_cpu::mixed_affinity;

NGRAPH_RTTI_DEFINITION(MixedAffinity, "MixedAffinity", 0);

std::unordered_map<Properties, Subgraph> MixedAffinity::formSubgraphs(const std::shared_ptr<ov::Model>& m) {
    std::unordered_map<Properties, Subgraph> subgraphs;

    auto subgraph_props_match = [](const std::shared_ptr<ov::Node>& node, const Properties& props) {
        return has_properties(node) && get_properties(node) == props;
    };

    auto add_start = [&subgraphs](const ov::Input<ov::Node>& start, const Properties& key) {
        if (subgraphs.count(key))
            subgraphs[key].starts.push_back(start);
        else
            subgraphs[key] = Subgraph{{start}, {}};
    };

    auto add_end = [&subgraphs](const ov::Output<ov::Node>& end, const Properties& key) {
        if (subgraphs.count(key))
            subgraphs[key].ends.push_back(end);
        else
            subgraphs[key] = Subgraph{{}, {end}};
    };

    for (const auto& node : m->get_ordered_ops()) {
        if (!has_properties(node))
            continue;

        const Properties props = get_properties(node);
        for (const auto& input : node->inputs()) {
            const auto input_node = input.get_source_output().get_node_shared_ptr();
            const bool non_data_const = input.get_index() > 0 && ov::is_type<ov::opset1::Constant>(input_node);
            if (!non_data_const && !subgraph_props_match(input_node, props)) {
                add_start(input, props);
            }
        }

        for (const auto& output : node->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                const auto output_node = target_input.get_node()->shared_from_this();
                if (!subgraph_props_match(output_node, props)) {
                    add_end(output, props);
                }
            }
        }
    }

    return subgraphs;
}

bool MixedAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager markup_manager(get_pass_config());
    markup_manager.set_per_pass_validation(false);
    markup_manager.register_pass<MarkupOptimalBS>();
    markup_manager.register_pass<PropagateOptimalBS>();
    markup_manager.run_passes(m);

    const auto& subgraphs = formSubgraphs(m);
    ov::pass::Manager switch_affinity_manager(get_pass_config());
    switch_affinity_manager.register_pass<SwitchAffinity>(subgraphs);
    switch_affinity_manager.run_passes(m);

    return false;
}
