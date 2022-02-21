// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity.hpp"

#include <memory>
#include <vector>
#include <unordered_map>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include "transformations/utils/utils.hpp"

#include "rt_info/optimal_batch_size.hpp"

#include "markup_optimal_bs.hpp"
#include "propagate_optimal_bs.hpp"
#include "switch_affinity.hpp"

#include <ngraph/pass/serialize.hpp>

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MixedAffinity, "MixedAffinity", 0);

namespace {
std::unordered_map<size_t, ov::intel_cpu::Subgraph> formSubgraphs(const std::shared_ptr<ov::Model>& m) {
    std::unordered_map<size_t, ov::intel_cpu::Subgraph> subgraphs;

    auto optimal_bs_is_equal = [](const std::shared_ptr<ov::Node>& node, const size_t value) {
        return ov::intel_cpu::has_optimal_bs(node) && ov::intel_cpu::get_optimal_bs(node) == value;
    };

    auto add_start = [&subgraphs](const ov::Input<ov::Node>& start, const size_t opt_bs) {
        if (subgraphs.count(opt_bs)) {
            subgraphs[opt_bs].starts.insert(start);
        } else {
            subgraphs[opt_bs] = ov::intel_cpu::Subgraph{{start}, {}};
        }
    };

    auto add_end = [&subgraphs](const ov::Output<ov::Node>& end, const size_t opt_bs) {
        if (subgraphs.count(opt_bs)) {
            subgraphs[opt_bs].ends.insert(end);
        } else {
            subgraphs[opt_bs] = ov::intel_cpu::Subgraph{{}, {end}};
        }
    };

    for (const auto& node : m->get_ordered_ops()) {
        if (!ov::intel_cpu::has_optimal_bs(node))
            continue;

        const size_t opt_bs = ov::intel_cpu::get_optimal_bs(node);
        // TODO: remove this WA
        if (opt_bs == 0)
            continue;

        for (const auto& input : node->inputs()) {
            const auto node = input.get_source_output().get_node_shared_ptr();

            if (!ov::is_type<ngraph::opset1::Constant>(node) && !optimal_bs_is_equal(node, opt_bs))
                add_start(input, opt_bs);
        }

        for (const auto& output : node->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                const auto node = target_input.get_node()->shared_from_this();

                if (!ov::is_type<ngraph::opset1::Constant>(node) && !optimal_bs_is_equal(node, opt_bs))
                    add_end(output, opt_bs);
            }
        }
    }

    return subgraphs;
}
}  // namespace

bool ov::intel_cpu::MixedAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager markup_manager(get_pass_config());
    markup_manager.register_pass<ngraph::pass::Serialize>("C://models//test.xml", "C://models//test.bin");
    markup_manager.register_pass<ov::intel_cpu::MarkupOptimalBS>();
    markup_manager.register_pass<ov::intel_cpu::PropagateOptimalBS>();
    markup_manager.run_passes(m);

    // get graph components separated by batch size
    const auto& subgraphs = formSubgraphs(m);

    ov::pass::Manager switch_affinity_manager(get_pass_config());
    // TODO: remove 'share_constants' parameter
    switch_affinity_manager.register_pass<ov::intel_cpu::SwitchAffinity>(subgraphs, false);
    switch_affinity_manager.register_pass<ngraph::pass::Serialize>("C://models//affinity.xml",
                                                                   "C://models//affinity.bin");
    switch_affinity_manager.run_passes(m);

    return false;
}
