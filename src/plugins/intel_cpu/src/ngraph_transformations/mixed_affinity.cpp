// Copyright (C) 2022-2023 Intel Corporation
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
#include "rt_info/num_splits.hpp"

#include "markup_optimal_bs.hpp"
#include "propagate_optimal_bs.hpp"
#include "switch_affinity.hpp"

#include <dimension_tracker.hpp>

#include <ngraph/pass/serialize.hpp>
#include <ngraph/pass/visualize_tree.hpp>

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MixedAffinity, "MixedAffinity", 0);

namespace {
size_t get_batch_idx(const ov::PartialShape& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (ov::DimensionTracker::get_label(shape[i]) == ov::intel_cpu::batch_label)
            return i;
    }
    return 0ul;
}

using namespace ov::intel_cpu::mixed_affinity;
std::unordered_map<Characteristics, Subgraph> formSubgraphs(const std::shared_ptr<ov::Model>& m) {
    std::unordered_map<Characteristics, Subgraph> subgraphs;

    auto optimal_bs_is_equal = [](const std::shared_ptr<ov::Node>& node, const size_t value) {
        return ov::intel_cpu::has_optimal_bs(node) && ov::intel_cpu::get_optimal_bs(node) == value;
    };

    auto n_splits_is_equal = [](const std::shared_ptr<ov::Node>& node, const size_t value) {
        return ov::intel_cpu::has_num_splits(node) && ov::intel_cpu::get_num_splits(node) == value;
    };

    auto add_start = [&subgraphs](const ov::Input<ov::Node>& start, const Characteristics& key) {
        if (subgraphs.count(key))
            subgraphs[key].starts.insert(start);
        else
            subgraphs[key] = Subgraph{{start}, {}};
    };

    auto add_end = [&subgraphs](const ov::Output<ov::Node>& end, const Characteristics& key) {
        if (subgraphs.count(key))
            subgraphs[key].ends.insert(end);
        else
            subgraphs[key] = Subgraph{{}, {end}};
    };

    for (const auto& node : m->get_ordered_ops()) {
        if (!ov::intel_cpu::has_optimal_bs(node))
            continue;

        NGRAPH_CHECK(ov::intel_cpu::has_num_splits(node),
                     "formSubgraphs: node ",
                     node->get_friendly_name(),
                     " lacks 'NumSplits' rt info that must be if rt info contains 'OptimalBatchSize' ");
        const size_t n_splits = ov::intel_cpu::get_num_splits(node);
        const size_t opt_bs = ov::intel_cpu::get_optimal_bs(node);
        for (const auto& input : node->inputs()) {
            const auto input_node = input.get_source_output().get_node_shared_ptr();
            const bool non_data_const = input.get_index() > 0 && ov::is_type<ngraph::opset1::Constant>(input_node);
            if (!non_data_const && (!optimal_bs_is_equal(input_node, opt_bs) || !n_splits_is_equal(input_node, n_splits))) {
                add_start(input, Characteristics(opt_bs, n_splits));
            }
        }

        for (const auto& output : node->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                const auto output_node = target_input.get_node()->shared_from_this();
                if (!optimal_bs_is_equal(output_node, opt_bs) || !n_splits_is_equal(output_node, n_splits)) {
                    add_end(output, Characteristics(opt_bs, n_splits));
                }
            }
        }
    }

    return subgraphs;
}
}  // namespace

bool ov::intel_cpu::MixedAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager markup_manager(get_pass_config());
    markup_manager.set_per_pass_validation(false);
    // markup_manager.register_pass<ngraph::pass::Serialize>("/home/vgolubev/models/test.xml", "/home/vgolubev/models/test.bin");
    markup_manager.register_pass<ov::intel_cpu::MarkupOptimalBS>();
    markup_manager.register_pass<ov::intel_cpu::PropagateOptimalBS>();
    // markup_manager.register_pass<ngraph::pass::VisualizeTree>("/home/vgolubev/models/test.before.svg");
    markup_manager.run_passes(m);

    // get graph components separated by batch size
    const auto& subgraphs = formSubgraphs(m);

    for (const auto& subgraph : subgraphs) {
        std::cout << "Batch=" << subgraph.first.opt_bs << std::endl;
        std::cout << "N_splits=" << subgraph.first.n_splits << std::endl;
        std::cout << "\tStarts:\n";
        for (const auto start : subgraph.second.starts) {
            std::cout << "\t\t" << start << std::endl;
        }
        std::cout << "\tEnds:\n";
        for (const auto end : subgraph.second.ends) {
            std::cout << "\t\t" << end << std::endl;
        }
    }

    ov::pass::Manager switch_affinity_manager(get_pass_config());
    // TODO: remove 'share_constants' parameter
    switch_affinity_manager.register_pass<ov::intel_cpu::SwitchAffinity>(subgraphs, true);
    // switch_affinity_manager.register_pass<ngraph::pass::VisualizeTree>("/home/vgolubev/models/test.before.svg");
    // switch_affinity_manager.register_pass<ngraph::pass::Serialize>("/home/vgolubev/models/affinity.xml", "/home/vgolubev/models/affinity.bin");
    switch_affinity_manager.run_passes(m);

    return false;
}
