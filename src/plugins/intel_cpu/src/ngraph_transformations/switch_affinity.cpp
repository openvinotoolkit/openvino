// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "switch_affinity.hpp"

#include <transformations/serialize.hpp>

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include "rt_info/graph_component.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

#include <transformations/serialize.hpp>

namespace {
using namespace ngraph;

void serialize(const std::shared_ptr<ngraph::Function> func, std::string name) {
    std::string path = "C://models//" + name;
    ngraph::pass::Serialize(path + ".xml", path + ".bin").run_on_function(func);
}

void setBatch(const std::shared_ptr<ngraph::Function> func, size_t batch) {
    for (auto&& param : func->get_parameters()) {
        auto param_shape = param->get_partial_shape();
        param_shape[0] = batch;
        param->set_partial_shape(param_shape);
    }
    func->validate_nodes_and_infer_types();
}

bool switchToImageAffinity(const std::set<ov::Input<ov::Node>>& starts,
                           const std::set<ov::Output<ov::Node>>& ends,
                           const size_t optimal_bs,
                           const bool share_constants,
                           const std::shared_ptr<Function>& orig_f) {
    // TODO: in general, batch could be nonzero dimension
    const size_t batch_size = starts.begin()->get_partial_shape()[0].get_length();
    const size_t num_splits = batch_size / optimal_bs;
    if (batch_size == optimal_bs) {
        return false;
    }

    ov::NodeVector start_splits;
    ov::ParameterVector main_params;
    start_splits.reserve(starts.size());
    main_params.reserve(starts.size());

    // insert split by batch in the original function and
    // create parameters to extract graph component from original function
    for (const auto& start : starts) {
        // TODO: in general, batch could be nonzero dimension
        const size_t axis_value = 0;
        const auto split_axis = opset1::Constant::create(element::i32, {}, {axis_value});
        const auto split = std::make_shared<opset1::Split>(start.get_source_output(), split_axis, num_splits);
        const auto main_param = std::make_shared<ngraph::opset1::Parameter>(start.get_element_type(), start.get_partial_shape());

        start.replace_source_output(main_param);
        start_splits.push_back(split);
        main_params.push_back(main_param);
    }

    ov::OutputVector result_vec;
    result_vec.reserve(ends.size());
    for (const auto& end : ends) {
        result_vec.push_back(end);
    }

    // create a function from a graph component
    const auto subgraph = std::make_shared<ngraph::Function>(result_vec, main_params);
    NodeMap constants;
    if (share_constants) {
        for (const auto& op : subgraph->get_ordered_ops()) {
            if (ov::is_type<opset1::Constant>(op)) {
                constants[op.get()] = op;
            }
        }
    }

    // map to match the old output and new outputs with optimal batch from the subgraph
    std::map<Output<Node>, OutputVector> concatenate_map;
    for (const auto& original_out : result_vec) {
        concatenate_map[original_out] = OutputVector(num_splits);
    }

    for (size_t batch_idx = 0; batch_idx < num_splits; ++batch_idx) {
        auto cloned_nodes = constants;
        const auto subgraph_with_opt_batch = ngraph::clone_function(*subgraph, cloned_nodes);
        setBatch(subgraph_with_opt_batch, optimal_bs);

        // starts processing
        for (size_t start_idx = 0; start_idx < start_splits.size(); ++start_idx) {
            const auto& cur_param = subgraph_with_opt_batch->get_parameters()[start_idx];
            replace_output_update_name(cur_param, start_splits[start_idx]->output(batch_idx));
        }

        // ends processing
        for (auto& elem : concatenate_map) {
            const auto& orig_out = elem.first;
            const auto& clone_with_opt_batch = cloned_nodes.find(orig_out.get_node());
            // TODO: exception if not found

            elem.second[batch_idx] = clone_with_opt_batch->second->output(orig_out.get_index());
        }
    }

    // concatenate per-batch graphs
    for (const auto& elem : concatenate_map) {
        // TODO: batch dimension could be non-zero
        const size_t concat_axis = 0;
        const auto concat = std::make_shared<ngraph::opset1::Concat>(elem.second, concat_axis);
        replace_output_update_name(elem.first, concat->output(0));
        copy_runtime_info(elem.first.get_node_shared_ptr(), concat);
    }
    return true;
}

struct Subgraph {
    std::set<ov::Input<ov::Node>> starts;
    std::set<ov::Output<ov::Node>> ends;
};
} // namespace

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::SwitchAffinity, "SwitchAffinity", 0);

bool ov::intel_cpu::SwitchAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    std::unordered_map<size_t, Subgraph> subgraphs;
    bool rewritten = false;

    auto optimal_bs_is_equal = [](const std::shared_ptr<ov::Node>& node, const size_t value) {
        return has_optimal_bs(node) && get_optimal_bs(node) == value;
    };

    auto add_start = [&subgraphs](const ov::Input<ov::Node>& start, const size_t opt_bs) {
        if (subgraphs.count(opt_bs)) {
            subgraphs[opt_bs].starts.insert(start);
        } else {
            subgraphs[opt_bs] = Subgraph{{start}, {}};
        }
    };

    auto add_end = [&subgraphs](const ov::Output<ov::Node>& end, const size_t opt_bs) {
        if (subgraphs.count(opt_bs)) {
            subgraphs[opt_bs].ends.insert(end);
        } else {
            subgraphs[opt_bs] = Subgraph{{}, {end}};
        }
    };

    for (const auto& node : m->get_ordered_ops()) {
        if (!has_optimal_bs(node))
            continue;

        const size_t opt_bs = get_optimal_bs(node);
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

    for (const auto& subgraph : subgraphs) {
        std::cout << "SUBGRAPH" << std::endl;
        std::cout << "Batch size: " << subgraph.first << std::endl;
        std::cout << "Starts: " << std::endl;
        for (const auto& start : subgraph.second.starts) {
            std::cout << start << std::endl;
        }
        std::cout << "Ends: " << std::endl;
        for (const auto& end : subgraph.second.ends) {
            std::cout << end.get_node_shared_ptr() << std::endl;
        }
        std::cout << "\n\n" << std::endl;
        switchToImageAffinity(subgraph.second.starts, subgraph.second.ends, subgraph.first, share_constants, m);
    }

    return rewritten;
}

ov::intel_cpu::SwitchAffinity::SwitchAffinity(const bool share_constants)
    : ngraph::pass::FunctionPass(),
      share_constants(share_constants) {}
