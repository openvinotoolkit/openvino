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

void fillCandidatesOutsToConcatenateFromOriginalFunc(const std::shared_ptr<ngraph::Function>& main_function,
                                     const NodeMap& cloned_nodes,
                                     std::map<Output<Node>, OutputVector>& candidates_to_concatenate,
                                     const size_t num_splits) {
    for (const auto& node : main_function->get_ops()) {
        if (ov::is_type<opset1::Parameter>(node) || ov::is_type<opset1::Constant>(node))
            continue;

        auto outputs = node->outputs();
        for (const auto& output : outputs) {
            auto target_inputs = output.get_target_inputs();
            if (std::any_of(target_inputs.begin(), target_inputs.end(), [&](const ov::Input<Node>& input) {
                    return cloned_nodes.count(input.get_node()) == 0;
                })) {
                candidates_to_concatenate[output] = OutputVector(num_splits);
                break;
            }
        }
    }
}

void addOutToConcatenate(const std::shared_ptr<ngraph::Function>& clone_function,
                         const NodeMap& cloned_nodes,
                         std::map<Output<Node>, OutputVector>& concatenate_map,
                         const size_t idx) {
    for (auto& elem : concatenate_map) {
        const auto& orig_out = elem.first;
        const auto& candidate = cloned_nodes.find(orig_out.get_node());
        // TODO: exception if not found

        auto cloned_out = (*candidate).second->output(orig_out.get_index());
        elem.second[idx] = cloned_out;
    }
}

void setBatch(const std::shared_ptr<ngraph::Function> func, size_t batch) {
    for (auto&& param : func->get_parameters()) {
        auto param_shape = param->get_partial_shape();
        param_shape[0] = batch;
        param->set_partial_shape(param_shape);
    }
    func->validate_nodes_and_infer_types();
}

bool switchToImageAffinity(const ov::NodeVector& starts,
                           const ov::NodeVector& ends,
                           const bool share_constants) {
    const size_t batch_size = starts[0]->get_input_partial_shape(0)[0].get_length();
    const size_t reduced_batch_size = MKLDNNPlugin::get_optimal_bs(starts[0]);
    const size_t num_splits = batch_size / reduced_batch_size;
    if (batch_size == reduced_batch_size) {
        return false;
    }

    // split original function at the start nodes
    ov::NodeVector start_splits(starts.size());
    ov::ParameterVector main_params(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) {
        const auto& start = starts[i];
        const auto axis = opset1::Constant::create(element::i32, {}, {0});
        const auto split = std::make_shared<opset1::Split>(start->input_value(0), axis, num_splits);
        const auto main_param = std::make_shared<ngraph::opset1::Parameter>(start->get_input_element_type(0),
                                                                            start->get_input_partial_shape(0));
        start->set_argument(0, main_param);

        start_splits[i] = split;
        main_params[i] = main_param;
    }

    const auto main_function = std::make_shared<ngraph::Function>(ends, main_params);
    NodeMap constants;
    if (share_constants) {
        for (const auto& op : main_function->get_ordered_ops()) {
            if (ov::is_type<opset1::Constant>(op)) {
                constants[op.get()] = op;
            }
        }
    }

    auto cloned_nodes = constants;
    const auto cur_function = ngraph::clone_function(*main_function, cloned_nodes);
    setBatch(cur_function, reduced_batch_size);

    std::map<Output<Node>, OutputVector> concatenate_map;
    fillCandidatesOutsToConcatenateFromOriginalFunc(main_function, cloned_nodes, concatenate_map, num_splits);
    for (size_t start_idx = 0; start_idx < start_splits.size(); ++start_idx) {
        replace_output_update_name(cur_function->get_parameters()[start_idx]->output(0),
                                   start_splits[start_idx]->output(0));
        addOutToConcatenate(cur_function, cloned_nodes, concatenate_map, 0);

        // insert per-batch graphs after splits
        for (size_t i = 1; i < num_splits; ++i) {
            cloned_nodes = constants;
            const auto cur_function = ngraph::clone_function(*main_function, cloned_nodes);
            setBatch(cur_function, reduced_batch_size);
            addOutToConcatenate(cur_function, cloned_nodes, concatenate_map, i);
            replace_output_update_name(cur_function->get_parameters()[start_idx]->output(0),
                                       start_splits[start_idx]->output(i));
        }
    }

    // concatenate per-batch graphs
    for (const auto& elem : concatenate_map) {
        const auto original_node = elem.first.get_node_shared_ptr();
        // TODO: batch dimension could be non-zero
        const size_t axis = 0;
        const auto concat = std::make_shared<ngraph::opset1::Concat>(elem.second, axis);
        copy_runtime_info(original_node, concat);
        replace_node(original_node, concat);
    }
    return true;
}
} // namespace

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::SwitchAffinity, "SwitchAffinity", 0);

bool MKLDNNPlugin::SwitchAffinity::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool rewritten = false;

    std::set<std::shared_ptr<MKLDNNPlugin::GraphComponent>> switched_subgraphs;
    for (const auto& node : f->get_ordered_ops()) {
        if (MKLDNNPlugin::has_graph_component(node)) {
            const auto graph_component = MKLDNNPlugin::get_graph_component(node);
            if (switched_subgraphs.count(graph_component) == 0) {
                rewritten |= switchToImageAffinity(graph_component->get_starts(), graph_component->get_ends(), share_constants);
                switched_subgraphs.insert(graph_component);
            }
        }
    }
    return rewritten;
}

MKLDNNPlugin::SwitchAffinity::SwitchAffinity(const bool share_constants)
    : ngraph::pass::FunctionPass(),
      share_constants(share_constants) {}
