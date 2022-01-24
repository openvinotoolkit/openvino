// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "affinity_switcher.hpp"

#include <transformations/serialize.hpp>

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
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
                                     const size_t batch_size) {
    for (const auto& node : main_function->get_ops()) {
        if (ov::is_type<opset1::Parameter>(node) || ov::is_type<opset1::Constant>(node))
            continue;

        auto outputs = node->outputs();
        for (const auto& output : outputs) {
            auto target_inputs = output.get_target_inputs();
            if (std::any_of(target_inputs.begin(), target_inputs.end(), [&](const ov::Input<Node>& input) {
                    return cloned_nodes.count(input.get_node()) == 0;
                })) {
                candidates_to_concatenate[output] = OutputVector(batch_size);
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

bool switchToImageAffinity(std::shared_ptr<ngraph::Node> start,
                           std::shared_ptr<ngraph::Node> end,
                           const NodeMap& constants = {}) {
    // split original function
    const size_t batch_size = end->get_input_partial_shape(0)[0].get_length();
    const auto axis = opset1::Constant::create(element::i32, {}, {0});
    const auto split = std::make_shared<opset1::Split>(start->input_value(0), axis, batch_size);

    const auto main_param = std::make_shared<ngraph::opset1::Parameter>(start->get_input_element_type(0),
                                                                        start->get_input_partial_shape(0));
    start->set_argument(0, main_param);
    const auto main_function = std::make_shared<ngraph::Function>(NodeVector{end}, ParameterVector{main_param});

    // TODO: input_shape if not parameter
    auto single_batch_shape = start->get_output_partial_shape(0);
    const size_t reduced_batch_size = 1;
    single_batch_shape[0] = reduced_batch_size;

    // original_node -- shared_ptr on clone
    NodeMap cloned_nodes = constants;
    const auto cur_function = ngraph::clone_function(*main_function, cloned_nodes);
    setBatch(cur_function, reduced_batch_size);

    replace_output_update_name(cur_function->get_parameters()[0]->output(0), split->output(0));

    std::map<Output<Node>, OutputVector> concatenate_map;
    fillCandidatesOutsToConcatenateFromOriginalFunc(main_function, cloned_nodes, concatenate_map, batch_size);
    addOutToConcatenate(cur_function, cloned_nodes, concatenate_map, 0);

    // insert per-batch graphs after split
    for (size_t i = 1; i < batch_size; ++i) {
        cloned_nodes = constants;
        const auto cur_function = ngraph::clone_function(*main_function, cloned_nodes);
        setBatch(cur_function, reduced_batch_size);
        addOutToConcatenate(cur_function, cloned_nodes, concatenate_map, i);
        replace_output_update_name(cur_function->get_parameters()[0]->output(0), split->output(i));
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

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::AffinitySwitcher, "AffinitySwitcher", 0);

bool MKLDNNPlugin::AffinitySwitcher::run_on_function(std::shared_ptr<ngraph::Function> f) {
    NodeMap constants;
    bool rewritten = false;
    std::shared_ptr<Node> start, end;

    for (const auto& node : f->get_ordered_ops()) {
        if (share_constants && ov::is_type<opset1::Constant>(node)) {
            constants[node.get()] = node;
        }

        // TODO: remove after markup pass implementation
        if (node->get_friendly_name() == "resnet_model/conv2d/Conv2D") {
            start = node;
        }

        for (const auto& input : node->input_values()) {
            const auto pShape = input.get_partial_shape();
            const auto rank = pShape.rank();

            if (rank.is_dynamic() || rank.get_length() == 0 || pShape[0].is_dynamic() || pShape[0].get_length() == 1)
                continue;

            if (transformation_callback(node)) {
                end = node;
            }
        }
    }
    if (start && end) {
        rewritten |= switchToImageAffinity(start, end, constants);
    }
    return rewritten;
}

MKLDNNPlugin::AffinitySwitcher::AffinitySwitcher(const bool share_constants)
    : ngraph::pass::FunctionPass(),
      share_constants(share_constants) {}
