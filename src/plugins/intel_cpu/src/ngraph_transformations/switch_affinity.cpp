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

std::map<Output<Node>, OutputVector> createConcatenateMap(const OutputVector& original_outputs, const size_t num_splits) {
    std::map<Output<Node>, OutputVector> res;
    for (const auto& original_out : original_outputs) {
        res[original_out] = OutputVector(num_splits);
    }
    return res;
}

void addCurOutToConcatenateMap(const NodeMap& cloned_nodes,
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

bool switchToImageAffinity(const std::set<ov::Output<ov::Node>>& starts,
                           const std::set<ov::Output<ov::Node>>& ends,
                           const size_t optimal_bs,
                           const bool share_constants,
    const std::shared_ptr<Function>& orig_f) {
    const size_t batch_size = starts.begin()->get_partial_shape()[0].get_length();
    const size_t num_splits = batch_size / optimal_bs;
    if (batch_size == optimal_bs) {
        return false;
    }

    // split original function at the start nodes
    ov::NodeVector start_splits;
    ov::ParameterVector main_params;
    start_splits.reserve(starts.size());
    main_params.reserve(starts.size());

    auto get_input_idx = [](const ov::Output<ov::Node>& parent_out, const std::shared_ptr<ov::Node>& child){
        for (size_t i = 0; i < child->get_input_size(); ++i) {
            if (child->input_value(i) == parent_out) {
                return i;
            }
        }
        // TODO: replace to exception
        assert(false);
    };

    for (const auto& start : starts) {
        const auto split_axis = opset1::Constant::create(element::i32, {}, {0});
        const auto split = std::make_shared<opset1::Split>(start, split_axis, num_splits);
        const auto main_param = std::make_shared<ngraph::opset1::Parameter>(start.get_element_type(), start.get_partial_shape());
        std::cout << split << std::endl;
        for (const auto& input : start.get_target_inputs()) {
            if (input.get_node() == split.get())
                continue;

            const auto input_node = input.get_node()->shared_from_this();
            const size_t input_idx = get_input_idx(start, input_node);
            input_node->set_argument(input_idx, main_param);
        }
        std::cout << split << std::endl;

        start_splits.push_back(split);
        main_params.push_back(main_param);
    }

    serialize(orig_f, "orig_func");
    ov::OutputVector result_vec;
    result_vec.reserve(ends.size());
    for (const auto& end : ends) {
        result_vec.push_back(end);
    }

    const auto main_function = std::make_shared<ngraph::Function>(result_vec, main_params);
    serialize(main_function, "main_func");
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
    setBatch(cur_function, optimal_bs);

    auto concatenate_map = createConcatenateMap(result_vec, num_splits);
    //fillCandidatesOutsToConcatenateFromOriginalFunc(main_function, cloned_nodes, concatenate_map, num_splits);
    for (size_t start_idx = 0; start_idx < start_splits.size(); ++start_idx) {
        std::cout << start_splits[start_idx] << std::endl;
        replace_output_update_name(cur_function->get_parameters()[start_idx]->output(0),
                                   start_splits[start_idx]->output(0));
        std::cout << start_splits[start_idx] << std::endl;
        addCurOutToConcatenateMap(cloned_nodes, concatenate_map, 0);
        serialize(orig_f, "orig_func");

        // insert per-batch graphs after splits
        for (size_t i = 1; i < num_splits; ++i) {
            cloned_nodes = constants;
            const auto cur_function = ngraph::clone_function(*main_function, cloned_nodes);
            setBatch(cur_function, optimal_bs);
            addCurOutToConcatenateMap(cloned_nodes, concatenate_map, i);
            replace_output_update_name(cur_function->get_parameters()[start_idx]->output(0),
                                       start_splits[start_idx]->output(i));
            std::cout << start_splits[start_idx] << std::endl;
            serialize(orig_f, "orig_func");
        }
    }

    // TODO: batch dimension could be non-zero
    const size_t concat_axis = 0;
    // concatenate per-batch graphs
    for (const auto& elem : concatenate_map) {
        const auto concat = std::make_shared<ngraph::opset1::Concat>(elem.second, concat_axis);
        const auto original_node = elem.first.get_node_shared_ptr();
        copy_runtime_info(original_node, concat);
        replace_node(original_node, concat);
    }
    return true;
}

struct Subgraph {
    std::set<ov::Output<ov::Node>> starts;
    std::set<ov::Output<ov::Node>> ends;
};
} // namespace

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::SwitchAffinity, "SwitchAffinity", 0);

bool MKLDNNPlugin::SwitchAffinity::run_on_function(std::shared_ptr<ngraph::Function> f) {
    std::unordered_map<size_t, Subgraph> subgraphs;
    bool rewritten = false;

    auto optimal_bs_is_equal = [](const std::shared_ptr<ov::Node>& node, const size_t value) {
        return has_optimal_bs(node) && get_optimal_bs(node) == value;
    };

    auto add_start = [&subgraphs](const ov::Output<ov::Node>& start, const size_t opt_bs) {
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

    for (const auto& node : f->get_ordered_ops()) {
        if (!has_optimal_bs(node))
            continue;

        const size_t opt_bs = get_optimal_bs(node);
        for (const auto& input : node->input_values()) {
            const auto input_node = input.get_node_shared_ptr();
            if (ov::is_type<ngraph::opset1::Constant>(input_node))
                continue;

            if (!optimal_bs_is_equal(input_node, opt_bs)) {
                add_start(input, opt_bs);
            }
        }

        for (const auto& output : node->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                const auto target_input_node = target_input.get_node()->shared_from_this();
                if (ov::is_type<ngraph::opset1::Constant>(target_input_node))
                    continue;

                if (!optimal_bs_is_equal(target_input_node, opt_bs)) {
                    add_end(output, opt_bs);
                }
            }
        }
    }

    for (const auto& subgraph : subgraphs) {
        if (subgraph.first == 0 || subgraph.first == 2)
            continue;
        std::cout << "SUBGRAPH" << std::endl;
        std::cout << "Batch size: " << subgraph.first << std::endl;
        std::cout << "Starts: " << std::endl;
        for (const auto& start : subgraph.second.starts) {
            std::cout << start.get_node_shared_ptr() << std::endl;
        }
        std::cout << "Ends: " << std::endl;
        for (const auto& end : subgraph.second.ends) {
            std::cout << end.get_node_shared_ptr() << std::endl;
        }
        std::cout << "\n\n" << std::endl;
        switchToImageAffinity(subgraph.second.starts, subgraph.second.ends, subgraph.first, share_constants, f);
    }

    //for (const auto& node : f->get_ordered_ops()) {
    //    if (MKLDNNPlugin::has_graph_component(node)) {
    //        const auto graph_component = MKLDNNPlugin::get_graph_component(node);
    //        if (switched_subgraphs.count(graph_component) == 0) {
    //            rewritten |= switchToImageAffinity(graph_component->get_starts(), graph_component->get_ends(), share_constants);
    //            switched_subgraphs.insert(graph_component);
    //        }
    //    }
    //}
    return rewritten;
}

MKLDNNPlugin::SwitchAffinity::SwitchAffinity(const bool share_constants)
    : ngraph::pass::FunctionPass(),
      share_constants(share_constants) {}
