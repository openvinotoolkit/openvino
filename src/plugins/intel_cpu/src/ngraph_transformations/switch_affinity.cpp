// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "switch_affinity.hpp"

#include <transformations/serialize.hpp>

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

namespace {
using namespace ngraph;

void serialize(const std::shared_ptr<ov::Model> model, std::string name) {
    std::string path = "/home/vgolubev/models/" + name;
    ngraph::pass::Serialize(path + ".xml", path + ".bin").run_on_model(model);
}

void force_replace_output(Output<Node> target, const Output<Node>& replacement) {
    // Update output tensor name
    const std::string new_name = ngraph::op::util::get_ie_output_name(target);
    replacement.get_node()->set_friendly_name(new_name);
    NGRAPH_SUPPRESS_DEPRECATED_START
    const auto& output_tensor_name = ov::descriptor::get_ov_tensor_legacy_name(target.get_tensor());
    if (!output_tensor_name.empty()) {
        ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), output_tensor_name);
    } else {
        ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), new_name);
    }

    // Save replacement tensor name before replacement as they will be overridden by the target tensor name
    const auto tensor_name = ov::descriptor::get_ov_tensor_legacy_name(replacement.get_tensor());

    target.replace(replacement);

    // Restore back original replacement tensor name
    ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), tensor_name);
    NGRAPH_SUPPRESS_DEPRECATED_END

    copy_runtime_info({replacement.get_node_shared_ptr(), target.get_node_shared_ptr()},
                      replacement.get_node_shared_ptr());
}

void setBatch(const std::shared_ptr<ov::Model> model, size_t batch) {
    for (auto&& param : model->get_parameters()) {
        auto param_shape = param->get_partial_shape();
        param_shape[0] = batch;
        param->set_partial_shape(param_shape);
    }
    model->validate_nodes_and_infer_types();
}

bool switchToImageAffinity(const std::set<ov::Input<ov::Node>>& starts,
                           const std::set<ov::Output<ov::Node>>& ends,
                           const size_t optimal_bs,
                           const bool share_constants) {
    ov::NodeVector start_splits;
    ov::ParameterVector main_params;
    start_splits.reserve(starts.size());
    main_params.reserve(starts.size());

    size_t num_splits = 1;
    // create split by batch in the original ov::Model if necessary and
    // create parameters to extract graph component from original ov::Model
    for (const auto& start : starts) {
        // weights will be shared between clones and mustn't be splitted
        const bool is_weights = start.get_index() == 1 && (ov::is_type<ngraph::opset1::Convolution>(start.get_node()) ||
                                                           ov::is_type<ngraph::opset1::GroupConvolution>(start.get_node()) ||
                                                           ov::is_type<ngraph::opset1::ConvolutionBackpropData>(start.get_node()) ||
                                                           ov::is_type<ov::intel_cpu::FullyConnectedNode>(start.get_node()));
        const size_t cur_batch_size = start.get_partial_shape()[0].get_length();
        if (is_weights || cur_batch_size == 1) {
            start_splits.push_back(start.get_source_output().get_node_shared_ptr());
        } else {
            // TODO: in general, batch could be nonzero dimension
            const size_t axis_value = 0;
            const size_t cur_num_splits = cur_batch_size / optimal_bs;
            num_splits = std::max(num_splits, cur_num_splits);
            assert(cur_batch_size % optimal_bs == 0);
            if (cur_num_splits == 1)
                return false;

            const auto split_axis = opset1::Constant::create(element::i32, {}, {axis_value});
            const auto split = std::make_shared<opset1::Split>(start.get_source_output(), split_axis, cur_num_splits);
            start_splits.push_back(split);
        }

        const auto main_param = std::make_shared<ngraph::opset1::Parameter>(start.get_element_type(), start.get_partial_shape());
        main_params.push_back(main_param);
    }

    if (num_splits == 1)
        return false;

    // Temporary insert params instead of start to extract subgraph
    size_t k = 0;
    for (const auto& start : starts) {
        start.replace_source_output(main_params[k]);
        k++;
    }

    ov::OutputVector result_vec;
    result_vec.reserve(ends.size());
    for (const auto& end : ends) {
        result_vec.push_back(end);
    }

    // create a ov::Model from a graph component
    const auto subgraph = std::make_shared<ov::Model>(result_vec, main_params);
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

    auto change_names = [&](const std::shared_ptr<ov::Model>& m, const size_t batch_idx) {
        for (const auto& n : m->get_ordered_ops()) {
            n->set_friendly_name(n->get_friendly_name() + "_" + std::to_string(batch_idx));
        }
    };

    for (size_t batch_idx = 0; batch_idx < num_splits; ++batch_idx) {
        auto cloned_nodes = constants;
        const auto subgraph_with_opt_batch = ngraph::clone_function(*subgraph, cloned_nodes);
        change_names(subgraph_with_opt_batch, batch_idx);
        setBatch(subgraph_with_opt_batch, optimal_bs);

        // starts processing
        for (size_t start_idx = 0; start_idx < start_splits.size(); ++start_idx) {
            const auto& cur_param = subgraph_with_opt_batch->get_parameters()[start_idx];
            const auto out_to_replace = start_splits[start_idx]->get_output_size() == 1
                                            ? start_splits[start_idx]->output(0)
                                            : start_splits[start_idx]->output(batch_idx);
            replace_output_update_name(cur_param, out_to_replace);
        }

        // ends processing
        for (auto& elem : concatenate_map) {
            const auto& orig_out = elem.first;
            const auto& clone_with_opt_batch = cloned_nodes.find(orig_out.get_node());
            if (clone_with_opt_batch == cloned_nodes.end())
                OPENVINO_UNREACHABLE("Mixed Affinity: clone with optimal batch wasn't found");

            elem.second[batch_idx] = clone_with_opt_batch->second->output(orig_out.get_index());
        }
    }

    // concatenate per-batch graphs
    for (const auto& elem : concatenate_map) {
        // TODO: batch dimension could be non-zero
        const size_t concat_axis = 0;
        const auto concat = std::make_shared<ngraph::opset1::Concat>(elem.second, concat_axis);
        force_replace_output(elem.first, concat->output(0));
        ov::intel_cpu::cleanMemoryFormats(concat);
    }
    return true;
}
} // namespace

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::SwitchAffinity, "SwitchAffinity", 0);

bool ov::intel_cpu::SwitchAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool rewritten = false;
    for (const auto& subgraph : subgraphs) {
        bool status = switchToImageAffinity(subgraph.second.starts, subgraph.second.ends, subgraph.first, share_constants);
        rewritten |= status;
    }
    return rewritten;
}

ov::intel_cpu::SwitchAffinity::SwitchAffinity(const std::unordered_map<size_t, Subgraph>& subgraphs,
                                              const bool share_constants)
    : ngraph::pass::FunctionPass(),
      subgraphs(subgraphs),
      share_constants(share_constants) {}
