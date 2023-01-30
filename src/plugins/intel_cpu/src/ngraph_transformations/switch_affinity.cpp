// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "switch_affinity.hpp"

#include <memory>
#include <vector>

#include <openvino/opsets/opset1.hpp>
#include <dimension_tracker.hpp>
#include <openvino/op/op.hpp>

#include "ngraph_transformations/op/fully_connected.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using namespace ov::intel_cpu::mixed_affinity;
static const char skip_reshape[] = "skip_reshape";

void force_replace_output(ov::Output<ov::Node> target, const ov::Output<ov::Node>& replacement) {
    const std::string new_name = ov::op::util::get_ie_output_name(target);
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

void setBatch(const std::shared_ptr<ov::Model> model, const size_t batch_value) {
    for (auto&& param : model->get_parameters()) {
        if (param->get_rt_info().count(skip_reshape))
            continue;
        auto param_shape = param->get_partial_shape();
        const size_t batch_idx = get_batch_idx(param_shape);

        param_shape[batch_idx] = batch_value;
        ov::DimensionTracker::set_label(param_shape[batch_idx], batch_label);
        param->set_partial_shape(param_shape);
    }
    model->validate_nodes_and_infer_types();
}

bool switchToImageAffinity(const Properties& props, const Subgraph& subgraph_borders, const bool share_constants) {
    if (props.n_splits == 1)
        return false;

    ov::NodeVector start_splits;
    ov::ParameterVector main_params;
    start_splits.reserve(subgraph_borders.starts.size());
    main_params.reserve(subgraph_borders.starts.size());

    // create split by batch in the original ov::Model if necessary and
    // create parameters to extract graph component from original ov::Model
    for (const auto& start : subgraph_borders.starts) {
        // weights will be shared between clones and mustn't be splitted
        const bool is_shared_weights = start.get_index() == 1 && (ov::is_type<ov::opset1::Convolution>(start.get_node()) ||
                                                                  ov::is_type<ov::opset1::GroupConvolution>(start.get_node()) ||
                                                                  ov::is_type<ov::opset1::ConvolutionBackpropData>(start.get_node()) ||
                                                                  ov::is_type<ov::opset1::GroupConvolutionBackpropData>(start.get_node()) ||
                                                                  ov::is_type<ov::op::util::DeformableConvolutionBase>(start.get_node()) ||
                                                                  ov::is_type<ov::intel_cpu::FullyConnectedNode>(start.get_node()));

        auto start_shape = start.get_partial_shape();
        size_t batch_idx = get_batch_idx(start_shape);
        // if dimension label isn't set for input shape, than check output one
        if (ov::DimensionTracker::get_label(start_shape[batch_idx]) != batch_label) {
            const auto& out_shape = start.get_node()->get_output_partial_shape(0);
            batch_idx = get_batch_idx(out_shape);
            NGRAPH_CHECK(ov::DimensionTracker::get_label(out_shape[batch_idx]) == batch_label,
                         "Batch label must be set for each node output in mixed affinity subgraph.");
        }

        ov::DimensionTracker::set_label(start_shape[batch_idx], batch_label);
        const auto main_param = std::make_shared<ov::opset1::Parameter>(start.get_element_type(), start_shape);
        const size_t cur_bs = start_shape[batch_idx].get_length();
        if (is_shared_weights || cur_bs == 1) {
            start_splits.push_back(start.get_source_output().get_node_shared_ptr());
            main_param->get_rt_info()[skip_reshape] = true;
        } else {
            const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {batch_idx});
            const auto split = std::make_shared<ov::opset1::Split>(start.get_source_output(), split_axis, props.n_splits);
            start_splits.push_back(split);
        }

        main_params.push_back(main_param);
    }

    // Temporary insert params instead of start to extract a subgraph
    for (size_t i = 0; i < subgraph_borders.starts.size(); ++i) {
        subgraph_borders.starts[i].replace_source_output(main_params[i]);
    }

    ov::OutputVector result_vec;
    result_vec.reserve(subgraph_borders.ends.size());
    for (const auto& end : subgraph_borders.ends) {
        result_vec.push_back(end);
    }

    // create an ov::Model from a graph component
    const auto subgraph = std::make_shared<ov::Model>(result_vec, main_params);
    ngraph::NodeMap common_nodes;
    if (share_constants) {
        for (const auto& op : subgraph->get_ordered_ops()) {
            if (ov::is_type<ov::opset1::Constant>(op)) {
                common_nodes[op.get()] = op;
            }
        }
    }

    // map to match an old output and new outputs with optimal batch from subgraphs
    std::map<ov::Output<ov::Node>, ov::OutputVector> concatenate_map;
    for (const auto& original_out : result_vec) {
        concatenate_map[original_out] = ov::OutputVector(props.n_splits);
    }

    auto change_names = [&](const std::shared_ptr<ov::Model>& m, const size_t batch_idx) {
        const std::string suffix = "_" + std::to_string(batch_idx);
        for (const auto& n : m->get_ordered_ops()) {
            n->set_friendly_name(n->get_friendly_name() + suffix);
            for (const auto& output : n->outputs()) {
                std::unordered_set<std::string> new_names;
                for (const auto& name : output.get_tensor_ptr()->get_names())
                    new_names.insert(name + suffix);
                output.get_tensor_ptr()->set_names(new_names);
            }
        }
    };

    setBatch(subgraph, props.opt_bs);
    for (size_t branch_idx = 0; branch_idx < props.n_splits; ++branch_idx) {
        auto cloned_nodes = common_nodes;
        const auto branch = ov::clone_model(*subgraph, cloned_nodes);
        change_names(branch, branch_idx);

        // starts processing
        for (size_t start_idx = 0; start_idx < start_splits.size(); ++start_idx) {
            const auto& cur_param = branch->get_parameters()[start_idx];
            const auto out_to_replace = start_splits[start_idx]->get_output_size() == 1
                                            ? start_splits[start_idx]->output(0)
                                            : start_splits[start_idx]->output(branch_idx);
            replace_output_update_name(cur_param, out_to_replace);
        }

        // ends processing
        for (auto& elem : concatenate_map) {
            const auto& orig_out = elem.first;
            const auto& clone_with_opt_batch = cloned_nodes.find(orig_out.get_node());
            if (clone_with_opt_batch == cloned_nodes.end())
                OPENVINO_UNREACHABLE("Mixed Affinity: clone with optimal batch wasn't found");
            elem.second[branch_idx] = clone_with_opt_batch->second->output(orig_out.get_index());
        }
    }

    // concatenate per-batch graphs
    for (const auto& elem : concatenate_map) {
        const size_t batch_idx = get_batch_idx(elem.first.get_partial_shape());
        const auto concat = std::make_shared<ov::opset1::Concat>(elem.second, batch_idx);
        force_replace_output(elem.first, concat->output(0));
        ov::intel_cpu::cleanMemoryFormats(concat);
    }
    return true;
}
} // namespace

NGRAPH_RTTI_DEFINITION(SwitchAffinity, "SwitchAffinity", 0);

bool SwitchAffinity::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool rewritten = false;
    for (const auto& subgraph : subgraphs) {
        bool status = switchToImageAffinity(subgraph.first, subgraph.second, share_constants);
        rewritten |= status;
    }
    return rewritten;
}

SwitchAffinity::SwitchAffinity(const std::unordered_map<Properties, Subgraph>& subgraphs, const bool share_constants)
    : ov::pass::ModelPass(),
      subgraphs(subgraphs),
      share_constants(share_constants) {}
