// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/switch_merge_resolve.hpp"

#include <unordered_map>
#include <unordered_set>

#include "helper_ops/merge.hpp"
#include "helper_ops/switch.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/opsets/opset11.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow;
using namespace ov::frontend;
using namespace ov::opset11;
using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {

bool pass::SwitchMergeResolver::run_on_model(const shared_ptr<Model>& m) {
    // collect a vector of Switch and Merge nodes corresponding to each CF marker
    unordered_map<int32_t, vector<shared_ptr<Switch>>> switch_map;
    unordered_map<int32_t, vector<shared_ptr<Merge>>> merge_map;
    // fuse Switch-Merge sub-graphs into If operations in topological order
    // however, it seems to be unprincipal
    vector<int32_t> fuse_markers;
    for (const auto& node : m->get_ordered_ops()) {
        if (const auto& switch_node = as_type_ptr<Switch>(node)) {
            FRONT_END_GENERAL_CHECK(
                switch_node->is_cond_flow_marker_set(),
                "[TensorFlow Frontend] internal error: Switch node does not have conditional flow marker");
            auto marker = switch_node->get_cond_flow_marker();
            if (find(fuse_markers.begin(), fuse_markers.end(), marker) == fuse_markers.end()) {
                fuse_markers.push_back(marker);
            }

            if (switch_map.count(marker) > 0) {
                switch_map[marker].push_back(switch_node);
            } else {
                switch_map[marker] = {switch_node};
            }
        } else if (const auto& merge_node = as_type_ptr<Merge>(node)) {
            // in case TF1 While (with Switch, Merge, Enter, Exit and NextIteration)
            // it skips further fusing
            if (!merge_node->is_cond_flow_eliminated()) {
                continue;
            }
            auto eliminated_markers = merge_node->get_eliminated_cond_flow_marker();

            for (const auto& eliminated_marker : eliminated_markers) {
                if (merge_map.count(eliminated_marker) > 0) {
                    merge_map[eliminated_marker].push_back(merge_node);
                } else {
                    merge_map[eliminated_marker] = {merge_node};
                }
            }
        }
    }

    // fuse Switch-Merge sub-graphs into If operation in topological order
    for (const auto& marker : fuse_markers) {
        auto switch_nodes = switch_map[marker];

        // in case TF1 While (with Switch, Merge, Enter, Exit and NextIteration)
        // it skips further fusing
        if (merge_map.count(marker) == 0) {
            continue;
        }
        auto merge_nodes = merge_map[marker];

        FRONT_END_GENERAL_CHECK(
            switch_nodes.size() > 0,
            "[TensorFlow Frontend] internal error: conditional flow must contain Switch node to fuse");
        FRONT_END_GENERAL_CHECK(
            merge_nodes.size() > 0,
            "[TensorFlow Frontend] internal error: conditional flow must contain Merge nodes to fuse");

        auto cond = switch_nodes[0]->input_value(1);

        // collect Parameter nodes and Result nodes for then and else bodies
        // set inputs and outputs for If node
        // create then bodies for which condition is true
        ParameterVector then_params;
        ParameterVector else_params;
        ResultVector then_results;
        ResultVector else_results;
        OutputVector if_inputs;
        OutputVector if_outputs;
        vector<unordered_set<string>> if_outputs_names;
        for (const auto& switch_node : switch_nodes) {
            FRONT_END_GENERAL_CHECK(switch_node->input_values().size() == 2,
                                    "[TensorFlow Frontend] internal error: Switch node must have two inputs");
            if_inputs.push_back(switch_node->input_value(0));
            FRONT_END_GENERAL_CHECK(switch_node->outputs().size() == 2,
                                    "[TensorFlow Frontend] internal error: Switch node must have two outputs");
            auto switch_output_false = switch_node->outputs()[0];
            auto switch_output_true = switch_node->outputs()[1];
            auto parameter_node_true = make_shared<Parameter>(switch_node->get_output_element_type(0),
                                                              switch_node->get_output_partial_shape(0));
            auto parameter_node_false = make_shared<Parameter>(switch_node->get_output_element_type(1),
                                                               switch_node->get_output_partial_shape(1));
            then_params.push_back(parameter_node_true);
            switch_output_true.replace(parameter_node_true);
            else_params.push_back(parameter_node_false);
            switch_output_false.replace(parameter_node_false);
        }
        for (const auto& merge_node : merge_nodes) {
            size_t input_ind = 0;
            for (const auto& merge_input : merge_node->inputs()) {
                auto input_value = merge_input.get_source_output();

                const shared_ptr<const Node>& merge_producer = input_value.get_node_shared_ptr();
                auto producer_cf_marker = get_cf_marker(merge_producer);
                if (producer_cf_marker.existing_markers.count(marker) > 0) {
                    auto branch_index_set = producer_cf_marker.existing_markers[marker];
                    FRONT_END_GENERAL_CHECK(
                        branch_index_set.size() == 1,
                        "[TensorFlow Frontend] internal error: it must contain the single branch index");
                    auto branch_index = *next(branch_index_set.begin(), 0);

                    auto value_index = make_shared<Constant>(element::i32, Shape{}, input_ind);
                    auto result_node_value = make_shared<Result>(input_value);
                    auto result_node_index = make_shared<Result>(value_index);
                    input_value.remove_target_input(merge_input);

                    FRONT_END_GENERAL_CHECK(
                        branch_index == 0 || branch_index == 1,
                        "[TensorFlow Frontend] internal error: conditional branch with unexpected index");
                    if (branch_index == 0) {
                        // handle else-branch since the first output of Switch is output_false
                        else_results.push_back(result_node_value);
                        else_results.push_back(result_node_index);
                    } else if (branch_index == 1) {
                        // handle then-branch since the second output of Switch is output_true
                        then_results.push_back(result_node_value);
                        then_results.push_back(result_node_index);
                    }
                }
                ++input_ind;
            }
            for (const auto& merge_output : merge_node->outputs()) {
                if_outputs.push_back(merge_output);
                if_outputs_names.push_back(merge_output.get_names());
            }
        }

        // create body graphs for then and else branches
        auto then_body = make_shared<Model>(then_results, then_params);
        auto else_body = make_shared<Model>(else_results, else_params);

        auto if_op = make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);

        // set inputs
        size_t input_size = if_inputs.size();
        FRONT_END_GENERAL_CHECK(
            input_size == then_params.size(),
            "[TensorFlow Frontend] internal error: number of Switch inputs does not match a number of outputs");
        FRONT_END_GENERAL_CHECK(
            input_size == else_params.size(),
            "[TensorFlow Frontend] internal error: number of Switch inputs does not match a number of outputs");
        for (size_t ind = 0; ind < input_size; ++ind) {
            auto curr_input = if_inputs[ind];
            auto then_param = then_params[ind];
            auto else_param = else_params[ind];
            if_op->set_input(curr_input, then_param, else_param);
        }

        // set outputs
        FRONT_END_GENERAL_CHECK(then_results.size() == else_results.size(),
                                "[TensorFlow Frontend] internal error: number of result nodes in "
                                "then and else branches do not match.");
        size_t output_size = then_results.size();
        for (size_t ind = 0; ind < output_size; ++ind) {
            if_op->set_output(then_results[ind], else_results[ind]);
        }

        auto ov_outputs = if_op->outputs();
        FRONT_END_GENERAL_CHECK(if_op->outputs().size() == if_outputs.size(),
                                "[TensorFlow Frontend] internal error: number of actual If outputs does not match "
                                "expected number for Switch-Merge nodes fusing into If operation");
        // replace source for all consumers of Merge nodes outputs
        // and move their tensor names to new outputs
        size_t output_ind = 0;
        for (auto& if_output : if_op->outputs()) {
            if_outputs[output_ind].replace(if_output);
            if_output.set_names(if_outputs_names[output_ind]);
            ++output_ind;
        }
    }

    return true;
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
