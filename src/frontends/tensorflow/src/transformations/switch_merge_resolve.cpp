// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/switch_merge_resolve.hpp"

#include <unordered_map>
#include <unordered_set>

#include "helper_ops/merge.hpp"
#include "helper_ops/switch.hpp"
#include "openvino/core/node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow;
using namespace ov::frontend;
using namespace ov::op;
using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {

namespace {
using ClusterType = pair<unordered_set<shared_ptr<Switch>>, unordered_set<shared_ptr<Merge>>>;

bool intersected(const unordered_set<shared_ptr<Switch>>& s1, const unordered_set<shared_ptr<Switch>>& s2) {
    bool is_intersected = false;
    for (const auto& node1 : s1) {
        for (const auto& node2 : s2) {
            if (node1 == node2) {
                is_intersected = true;
                break;
            }
        }
    }
    return is_intersected;
}

void generate_if_clusters(const shared_ptr<Model>& ov_model,
                          unordered_map<uint32_t, unordered_set<shared_ptr<Switch>>>& switch_clusters,
                          unordered_map<uint32_t, unordered_set<shared_ptr<Merge>>>& merge_clusters) {
    // each pair represents a cluster circled with Switch and Merge nodes
    vector<ClusterType> clusters;
    for (const auto& node : ov_model->get_ordered_ops()) {
        if (const auto& merge_node = as_type_ptr<Merge>(node)) {
            if (!merge_node->is_cond_flow_eliminated()) {
                continue;
            }
            auto eliminated_markers = merge_node->get_eliminated_cond_flow_marker();

            // combine all Switch nodes for which conditional flow is resolved
            // by the current Merge node
            SetOfSwitchNodes switch_nodes;
            for (const auto& eliminated_marker : eliminated_markers) {
                auto curr_switch_nodes = merge_node->get_switch_nodes_set_by_cond_index(eliminated_marker);
                switch_nodes.insert(curr_switch_nodes.begin(), curr_switch_nodes.end());
            }

            // insert into clusters
            ClusterType combined_cluster = {switch_nodes, {merge_node}};
            vector<ClusterType> refined_clusters;
            for (const auto& cluster : clusters) {
                const auto& cluster_switches = cluster.first;
                const auto& cluster_merges = cluster.second;
                if (intersected(cluster_switches, combined_cluster.first)) {
                    combined_cluster.first.insert(cluster_switches.begin(), cluster_switches.end());
                    combined_cluster.second.insert(cluster_merges.begin(), cluster_merges.end());
                } else {
                    refined_clusters.push_back(cluster);
                }
            }
            refined_clusters.push_back(combined_cluster);
            clusters = refined_clusters;
        }
    }

    // repack clusters to two separate maps for Switch and Merge nodes
    switch_clusters.clear();
    merge_clusters.clear();
    uint32_t clusters_size = static_cast<uint32_t>(clusters.size());
    for (uint32_t cluster_id = 0; cluster_id < clusters_size; ++cluster_id) {
        switch_clusters[cluster_id] = clusters[cluster_id].first;
        merge_clusters[cluster_id] = clusters[cluster_id].second;
    }
}

shared_ptr<v0::Parameter> replace_switch_output_with_parameter(const shared_ptr<Switch>& switch_node,
                                                               size_t output_ind) {
    FRONT_END_GENERAL_CHECK(output_ind < 2,
                            "[TensorFlow Frontend] internal error: incorrect output index for Switch node");
    auto switch_output = switch_node->output(output_ind);

    auto parameter_node =
        make_shared<v0::Parameter>(switch_output.get_element_type(), switch_output.get_partial_shape());
    auto cf_marker = get_cf_marker(switch_node);
    auto switch_marker = switch_node->get_cond_flow_marker();
    cf_marker.existing_markers_with_branches[switch_marker].insert(static_cast<uint32_t>(output_ind));
    cf_marker.existing_markers_with_switches[switch_marker].insert(switch_node);
    cf_marker.new_markers.clear();
    set_cf_marker(cf_marker, parameter_node);

    switch_output.replace(parameter_node);

    return parameter_node;
}

void insert_result_before_merge(const shared_ptr<Merge>& merge_node,
                                size_t input_ind,
                                uint32_t& branch_index,
                                shared_ptr<v0::Result>& result_output,
                                shared_ptr<v0::Result>& result_value_index) {
    // check that handled Marge node contains conditional flow marker
    auto merge_node_name = merge_node->get_friendly_name();
    FRONT_END_GENERAL_CHECK(cf_marker_exists(merge_node),
                            "[TensorFlow Frontend] internal error: Merge node " + merge_node_name +
                                " does not have conditional flow marker");

    // get eliminated marker and check that eliminated marker exists
    // Merge node may contain several eliminated markers, in this case it means some Switch nodes have different
    // condition nodes and values generated by this condition nodes are identical
    auto merge_cf_marker = get_cf_marker(merge_node);
    FRONT_END_GENERAL_CHECK(merge_cf_marker.merge_eliminated_markers.size() > 0,
                            "[TensorFlow Frontend] internal error: Merge node " + merge_node_name +
                                " does not contain any eliminated marker");
    auto eliminated_marker = merge_cf_marker.merge_eliminated_markers.begin()->first;

    // check that producer contains the same conditional flow marker
    // and retrive branch index for it
    const auto& merge_input = merge_node->input(input_ind);
    const auto& input_value = merge_node->input_value(input_ind);
    const shared_ptr<const Node>& merge_producer = merge_node->get_input_node_shared_ptr(input_ind);
    auto producer_cf_marker = get_cf_marker(merge_producer);
    FRONT_END_GENERAL_CHECK(
        producer_cf_marker.existing_markers_with_branches.count(eliminated_marker) > 0,
        "[TensorFlow Frontend] internal error: input producer for Merge node does not contain eliminated marker");

    auto branch_index_set = producer_cf_marker.existing_markers_with_branches[eliminated_marker];
    FRONT_END_GENERAL_CHECK(branch_index_set.size() == 1,
                            "[TensorFlow Frontend] internal error: it must contain the single branch index");
    branch_index = *next(branch_index_set.begin(), 0);

    auto value_index = make_shared<v0::Constant>(element::i32, Shape{}, input_ind);
    result_output = make_shared<v0::Result>(input_value);
    result_value_index = make_shared<v0::Result>(value_index);
    input_value.remove_target_input(merge_input);
}

}  // namespace

bool pass::SwitchMergeResolver::run_on_model(const shared_ptr<Model>& m) {
    // run this transformation recursively since this is a model pass
    for (const auto& op : m->get_ordered_ops()) {
        auto multisubgraph_op = as_type_ptr<ov::op::util::MultiSubGraphOp>(op);
        if (multisubgraph_op) {
            for (size_t i = 0; i < multisubgraph_op->get_internal_subgraphs_size(); ++i) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(i)));
            }
        }
    }

    // split set of Switch and Merge nodes to clusters
    // where each cluster of Switch and Merge nodes will represent
    // the single If operation for fusing
    unordered_map<uint32_t, unordered_set<shared_ptr<Switch>>> switch_clusters;
    unordered_map<uint32_t, unordered_set<shared_ptr<Merge>>> merge_clusters;
    generate_if_clusters(m, switch_clusters, merge_clusters);

    // fuse Switch-Merge sub-graphs into If operation
    for (const auto& marker_to_merge_nodes : merge_clusters) {
        const auto& cluster_id = marker_to_merge_nodes.first;
        const auto& merge_nodes = marker_to_merge_nodes.second;
        const auto& switch_nodes = switch_clusters[cluster_id];
        if (merge_nodes.size() == 0 || switch_nodes.size() == 0) {
            continue;
        }

        auto cond = (*(switch_nodes.begin()))->input_value(1);

        // collect Parameter nodes and Result nodes for then and else bodies
        // set inputs and outputs for If node
        // create then bodies for which condition is true
        ParameterVector then_params, else_params;
        ResultVector then_results, else_results;
        OutputVector if_inputs, if_outputs;
        vector<unordered_set<string>> if_outputs_names;

        for (const auto& switch_node : switch_nodes) {
            if_inputs.push_back(switch_node->input_value(0));
            auto parameter_node_false = replace_switch_output_with_parameter(switch_node, 0);
            auto parameter_node_true = replace_switch_output_with_parameter(switch_node, 1);
            then_params.push_back(parameter_node_true);
            else_params.push_back(parameter_node_false);
        }

        CfMarkerType if_cf_marker;
        for (const auto& merge_node : merge_nodes) {
            // combine conditional markers from all Merge nodes
            // from which it results If node
            const auto& merge_cf_marker = get_cf_marker(merge_node);
            copy_conditional_flow_marker(merge_cf_marker, if_cf_marker);

            for (size_t input_ind = 0; input_ind < merge_node->get_input_size(); ++input_ind) {
                uint32_t branch_index = 0;
                shared_ptr<v0::Result> result_output, result_value_index;
                insert_result_before_merge(merge_node, input_ind, branch_index, result_output, result_value_index);
                FRONT_END_GENERAL_CHECK(
                    branch_index == 0 || branch_index == 1,
                    "[TensorFlow Frontend] internal error: conditional branch with unexpected index");
                if (branch_index == 0) {
                    // handle else-branch since the first output of Switch is output_false
                    else_results.push_back(result_output);
                    else_results.push_back(result_value_index);
                } else if (branch_index == 1) {
                    // handle then-branch since the second output of Switch is output_true
                    then_results.push_back(result_output);
                    then_results.push_back(result_value_index);
                }
            }
            for (const auto& merge_output : merge_node->outputs()) {
                if_outputs.push_back(merge_output);
                if_outputs_names.push_back(merge_output.get_names());
            }
        }

        // create body graphs for then and else branches
        auto then_body = make_shared<Model>(then_results, then_params);
        auto else_body = make_shared<Model>(else_results, else_params);

        auto if_op = make_shared<v8::If>(cond);
        // in case TensorFlow models, we can deduce predicate shape that must be a scalar
        if_op->get_rt_info()["tf_switch_merge_if"] = true;

        set_cf_marker(if_cf_marker, if_op);
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
