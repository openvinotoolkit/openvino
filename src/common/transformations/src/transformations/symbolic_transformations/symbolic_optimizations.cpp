// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/shared_ops_optimization.hpp>
#include <transformations/symbolic_transformations/chained_maximum.hpp>
#include <transformations/symbolic_transformations/dereshape_matmul.hpp>
#include <transformations/symbolic_transformations/label_optimization.hpp>
#include <transformations/symbolic_transformations/nop_broadcast.hpp>
#include <transformations/symbolic_transformations/utils.hpp>

#include "itt.hpp"

void symbolic_set_up_for_shape(ov::DimensionTracker& dt, ov::PartialShape& shape) {
    if (shape.rank().is_dynamic())
        return;
    for (auto& d : shape) {
        bool is_static = d.is_static(), has_label = ov::DimensionTracker::has_label(d);
        if (is_static && has_label)
            dt.reset_tracking_info(d);  // remove labels from static dims on shapes to reduce label clutter
        if (is_static || has_label)
            continue;
        dt.set_up_for_tracking(d);
    }
}

void special_case_range_label_propagation(const std::shared_ptr<ov::Node>& node) {
    /* Label propagation through specific Range operation
          start    shift
            |  \   /
            |   Add   step == 1
            \    /    /
               Range
    */
    if (!ov::is_type<ov::op::v0::Range>(node) && !ov::is_type<ov::op::v4::Range>(node))
        return;

    auto output_shape = node->get_output_partial_shape(0);
    if (output_shape.rank().is_dynamic() || output_shape.size() != 1)
        return;

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto step_value = ov::get_constant_from_source(node->input_value(2));
    OPENVINO_SUPPRESS_DEPRECATED_END
    if (!step_value || step_value->cast_vector<int64_t>()[0] != 1)
        return;

    auto start_labels = node->get_input_tensor(0).get_value_label();
    if (start_labels.size() != 1 || start_labels[0] == ov::no_label)
        return;
    auto start_label = start_labels[0];

    auto stop_node = node->input_value(1).get_node_shared_ptr();
    if (!ov::is_type<ov::op::v1::Add>(stop_node))
        return;
    auto add_in0_labels = stop_node->get_input_tensor(0).get_value_label();
    if (add_in0_labels.size() != 1 || add_in0_labels[0] == ov::no_label)
        return;
    auto add_in0_label = add_in0_labels[0];

    auto add_in1_labels = stop_node->get_input_tensor(1).get_value_label();
    if (add_in1_labels.size() != 1 || add_in1_labels[0] == ov::no_label)
        return;
    auto add_in1_label = add_in1_labels[0];

    if (add_in0_label == start_label)
        ov::DimensionTracker::set_label(output_shape[0], add_in1_label);
    else if (add_in1_label == start_label)
        ov::DimensionTracker::set_label(output_shape[0], add_in0_label);
    node->set_output_type(0, node->get_output_element_type(0), output_shape);
}

ov::pass::SymbolicPropagation::SymbolicPropagation() {
    m_te = std::make_shared<ov::TableOfEquivalence>();
}

bool ov::pass::SymbolicPropagation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(SymbolicPropagation);

    auto te = m_te;
    m->get_rt_info()["TABLE_OF_EQUIVALENCE"] = te;
    ov::DimensionTracker dt(te);

    for (const auto& op : m->get_ordered_ops()) {
        op->invalidate_values();
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op))
            for (const auto& sub_graph : multi_subgraph_op->get_functions())
                if (sub_graph)
                    run_on_model(sub_graph);
        for (auto& output : op->outputs()) {
            output.get_rt_info()["SKIP_INVALIDATION"] = true;
            output.get_rt_info()["TABLE_OF_EQUIVALENCE"] = te;
        }
        op->validate_and_infer_types();

        // additional label propagation rules must be triggered here
        special_case_range_label_propagation(op);
        // additional label propagation rules must be triggered here

        for (auto& output : op->outputs()) {
            auto shape = output.get_partial_shape();
            symbolic_set_up_for_shape(dt, shape);
            OPENVINO_SUPPRESS_DEPRECATED_START
            output.get_tensor().set_tensor_type(output.get_element_type(), shape);
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
    }
    return true;
}

ov::pass::LabelResolvingThroughSelect::LabelResolvingThroughSelect() {
    MATCHER_SCOPE(LabelResolvingThroughSelect);
    auto input_reshape = pattern::wrap_type<op::v1::Reshape>();
    auto select = pattern::wrap_type<op::v1::Select>({pattern::any_input(), pattern::any_input(), input_reshape});
    auto softmax = pattern::wrap_type<op::v1::Softmax>({select});  // axis?
    auto reshape = pattern::wrap_type<op::v1::Reshape>({softmax, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& value_map = m.get_pattern_value_map();
        auto reshape_0 = value_map.at(input_reshape).get_node_shared_ptr();
        auto reshape_1 = value_map.at(reshape).get_node_shared_ptr();
        if (reshape_keeps_last_two_dims(reshape_1))
            return false;  // reshape doesn't need optimization
        if (!last_two_dims_are_equal(reshape_0->get_output_partial_shape(0), reshape_1->get_output_partial_shape(0)))
            return false;
        // we established that data from input_reshape wasn't broadcasted via Select
        auto select_output = value_map.at(select);
        auto select_output_pshape = select_output.get_partial_shape();
        const auto& reshape_pshape = reshape_1->get_output_partial_shape(0);
        if (!equalize_two_last_dims(reshape_pshape, select_output_pshape))
            return false;
        select_output.get_node_shared_ptr()->set_output_type(0, select_output.get_element_type(), select_output_pshape);
        value_map.at(softmax).get_node_shared_ptr()->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

bool ov::pass::SymbolicOptimizations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SymbolicOptimizations);
    std::cout << "# nodes before: " << m->get_ops().size();
    ov::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);

    REGISTER_PASS(manager, SymbolicPropagation)

    // symbolic based transformations allowing for better static dimension propagation
    REGISTER_PASS(manager, ChainedMaximumOptimization)
    REGISTER_PASS(manager, NopBroadcast)

    // regular transformations which are needed right now since they are
    REGISTER_PASS(manager, NopElimination)        // Broadcast (Tile) Ones + Remove Slice Before GatherElements
    REGISTER_PASS(manager, SharedOpOptimization)  // Shared GatherElements

    // transformations which use labels for optimizations
    REGISTER_PASS(manager,
                  LabelResolvingThroughSelect)  // helps to figure out that broadcasting didn't happen through Select op
    REGISTER_PASS(manager, DeReshapeMatMul)  // should become one transformation with DeReshapeMatMulWithComplications
    REGISTER_PASS(manager, DeReshapeMatMulWithComplications)  // should become one transformation with DeReshapeMatMul

    REGISTER_PASS(manager, ApplyTableOfEquivalence)
    REGISTER_PASS(manager, OptimizeLabelsUsedAsValues)

    REGISTER_PASS(manager, NopElimination)
    REGISTER_PASS(manager, SharedOpOptimization)

//    REGISTER_PASS(manager, RPE_Optimization)  // should be called after SymbolicOptimizations in plugin

    // cleanup labels, erase SKIP_INVALIDATION
    manager.run_passes(m);
    std::cout << " after: " << m->get_ops().size() << std::endl;
    return true;  // cleans up all the label information
}
