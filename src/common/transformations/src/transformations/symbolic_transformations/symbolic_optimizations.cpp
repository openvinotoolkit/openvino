// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/symbolic_info.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/dimension_tracking.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/shared_ops_optimization.hpp>
#include <transformations/symbolic_transformations/chained_maximum.hpp>
#include <transformations/symbolic_transformations/label_optimization.hpp>
#include <transformations/symbolic_transformations/nop_broadcast.hpp>

#include "itt.hpp"

namespace {
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
}  // namespace

ov::pass::SymbolicPropagation::SymbolicPropagation() {
    m_te = std::make_shared<ov::TableOfEquivalence>();
}

bool ov::pass::SymbolicPropagation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(SymbolicPropagation);

    auto te = m_te;
    ov::set_up_symbolic_info(m, te);
    ov::DimensionTracker dt(te);

    for (const auto& op : m->get_ordered_ops()) {
        // since we disable invalidation with the following two lines, we have to invalidate manually here
        op->invalidate_values();
        for (auto& output : op->outputs())
            ov::set_up_symbolic_info(output, te);
        op->revalidate_and_infer_types();
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op))
            for (const auto& sub_graph : multi_subgraph_op->get_functions())
                if (sub_graph)
                    run_on_model(sub_graph);

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

ov::pass::SymbolicOptimizations::SymbolicOptimizations(bool full_run) {
    m_manager = std::make_shared<pass::Manager>();
    m_manager->set_per_pass_validation(false);

#define REGISTER_SYMBOLIC(region, ...) m_manager->register_pass<region>(__VA_ARGS__);

    REGISTER_SYMBOLIC(SymbolicPropagation)
    if (full_run) {
        // symbolic based transformations allowing for better static dimension propagation
        REGISTER_SYMBOLIC(ChainedMaximumOptimization)
        REGISTER_SYMBOLIC(NopBroadcast)
        // regular transformations which are needed right now since they clean up unnecessary operations
        REGISTER_SYMBOLIC(NopElimination)        // Broadcast (Tile) Ones + Remove Slice Before GatherElements
        REGISTER_SYMBOLIC(SharedOpOptimization)  // Shared GatherElements
    }
    // transformations which use labels for optimizations
    REGISTER_SYMBOLIC(ApplyTableOfEquivalence)
    if (full_run) {
        REGISTER_SYMBOLIC(OptimizeLabelsUsedAsValues)  // reduce shape sub-graphs
    }
}

bool ov::pass::SymbolicOptimizations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SymbolicOptimizations);
    m_manager->run_passes(m);
    ov::remove_symbolic_info(m);
    return true;
}
