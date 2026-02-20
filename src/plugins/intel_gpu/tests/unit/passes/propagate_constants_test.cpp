// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "pass_manager.h"

using namespace cldnn;
using namespace ::tests;

// Verifies that when a non-trivial constant with dynamic layout is propagated,
// its downstream consumer transitions from dynamic to static and gets a proper
// static implementation selected via the need_impl_reselection mechanism.
//
// Topology:
//   input_layout("input", static) ---> eltwise("eltwise", sum)
//   data("weights") --> reorder("w_reorder") --/
//
// Before propagation: w_reorder layout is set to dynamic, making eltwise dynamic
// during compile_graph (dynamic impl assigned).
// After propagation: w_reorder is folded into a static data node (was_dynamic=true),
// triggering impl reselection. Eltwise inputs are now all static, so it receives
// a static implementation.
TEST(propagate_constants, need_impl_reselection_dynamic_to_static_transition) {
    auto& engine = get_test_engine();

    auto static_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};

    topology topology(
        input_layout("input", static_layout),
        data("weights", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        reorder("w_reorder", input_info("weights"), format::bfyx, data_types::f32),
        eltwise("eltwise", input_info("input"), input_info("w_reorder"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Simulate unresolved dynamic shape on the constant reorder node.
    // This causes downstream eltwise to become dynamic during compile_graph.
    auto& reorder_node = prog->get_node("w_reorder");
    auto dyn_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    reorder_node.set_output_layout(dyn_layout, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    // After compile_graph: eltwise should have a dynamic impl (or nullptr) since one input is dynamic.
    auto& eltwise_node = prog->get_node("eltwise");
    auto impl_before = eltwise_node.get_selected_impl();
    ASSERT_TRUE(impl_before == nullptr || impl_before->is_dynamic());

    // propagate_constants folds w_reorder into static data.
    // was_dynamic=true triggers need_impl_reselection.
    // Eltwise transitions to static and should receive a static implementation.
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    // The original w_reorder should be replaced by a propagated data node
    // (program::replace renames the new node to the old id).
    auto& w_reorder_replaced = prog->get_node("w_reorder");
    ASSERT_TRUE(w_reorder_replaced.is_type<data>());

    auto impl_after = eltwise_node.get_selected_impl();
    ASSERT_NE(impl_after, nullptr);
    ASSERT_FALSE(impl_after->is_dynamic());
}

// Verifies that when need_impl_reselection is triggered, nodes that remain dynamic
// (due to a dynamic non-constant input) are correctly skipped — they should NOT
// receive a static implementation.
//
// Topology:
//   input_layout("input_static", static)  ---> eltwise("eltwise_becomes_static", sum)
//   input_layout("input_dynamic", dynamic) --> eltwise("eltwise_stays_dynamic", sum)
//   data("weights") --> reorder("w_reorder") --/ (shared constant input for both)
//
// After propagation: eltwise_becomes_static transitions to static (gets static impl),
// while eltwise_stays_dynamic remains dynamic (keeps dynamic impl or nullptr).
TEST(propagate_constants, need_impl_reselection_skips_still_dynamic_nodes) {
    auto& engine = get_test_engine();

    auto static_input_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};

    topology topology(
        input_layout("input_static", static_input_layout),
        input_layout("input_dynamic", dynamic_input_layout),
        data("weights", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        reorder("w_reorder", input_info("weights"), format::bfyx, data_types::f32),
        eltwise("eltwise_becomes_static", input_info("input_static"), input_info("w_reorder"), eltwise_mode::sum),
        eltwise("eltwise_stays_dynamic", input_info("input_dynamic"), input_info("w_reorder"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Set constant reorder to dynamic layout to trigger need_impl_reselection.
    auto& reorder_node = prog->get_node("w_reorder");
    auto dyn_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    reorder_node.set_output_layout(dyn_layout, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    // Both eltwises should be dynamic after compile_graph (one dynamic input each).
    auto& static_eltwise = prog->get_node("eltwise_becomes_static");
    auto& dynamic_eltwise = prog->get_node("eltwise_stays_dynamic");

    auto static_impl_before = static_eltwise.get_selected_impl();
    auto dynamic_impl_before = dynamic_eltwise.get_selected_impl();
    ASSERT_TRUE(static_impl_before == nullptr || static_impl_before->is_dynamic());
    ASSERT_TRUE(dynamic_impl_before == nullptr || dynamic_impl_before->is_dynamic());

    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    // eltwise_becomes_static: all inputs now static → should get static impl.
    auto static_impl_after = static_eltwise.get_selected_impl();
    ASSERT_NE(static_impl_after, nullptr);
    ASSERT_FALSE(static_impl_after->is_dynamic());

    // eltwise_stays_dynamic: input_dynamic is still dynamic → should keep dynamic
    // impl or remain without impl (both are valid for dynamic nodes).
    auto dynamic_impl_after = dynamic_eltwise.get_selected_impl();
    ASSERT_TRUE(dynamic_impl_after == nullptr || dynamic_impl_after->is_dynamic());
}

// Verifies that when all propagated constants are already static (was_dynamic=false),
// need_impl_reselection is NOT triggered and existing implementations are preserved.
//
// Topology (same structure, but w_reorder keeps its static layout):
//   input_layout("input", static) ---> eltwise("eltwise", sum)
//   data("weights") --> reorder("w_reorder") --/
//
// Since w_reorder is static before propagation, was_dynamic=false.
// The pass returns early without entering the impl reselection loop.
TEST(propagate_constants, no_reselection_when_constants_are_static) {
    auto& engine = get_test_engine();

    auto static_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};

    topology topology(
        input_layout("input", static_layout),
        data("weights", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        reorder("w_reorder", input_info("weights"), format::bfyx, data_types::f32),
        eltwise("eltwise", input_info("input"), input_info("w_reorder"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Do NOT set w_reorder to dynamic — it stays static.
    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    auto& eltwise_node = prog->get_node("eltwise");
    auto impl_before = eltwise_node.get_selected_impl();
    ASSERT_NE(impl_before, nullptr);
    ASSERT_FALSE(impl_before->is_dynamic());

    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    // w_reorder should be folded into a data node.
    auto& w_reorder_replaced = prog->get_node("w_reorder");
    ASSERT_TRUE(w_reorder_replaced.is_type<data>());

    // Eltwise's impl should be preserved (was_dynamic=false, no reselection triggered).
    auto impl_after = eltwise_node.get_selected_impl();
    ASSERT_NE(impl_after, nullptr);
    ASSERT_FALSE(impl_after->is_dynamic());
}
