// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "pass_manager.h"
#include <intel_gpu/primitives/convolution.hpp>
#include "openvino/reference/convolution.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace cldnn;
using namespace ::tests;

// Verifies that when constant propagation transforms a dynamic-layout constant node into
// static data, downstream consumers receive updated static implementations.
//
// In real models (e.g., SSM with dynamic shapes), shape inference may leave intermediate
// constant computation nodes with unresolved dynamic layouts during compile_graph.
// After propagate_constants evaluates and folds these constant subgraphs, the resulting
// data nodes have fully resolved static shapes, causing downstream consumers to transition
// from dynamic to static.
//
// This test uses a non-trivial constant subgraph (eltwise of two weight tensors) whose
// layout is set to dynamic to simulate unresolved shape inference. Creating a genuinely
// dynamic constant subgraph requires model-specific conditions that are impractical to
// reproduce in a unit test, so the manual override is necessary.
//
// Topology:
//   data("weights_a") ---+
//                         eltwise("w_sum") --> eltwise("eltwise", sum)
//   data("weights_b") ---+                   /
//   input_layout("input", static) ---------/
//
// After propagation: w_sum is folded into a static data node (was_dynamic=true),
// triggering impl reselection for eltwise which transitions to static.
TEST(propagate_constants, need_impl_reselection_dynamic_to_static_transition) {
    auto& engine = get_test_engine();

    auto static_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};

    topology topology(
        input_layout("input", static_layout),
        data("weights_a", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        data("weights_b", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        eltwise("w_sum", input_info("weights_a"), input_info("weights_b"), eltwise_mode::sum),
        eltwise("eltwise", input_info("input"), input_info("w_sum"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Simulate unresolved dynamic shape on the constant computation node.
    // This models real scenarios where shape inference cannot fully resolve
    // intermediate constant shapes during compile_graph.
    auto& w_sum_node = prog->get_node("w_sum");
    auto dyn_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    w_sum_node.set_output_layout(dyn_layout, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    // After compile_graph: eltwise should have a dynamic impl since one input is dynamic.
    auto& eltwise_node = prog->get_node("eltwise");
    auto impl_before = eltwise_node.get_selected_impl();
    ASSERT_TRUE(impl_before == nullptr || impl_before->is_dynamic());

    // propagate_constants evaluates and folds w_sum into static data.
    // Dynamic -> static transition triggers impl reselection for eltwise.
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    // w_sum should be replaced by a propagated data node
    // (program::replace renames the new node to the old id).
    auto& w_sum_replaced = prog->get_node("w_sum");
    ASSERT_TRUE(w_sum_replaced.is_type<data>());

    auto impl_after = eltwise_node.get_selected_impl();
    ASSERT_NE(impl_after, nullptr);
    ASSERT_FALSE(impl_after->is_dynamic());
}

// Verifies that when impl reselection is triggered, nodes that remain dynamic
// (due to a dynamic non-constant input) are correctly skipped — they should NOT
// receive a static implementation.
//
// Uses a non-trivial constant subgraph (eltwise of two weight tensors) shared by
// two consumers: one with a static input and one with a dynamic input.
//
// Topology:
//   input_layout("input_static", static)  ---> eltwise("eltwise_becomes_static", sum)
//   input_layout("input_dynamic", dynamic) --> eltwise("eltwise_stays_dynamic", sum)
//   data("weights_a") ---+                    /
//                         eltwise("w_sum") --/  (shared constant input for both)
//   data("weights_b") ---+
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
        data("weights_a", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        data("weights_b", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        eltwise("w_sum", input_info("weights_a"), input_info("weights_b"), eltwise_mode::sum),
        eltwise("eltwise_becomes_static", input_info("input_static"), input_info("w_sum"), eltwise_mode::sum),
        eltwise("eltwise_stays_dynamic", input_info("input_dynamic"), input_info("w_sum"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Simulate unresolved dynamic shape on the constant computation node.
    auto& w_sum_node = prog->get_node("w_sum");
    auto dyn_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    w_sum_node.set_output_layout(dyn_layout, true);

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
    ASSERT_EQ(impl_before, impl_after);
}

TEST(propagate_constants, constant_padded_conv_chain_matches_fp64_with_runtime_onednn_conv) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN convolution path requires IMMAD support";

    const ov::Shape input_shape{1, 8, 16, 16};
    const ov::Shape first_weights_shape{16, 8, 3, 3};
    const ov::Shape second_weights_shape{16, 16, 3, 3};
    const ov::Shape output_shape{1, 16, 16, 16};
    const layout input_tensor_layout{input_shape, data_types::f32, format::bfyx};
    const layout first_weights_layout{first_weights_shape, data_types::f32, format::bfyx};
    const layout second_weights_layout{second_weights_shape, data_types::f32, format::bfyx};

    auto make_weights = [](size_t count, unsigned int seed) {
        std::mt19937 generator(seed);
        std::normal_distribution<float> distribution(0.0f, 0.1f);
        std::vector<float> values(count);
        for (auto& value : values)
            value = distribution(generator);
        return values;
    };
    auto to_fp64 = [](const std::vector<float>& values) {
        return std::vector<double>(values.begin(), values.end());
    };

    const auto first_weights_values = make_weights(first_weights_layout.count(), 1);
    const auto second_weights_values = make_weights(second_weights_layout.count(), 2);
    const auto runtime_weights_values = make_weights(first_weights_layout.count(), 99);

    auto constant_input = engine.allocate_memory(input_tensor_layout);
    auto runtime_input = engine.allocate_memory(input_tensor_layout);
    auto first_weights = engine.allocate_memory(first_weights_layout);
    auto second_weights = engine.allocate_memory(second_weights_layout);
    auto runtime_weights = engine.allocate_memory(first_weights_layout);
    set_values(constant_input, std::vector<float>(input_tensor_layout.count(), 0.25f));
    set_values(runtime_input, std::vector<float>(input_tensor_layout.count(), 0.25f));
    set_values(first_weights, first_weights_values);
    set_values(second_weights, second_weights_values);
    set_values(runtime_weights, runtime_weights_values);

    // runtime_conv is intentionally on its own branch (fed by the non-constant "runtime_input"), not downstream of
    // constant_conv_2: it must stay outside propagate_constants' internal program so this test also covers that the
    // internal-program's cloned force_implementations map is filtered to the internal graph's own node ids.
    topology topology(
        input_layout("runtime_input", input_tensor_layout),
        data("runtime_weights", runtime_weights),
        convolution("runtime_conv", input_info("runtime_input"), "runtime_weights", "", 1,
                    {1, 1}, {1, 1}, {1, 1}, {1, 1}, false),
        data("constant_input", constant_input),
        data("first_weights", first_weights),
        data("second_weights", second_weights),
        convolution("constant_conv_1", input_info("constant_input"), "first_weights", "", 1,
                    {1, 1}, {1, 1}, {1, 1}, {1, 1}, false),
        convolution("constant_conv_2", input_info("constant_conv_1"), "second_weights", "", 1,
                    {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    const ov::intel_gpu::ImplementationDesc runtime_conv_impl{format::any, "", impl_types::onednn};
    // constant_conv_1 also carries a forcing entry whose impl_type is left at its default `any` (as a kernel- or
    // format-only forcing entry would be), covering the case where propagate_constants must keep its own backend
    // pin instead of letting graph_initializations reset it back to `any`.
    const ov::intel_gpu::ImplementationDesc constant_conv_format_only{};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"runtime_conv", runtime_conv_impl}, {"constant_conv_1", constant_conv_format_only}}));

    network network(engine, topology, config);
    // Guard against a false-positive pass if implementation forcing were ignored/fell back for this configuration.
    auto runtime_conv_impl_inst = network.get_primitive("runtime_conv")->get_impl();
    ASSERT_NE(runtime_conv_impl_inst, nullptr);
    ASSERT_TRUE(runtime_conv_impl_inst->is_onednn()) << "runtime_conv was expected to run with oneDNN due to implementation forcing";

    network.set_input_data("runtime_input", runtime_input);
    auto outputs = network.execute();
    ASSERT_NE(outputs.find("runtime_conv"), outputs.end());
    ASSERT_NE(outputs.find("constant_conv_2"), outputs.end());

    const auto first_weights_fp64 = to_fp64(first_weights_values);
    const auto second_weights_fp64 = to_fp64(second_weights_values);
    const std::vector<double> input_values_fp64(input_tensor_layout.count(), 0.25);
    std::vector<double> first_output_fp64(ov::shape_size(output_shape));
    std::vector<double> expected_output_fp64(ov::shape_size(output_shape));
    ov::reference::convolution(input_values_fp64.data(), first_weights_fp64.data(), first_output_fp64.data(),
                               input_shape, first_weights_shape, output_shape,
                               {1, 1}, {1, 1}, {1, 1}, {1, 1});
    ov::reference::convolution(first_output_fp64.data(), second_weights_fp64.data(), expected_output_fp64.data(),
                               output_shape, second_weights_shape, output_shape,
                               {1, 1}, {1, 1}, {1, 1}, {1, 1});

    auto output_memory = outputs.at("constant_conv_2").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_values(output_memory, get_test_stream());
    ASSERT_EQ(output_values.size(), expected_output_fp64.size());
    double max_abs_error = 0.0;
    for (size_t i = 0; i < expected_output_fp64.size(); ++i)
        max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(output_values[i]) - expected_output_fp64[i]));

    EXPECT_LE(max_abs_error, 1e-4) << "constant padded convolution chain max abs error: " << max_abs_error;
}

// Verifies that propagate_constants consistently reselects implementations
// for ALL affected nodes after a dynamic→static constant transition, even
// when processing earlier nodes causes side effects that mark later nodes'
// output layouts as valid.
//
// Regression test for: processing node A in reselection_targets triggers
// calc_output_layouts on its dependencies, which transitively calls
// set_output_layout on node B (also in reselection_targets), setting
// B's valid_output_layout = true. Without the fix, the
// is_valid_output_layout() guard in try_reselect_impl_for_node causes
// node B to skip reselection entirely — keeping its dynamic impl while
// node A gets a static impl. This inconsistency leads to kernel cache
// collisions and CL_INVALID_KERNEL_ARGS at runtime.
//
// Topology:
//   input_layout("input_1", static) --> eltwise("eltwise_1", sum) --+
//                                                                    |
//   data("weights_a") ---+                                           +-- shared constant
//                         eltwise("w_sum") --------------------------+
//   data("weights_b") ---+                                           |
//                                                                    |
//   input_layout("input_2", static) --> eltwise("eltwise_2", sum) --+
//
// Both eltwises share the same dynamic constant (w_sum) and have static
// inputs with identical layouts. After propagate_constants replaces w_sum
// with static data, both nodes enter reselection_targets and both must
// be reselected to static impl.
TEST(propagate_constants, all_reselection_targets_get_consistent_static_impl) {
    auto& engine = get_test_engine();

    const auto static_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};

    topology topology(
        input_layout("input_1", static_layout),
        input_layout("input_2", static_layout),
        data("weights_a", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        data("weights_b", engine.allocate_memory(layout{{1, 3, 4, 4}, data_types::f32, format::bfyx})),
        eltwise("w_sum", input_info("weights_a"), input_info("weights_b"), eltwise_mode::sum),
        eltwise("eltwise_1", input_info("input_1"), input_info("w_sum"), eltwise_mode::sum),
        eltwise("eltwise_2", input_info("input_2"), input_info("w_sum"), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    // Simulate unresolved dynamic shape on the constant computation node.
    auto& w_sum_node = prog->get_node("w_sum");
    auto dyn_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    w_sum_node.set_output_layout(dyn_layout, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    // After compile_graph both eltwises should have dynamic impl
    // (one of their inputs — w_sum — has a dynamic layout).
    auto& eltwise_1 = prog->get_node("eltwise_1");
    auto& eltwise_2 = prog->get_node("eltwise_2");

    ASSERT_TRUE(eltwise_1.get_selected_impl() == nullptr ||
                eltwise_1.get_selected_impl()->is_dynamic());
    ASSERT_TRUE(eltwise_2.get_selected_impl() == nullptr ||
                eltwise_2.get_selected_impl()->is_dynamic());

    // propagate_constants replaces w_sum with static data.
    // Both eltwise_1 and eltwise_2 are in reselection_targets.
    // The fix ensures both are checked for reselection regardless of
    // stale is_valid_output_layout state caused by processing order.
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    // Both eltwises must be consistently reselected to static impl.
    auto impl_1 = eltwise_1.get_selected_impl();
    auto impl_2 = eltwise_2.get_selected_impl();
    ASSERT_NE(impl_1, nullptr);
    ASSERT_NE(impl_2, nullptr);
    ASSERT_FALSE(impl_1->is_dynamic())
        << "eltwise_1 should have been reselected to static impl";
    ASSERT_FALSE(impl_2->is_dynamic())
        << "eltwise_2 should have been reselected to static impl";
}
