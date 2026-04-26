// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "convolution_inst.h"
#include "reshape_inst.h"
#include "reorder_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"
#include "program_helpers.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

// Regression test for memory pool corruption when a fused eltwise-sum (oneDNN
// post-op) consumes its residual input through an in-place chain of
// optimized-out reshapes/reorders.
//
// Subgraph under test (residual connection through reshape):
//
//   input ─► conv1 ─► reshape (optimized-out, in-place) ──┐
//                                                          │ eltwise sum
//   input ─► conv2 ─────────────────────────────────► conv3 ─► reorder
//
// After prepare_primitive_fusing, the eltwise gets fused into conv3 as a
// sum post-op.  prepare_buffer_fusing then marks the reshape as
// can_be_optimized (in-place passthrough of conv1's buffer).
//
// BUG (before fix): basic_memory_dependencies marks reshape and conv3 with
// can_share_buffer(false), but conv1 — the actual buffer owner in the pool —
// is left shareable.  The pool can then hand conv1's buffer to an unrelated
// primitive, which overwrites it while conv3 still needs to read the
// residual from it.
//
// FIX: basic_memory_dependencies now walks up the in-place chain from the
// eltwise dependency to the root buffer owner and marks it non-shareable.
//
// This test verifies the root owner (conv1) has can_share_buffer() == false
// after running the pass.

TEST(basic_memory_dependencies, root_buffer_owner_marked_non_shareable_through_inplace_chain) {
    auto& engine = get_test_engine();

    // Need oneDNN support for the fused sum post-op path
    if (!engine.get_device_info().supports_immad)
        return;

    // Use 1x1 convolutions so that shapes stay compatible for the eltwise sum.
    // conv1 output = [1, 16, 32, 32], reshape keeps the same shape (nop reshape),
    // conv2/conv3 output = [1, 16, 32, 32] => eltwise sum is valid.
    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight = engine.allocate_memory(layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));

    // Left branch:  input → conv1 → reshape (nop, will be optimized-out)
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reshape("reshape", input_info("conv1"), tensor(1, 16, 32, 32)));

    // Right branch: input → conv2 → conv3
    topology.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    // Residual: conv3 + reshape → reorder (output)
    topology.add(eltwise("eltwise", input_info("conv3"), input_info("reshape"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    // Build program and run the optimization pipeline up through basic_memory_dependencies.
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);

    // The full pre_optimize_graph + post_optimize_graph pipeline runs during build_program
    // with optimize_data=true, which includes:
    //   prepare_primitive_fusing  (fuses eltwise into conv3)
    //   prepare_buffer_fusing     (marks reshape as can_be_optimized)
    //   add_onednn_optimization_attributes

    // Now run memory dependency analysis (normally called from run_graph_compilation).
    program_wrapper::prepare_memory_dependencies(*prog);

    // Verify: After the fix, the eltwise should be fused into conv3 as a sum post-op.
    // The eltwise node should be removed from the graph.
    ASSERT_FALSE(has_node(*prog, "eltwise"));

    auto& conv3_node = prog->get_node("conv3");
    auto& fused_prims = conv3_node.get_fused_primitives();
    bool has_sum_post_op = false;
    for (auto& fused_op : fused_prims) {
        if (fused_op.is_type<eltwise>()) {
            auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(conv3_node, fused_op);
            if (fusing_type == add_fusing_type::sum) {
                has_sum_post_op = true;
                break;
            }
        }
    }

    // If the eltwise didn't get fused as sum post-op, this topology doesn't
    // reproduce the bug scenario on this hardware — skip rather than fail.
    if (!has_sum_post_op)
        return;

    // The reshape should be optimized-out (in-place), inheriting conv1's buffer.
    if (has_node(*prog, "reshape")) {
        auto& reshape_node = prog->get_node("reshape");
        // If reshape is present but not optimized, the in-place chain doesn't form
        // and the bug scenario doesn't apply.
        if (!reshape_node.can_be_optimized())
            return;
    }

    // CRITICAL CHECK: conv1 (the root buffer owner) must be marked non-shareable.
    // Before the fix, only reshape and conv3 were marked, leaving conv1's
    // buffer available for reuse by other primitives.
    auto& conv1_node = prog->get_node("conv1");
    ASSERT_FALSE(conv1_node.can_share_buffer())
        << "conv1 (root buffer owner behind in-place reshape) must have "
           "can_share_buffer() == false to prevent memory pool corruption";

    // Also verify that conv3 itself is non-shareable (existing behavior).
    ASSERT_FALSE(conv3_node.can_share_buffer())
        << "conv3 (node with fused sum post-op) must have "
           "can_share_buffer() == false";
}

// Variant: verify that end-to-end inference produces correct results when
// the residual connection passes through an optimized-out reshape.
// Compares GPU (with memory reuse) against GPU (without memory reuse).
TEST(basic_memory_dependencies, inplace_chain_eltwise_sum_correctness) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);

    tests::random_generator rg;
    rg.set_seed("basic_memory_dependencies_correctness");
    {
        auto rnd = rg.generate_random_1d<ov::float16>(weight_layout.count(), -1, 1);
        set_values(weight_mem, rnd);
    }

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight_mem));
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reshape("reshape", input_info("conv1"), tensor(1, 16, 32, 32)));
    topology.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("conv3"), input_info("reshape"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));

    auto input_mem = engine.allocate_memory(in_layout);
    {
        auto rnd = rg.generate_random_1d<ov::float16>(in_layout.count(), -1, 1);
        set_values(input_mem, rnd);
    }

    // Reference: run with memory reuse disabled
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_static_input_reorder(true));
    network net_ref(engine, topology, config_ref);
    net_ref.set_input_data("input", input_mem);
    auto outputs_ref = net_ref.execute();

    // Test: run with full optimization (memory reuse enabled)
    ExecutionConfig config_opt = get_test_default_config(engine);
    config_opt.set_property(ov::intel_gpu::optimize_data(true));
    network net_opt(engine, topology, config_opt);
    net_opt.set_input_data("input", input_mem);
    auto outputs_opt = net_opt.execute();

    auto out_ref = outputs_ref.at("reorder").get_memory();
    auto out_opt = outputs_opt.at("reorder").get_memory();

    ASSERT_NE(out_ref, nullptr);
    ASSERT_NE(out_opt, nullptr);
    ASSERT_EQ(out_ref->count(), out_opt->count());

    cldnn::mem_lock<float> ref_ptr(out_ref, get_test_stream());
    cldnn::mem_lock<float> opt_ptr(out_opt, get_test_stream());

    // Tolerance accounts for f16 precision differences between optimized and
    // non-optimized paths (different impl types, accumulation order).
    // Corruption typically produces errors orders of magnitude larger.
    const float tolerance = 1.0f;
    for (size_t i = 0; i < out_ref->count(); i++) {
        ASSERT_NEAR(ref_ptr[i], opt_ptr[i], tolerance)
            << "Mismatch at index " << i
            << ": ref=" << ref_ptr[i] << " opt=" << opt_ptr[i]
            << "\nThis may indicate memory pool corruption from buffer reuse "
               "through an in-place chain";
    }
}
