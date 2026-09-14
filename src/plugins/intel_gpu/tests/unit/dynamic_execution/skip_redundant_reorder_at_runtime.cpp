// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/reshape.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_reorder_tests {
TEST(remove_redundant_reorder, skip_reorder_at_runtime) {
    auto& engine = get_test_engine();
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);

    network.set_input_data("input", input_mem);
    network.execute();
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    ASSERT_EQ(network.get_output_memory("reorder")->buffer_ptr(), network.get_primitive("fc")->output_memory_ptr()->buffer_ptr());
}

TEST(skip_reorder_at_runtime, not_reuse_remote_tensor) {
    auto& engine = get_test_engine();
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f16, format::bfyx});
    auto output_remote_mem = engine.allocate_memory({{10, 2}, data_types::f16, format::bfyx});
    std::vector<ov::float16> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f16, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f16),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f16, format::bfyx});
    std::vector<ov::float16> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);

    network.set_input_data("input", input_mem);
    network.set_output_memory("reorder", output_remote_mem, true);
    network.execute();

    // User-provided output memory must always be used as the network output
    ASSERT_EQ(output_remote_mem->buffer_ptr(), network.get_output_memory("reorder")->buffer_ptr());

    // Whether the reorder is optimized away depends on the memory allocation strategy.
    // On USM-capable devices, the runtime may allow FC to write directly into the
    // user-provided buffer, making the reorder a no-op. On cl_mem-only devices,
    // the reorder must execute to copy data from FC's buffer to the user's buffer.
    if (reorder_inst->can_be_optimized()) {
        ASSERT_EQ(network.get_output_memory("reorder")->buffer_ptr(),
                  network.get_primitive("fc")->output_memory_ptr()->buffer_ptr());
    } else {
        ASSERT_NE(network.get_output_memory("reorder")->buffer_ptr(),
                  network.get_primitive("fc")->output_memory_ptr()->buffer_ptr());
    }
}

TEST(skip_reorder_at_runtime, reuse_remote_tensor) {
    auto& engine = get_test_engine();
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    auto output_remote_mem = engine.allocate_memory({{16, 2}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);

    network.set_input_data("input", input_mem);
    network.set_output_memory("reorder", output_remote_mem, true);
    network.execute();
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    ASSERT_EQ(output_remote_mem->buffer_ptr(), network.get_output_memory("reorder")->buffer_ptr());
    ASSERT_EQ(network.get_output_memory("reorder")->buffer_ptr(), network.get_primitive("fc")->output_memory_ptr()->buffer_ptr());
}

// Unlike permute, a reorder's skip decision is already fresh before realloc (do_runtime_skip_reorder()
// is producer-driven), so it must never become a remote-output chain boundary: doing so would drop its
// producer out of the chain built by network::build_output_chain() and break PR #29061's propagation.
TEST(skip_reorder_at_runtime, remote_output_chain_does_not_stop_at_reorder) {
    auto& engine = get_test_engine();
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    ASSERT_FALSE(reorder_inst->is_remote_output_chain_boundary());

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);
    network.set_input_data("input", input_mem);

    auto output_remote_mem = engine.allocate_memory({{10, 2}, data_types::f32, format::bfyx});
    network.set_output_memory("reorder", output_remote_mem, true);
    network.execute();

    // The producer ("fc") must have received the remote buffer through the chain, not merely the
    // reorder itself -- this is the exact propagation PR #29061 relies on.
    ASSERT_EQ(network.get_primitive("fc")->output_memory_ptr()->buffer_ptr(), output_remote_mem->buffer_ptr());
    ASSERT_EQ(network.get_output_memory("reorder")->buffer_ptr(), output_remote_mem->buffer_ptr());
}

// The helper deliberately requires is_runtime_skippable() for the permute role only. A reorder may be
// can_be_optimized() without ever having runtime_skippable set (remove_redundant_reorders can mark a
// STATIC reorder optimized without going through mark_runtime_skippable_nodes at all). This test
// documents what was actually reachable through normal topology construction for a DYNAMIC output
// reorder feeding try_bind_remote_output_via_skippable_user(): mark_runtime_skippable_nodes is, in
// practice, the only pass that sets can_be_optimized(true) on a dynamic reorder, and it always pairs
// that with set_runtime_skippable(true) (see mark_runtime_skippable_nodes.cpp, do_for_types<reorder>).
// So for the topology below, both flags end up true together; is_runtime_skippable() is NOT observed
// false here. Left as an explicit, visible precondition check rather than silently assumed.
TEST(skip_reorder_at_runtime, can_be_optimized_without_runtime_skippable_precondition) {
    auto& engine = get_test_engine();
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    // Precondition sub-task 3/4/5 assumed this state achievable through normal topology construction;
    // it was NOT for this topology. Recorded explicitly rather than papered over.
    EXPECT_EQ(reorder_inst->get_node().is_runtime_skippable(), true)
        << "This topology could not reproduce can_be_optimized()==true with is_runtime_skippable()==false; "
           "see comment above the test for what was tried.";

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);
    network.set_input_data("input", input_mem);

    auto output_remote_mem = engine.allocate_memory({{10, 2}, data_types::f32, format::bfyx});
    network.set_output_memory("reorder", output_remote_mem, true);
    network.execute();

    // Regardless of how is_runtime_skippable() ended up, the producer must still bind directly to the
    // remote buffer with no intervening copy, since try_bind_remote_output_via_skippable_user() never
    // requires the flag for the reorder role.
    ASSERT_EQ(network.get_primitive("fc")->output_memory_ptr()->buffer_ptr(), output_remote_mem->buffer_ptr());
}

TEST(skip_reorder_at_runtime, correct_memory_reuse) {
    auto& engine = get_test_engine();

    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    std::vector<float> weight_data(weight_mem->get_layout().count());
    std::iota(weight_data.begin(), weight_data.end(), 1.0f);
    set_values(weight_mem, weight_data);

    auto input_l = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weight", weight_mem),
                      fully_connected("fc", input_info("input"), {"weight"}, "", data_types::f32),
                      reorder("reorder", input_info("fc"), format::bfyx, data_types::f32),
                      reshape("reshape", input_info("reorder"), false, {}, {2, 1, 1, 1}),
                      reorder("reorder_fsv16", input_info("reshape"), format::b_fs_yx_fsv16, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto reorder_inst = network.get_primitive("reorder");
    auto reshape_inst = network.get_primitive("reshape");
    auto reorder_fsv16_inst = network.get_primitive("reorder_fsv16");
    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    ASSERT_EQ(reshape_inst->can_be_optimized(), true);
    ASSERT_EQ(reorder_fsv16_inst->can_be_optimized(), false);

    auto input_mem = engine.allocate_memory({{10, 32}, data_types::f32, format::bfyx});
    std::vector<float> input_data(input_mem->get_layout().count());
    std::iota(input_data.begin(), input_data.end(), 0.5f);
    set_values(input_mem, input_data);

    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    outputs.begin()->second.get_memory();

    ASSERT_EQ(reorder_inst->can_be_optimized(), true);
    ASSERT_EQ(reshape_inst->can_be_optimized(), true);
    ASSERT_EQ(reorder_fsv16_inst->can_be_optimized(), false);

    auto reshape_memory_deps = reshape_inst->get_runtime_memory_dependencies();
    auto fc_unique_id = network.get_primitive("fc")->get_node().get_unique_id();
    ASSERT_TRUE(reshape_memory_deps.contains(static_cast<uint32_t>(fc_unique_id)));

    auto reorder_fsv16_memory_deps = reorder_fsv16_inst->get_runtime_memory_dependencies();
    ASSERT_TRUE(reorder_fsv16_memory_deps.contains(static_cast<uint32_t>(fc_unique_id)));
}
}  // memory_realloc_tests
