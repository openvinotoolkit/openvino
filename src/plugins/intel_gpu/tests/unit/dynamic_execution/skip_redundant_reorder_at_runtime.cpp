// Copyright (C) 2023 Intel Corporation
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
    auto weight_mem = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    auto output_remote_mem = engine.allocate_memory({{10, 2}, data_types::f32, format::bfyx});
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
    ASSERT_EQ(reorder_inst->can_be_optimized(), false);
    ASSERT_EQ(output_remote_mem->buffer_ptr(), network.get_output_memory("reorder")->buffer_ptr());
    ASSERT_NE(network.get_output_memory("reorder")->buffer_ptr(), network.get_primitive("fc")->output_memory_ptr()->buffer_ptr());
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
    ASSERT_TRUE(reshape_memory_deps.find(fc_unique_id) != reshape_memory_deps.end());

    auto reorder_fsv16_memory_deps = reorder_fsv16_inst->get_runtime_memory_dependencies();
    ASSERT_TRUE(reorder_fsv16_memory_deps.find(fc_unique_id) != reorder_fsv16_memory_deps.end());
}
}  // memory_realloc_tests
