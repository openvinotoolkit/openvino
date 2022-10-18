// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_primitive_fusing, fuse_activation_to_fc_dyn) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 16, 32 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(fully_connected("fc", "input", { "weights" }));
    topology.add(activation("act", "fc", activation_func::relu));
    topology.add(reorder("reorder", "act", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<activation>(*prog));
}

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_legal) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 16, 20 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(fully_connected("fc", "input", { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", {"fc", "extra_input"}, eltwise_mode::sum));
    topology.add(reorder("reorder", "eltw", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{32, 20}, data_types::u8, format::bfyx });
    auto extra_input_memory = engine.allocate_memory(layout{ ov::PartialShape{32, 16}, data_types::f32, format::bfyx });

    net.set_input_data("input", input_memory);
    net.set_input_data("extra_input", extra_input_memory);

    auto output = net.execute();
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
}

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_illegal) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 1, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(fully_connected("fc", "input", { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", {"fc", "extra_input"}, eltwise_mode::sum));
    topology.add(reorder("reorder", "eltw", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::u8, format::bfyx });
    auto extra_input_memory = engine.allocate_memory(layout{ ov::PartialShape{2, 2}, data_types::f32, format::bfyx });
    set_values<uint8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    set_values<float>(extra_input_memory, {10, 20, 30, 40});

    net.set_input_data("input", input_memory);
    net.set_input_data("extra_input", extra_input_memory);

    auto output = net.execute();
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);

    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->size(), 4 * sizeof(float));

    mem_lock<float> lock(out_mem, net.get_stream());

    ASSERT_EQ(lock[0], 285 + 10);
    ASSERT_EQ(lock[1], 285 + 20);
    ASSERT_EQ(lock[2], 285 + 30);
    ASSERT_EQ(lock[3], 285 + 40);
}

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_illegal_const) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 1, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape{2, 2}, data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto extra_input_memory = engine.allocate_memory(in_eltw_layout);
    set_values<float>(extra_input_memory, {10, 20, 30, 40});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(data("extra_input", extra_input_memory));
    topology.add(fully_connected("fc", "input", { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", {"fc", "extra_input"}, eltwise_mode::sum));
    topology.add(reorder("reorder", "eltw", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::u8, format::bfyx });
    set_values<uint8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    net.set_input_data("input", input_memory);

    auto output = net.execute();
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);

    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->size(), 4 * sizeof(float));

    mem_lock<float> lock(out_mem, net.get_stream());

    ASSERT_EQ(lock[0], 285 + 10);
    ASSERT_EQ(lock[1], 285 + 20);
    ASSERT_EQ(lock[2], 285 + 30);
    ASSERT_EQ(lock[3], 285 + 40);
}

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_legal_scalar_const_broadcast) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 2, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape{1}, data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
    auto extra_input_memory = engine.allocate_memory(in_eltw_layout);
    set_values<float>(extra_input_memory, {10});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(data("extra_input", extra_input_memory));
    topology.add(fully_connected("fc", "input", { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", {"fc", "extra_input"}, eltwise_mode::sum));
    topology.add(reorder("reorder", "eltw", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::u8, format::bfyx });
    set_values<uint8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    net.set_input_data("input", input_memory);

    auto output = net.execute();
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);

    ASSERT_EQ(out_mem->count(), 2);
    ASSERT_EQ(out_mem->size(), 2 * sizeof(float));

    mem_lock<float> lock(out_mem, net.get_stream());

    ASSERT_EQ(lock[0], 285 + 10);
    ASSERT_EQ(lock[1], 120 + 10);
}

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_illegal_1) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 1, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // The topology below is intended to check the following tricky things:
    // 1. Cases where original eltw input is also optimized (act_e2 is fused into act_e1)
    // 1. There is another layers in fusion pattern (activations before & after eltwise)
    // 1. Changed inputs order of eltwise, i.e. fused fc node is the second input
    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(activation("act_e1", "extra_input", activation_func::relu));
    topology.add(activation("act_e2", "act_e1", activation_func::relu));
    topology.add(fully_connected("fc", "input", { "weights" }, "", data_types::f32));
    topology.add(activation("act_fc1", "fc", activation_func::relu));
    topology.add(eltwise("eltw", {"act_e2", "act_fc1"}, eltwise_mode::sum));
    topology.add(activation("act_fc2", "eltw", activation_func::relu));
    topology.add(reorder("reorder", "act_fc2", format::bfyx, data_types::f32));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    build_opts.set_option(build_option::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::u8, format::bfyx });
    auto extra_input_memory = engine.allocate_memory(layout{ ov::PartialShape{2, 2}, data_types::f32, format::bfyx });
    set_values<uint8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    set_values<float>(extra_input_memory, {10, 20, 30, 40});

    net.set_input_data("input", input_memory);
    net.set_input_data("extra_input", extra_input_memory);

    auto output = net.execute();
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);

    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->size(), 4 * sizeof(float));

    mem_lock<float> lock(out_mem, net.get_stream());

    ASSERT_EQ(lock[0], 285 + 10);
    ASSERT_EQ(lock[1], 285 + 20);
    ASSERT_EQ(lock[2], 285 + 30);
    ASSERT_EQ(lock[3], 285 + 40);
}
