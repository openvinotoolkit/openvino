// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "reduce_inst.h"
#include "reshape_inst.h"
#include "fully_connected_inst.h"
#include "gemm_inst.h"
#include "convolution_inst.h"
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
    topology.add(fully_connected("fc", input_info("input"), { "weights" }));
    topology.add(activation("act", input_info("fc"), activation_func::relu));
    topology.add(reorder("reorder", input_info("act"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<activation>(*prog));
}

TEST(prepare_primitive_fusing, dont_fuse_incompatible_eltwise) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape{-1, -1, 10}, data_types::f32, format::bfyx };
    auto const_layout = layout{ ov::PartialShape{1, 1, 1}, data_types::f32, format::bfyx };
    auto const_mem = engine.allocate_memory(const_layout);

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("const", const_mem));
    topology.add(eltwise("eltw_pre", { input_info("input"), input_info("const") }, eltwise_mode::sum));
    topology.add(reduce("reduce", input_info("eltw_pre"), reduce_mode::max, {2}, true));
    topology.add(eltwise("eltw", { input_info("input"), input_info("reduce") }, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node(*prog, "eltw"));
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
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", { input_info("fc"), input_info("extra_input") }, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

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
    auto weights = engine.allocate_memory({ ov::PartialShape{ 2, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", { input_info("fc"), input_info("extra_input")}, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

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
    auto weights = engine.allocate_memory({ ov::PartialShape{ 2, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape{2, 2}, data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto extra_input_memory = engine.allocate_memory(in_eltw_layout);
    set_values<float>(extra_input_memory, {10, 20, 30, 40});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(data("extra_input", extra_input_memory));
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", { input_info("fc"), input_info("extra_input") }, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

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
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f32));
    topology.add(eltwise("eltw", { input_info("fc"), input_info("extra_input") }, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

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
    auto weights = engine.allocate_memory({ ov::PartialShape{ 2, 10 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // The topology below is intended to check the following tricky things:
    // 1. Cases where original eltw input is also optimized (act_e2 is fused into act_e1)
    // 1. There is another layers in fusion pattern (activations before & after eltwise)
    // 1. Changed inputs order of eltwise, i.e. fused fc node is the second input
    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(activation("act_e1", input_info("extra_input"), activation_func::relu));
    topology.add(activation("act_e2", input_info("act_e1"), activation_func::relu));
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f32));
    topology.add(activation("act_fc1", input_info("fc"), activation_func::relu));
    topology.add(eltwise("eltw", { input_info("act_e2"), input_info("act_fc1")}, eltwise_mode::sum));
    topology.add(activation("act_fc2", input_info("eltw"), activation_func::relu));
    topology.add(reorder("reorder", input_info("act_fc2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

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

TEST(prepare_primitive_fusing, fuse_eltwise_to_fc_dyn_illegal_2) {
    auto& engine = get_test_engine();
    auto weights0 = engine.allocate_memory({ ov::PartialShape{ 2, 10 }, data_types::i8, format::bfyx });
    auto weights1 = engine.allocate_memory({ ov::PartialShape{ 4, 2 }, data_types::i8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::i8, format::bfyx };
    auto in_eltw_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    set_values<uint8_t>(weights0, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    set_values<uint8_t>(weights1, {1, 1, 1, 1, 1, 1, 1, 1});


    // The topology below is intended to check the following tricky things:
    // 1. Cases where original eltw input is also optimized (act_e2 is fused into act_e1)
    // 1. There is another layers in fusion pattern (activations before & after eltwise)
    // 1. Also, the input (act_fc1) of the fused node of the eltw (i.e., fc2) is fused to other node (fc1)

    topology topology;
    topology.add(data("weights0", weights0));
    topology.add(data("weights1", weights1));
    topology.add(input_layout("input", in_layout));
    topology.add(fully_connected("fc1", input_info("input"), { "weights0" }, "", data_types::i8));
    topology.add(activation("act_fc1", input_info("fc1"), activation_func::relu));
    topology.add(fully_connected("fc2", input_info("act_fc1"), { "weights1" }, "", data_types::i8));
    topology.add(activation("act_fc2", input_info("fc2"), activation_func::relu));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(activation("act_e1", input_info("extra_input"), activation_func::abs));
    topology.add(activation("act_e2", input_info("act_e1"), activation_func::relu));
    topology.add(eltwise("eltw", { input_info("act_fc2"), input_info("act_e2") }, eltwise_mode::sum));
    topology.add(activation("act_fc3", input_info("eltw"), activation_func::relu));
    topology.add(reorder("reorder", input_info("act_fc3"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::i8, format::bfyx });
    auto extra_input_memory = engine.allocate_memory(layout{ ov::PartialShape{4, 4}, data_types::f32, format::bfyx });
    set_values<int8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10});
    set_values<float>(extra_input_memory, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});

    net.set_input_data("input", input_memory);
    net.set_input_data("extra_input", extra_input_memory);

    auto output = net.execute();
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);

    ASSERT_EQ(out_l.batch(), 4);
    ASSERT_EQ(out_l.feature(), 4);
    ASSERT_EQ(out_mem->count(), 16);
    ASSERT_EQ(out_mem->size(), 16 * sizeof(float));

    mem_lock<float> lock(out_mem, net.get_stream());

    ASSERT_EQ(lock[0], 91);
    ASSERT_EQ(lock[1], 92);
    ASSERT_EQ(lock[2], 93);
    ASSERT_EQ(lock[3], 94);
}

TEST(prepare_primitive_fusing, dont_remove_only_dep_reshape) {
    // Topology:
    // input -> reshape(w/ 2nd non-const input) -> reshape(w/ 2nd const input) -> gemm
    //
    // Expectation:
    // If only the input size of depedency reshape is not 1 among the sequence of reshapes
    // The current reshape alone should not be removed, and removing redundant reshapes is skipped

    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto pattern_layout = layout{ ov::PartialShape{ 4 }, data_types::i64, format::bfyx };

    std::vector<int64_t> output_pattern { 0, 1, -1, 0 };

    topology topology;
    topology.add(input_layout("input1", in_layout));
    topology.add(input_layout("pattern1", pattern_layout));
    topology.add(input_layout("input2", in_layout));
    topology.add(reshape("reshape1", input_info("input1"), input_info("pattern1"), true, ov::PartialShape::dynamic(4)));
    topology.add(reshape("reshape2", input_info("reshape1"), true, output_pattern, ov::PartialShape::dynamic(4)));
    topology.add(gemm("gemm", { input_info("reshape2"), input_info("input2") }, data_types::f32, false, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node(*prog, "reshape2"));
}

TEST(prepare_primitive_fusing, eltwise_fusing_residual_connection) {
    // Extended eltwise fusing pattern
    //   in    w
    //     \  /
    //     conv   elt1_in1
    //     |  \    /
    //     |   elt1
    //     |    |
    //     |   act
    //     |  /
    //     elt2
    //     |
    //    reorder
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    topology topology;
    auto conv_in_layout = layout{ ov::PartialShape{1, 3, -1, -1}, data_types::f16, format::bfyx};
    auto weight_layout = layout{ ov::PartialShape{10, 3, 3, 3}, data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);
    auto weight_data = rg.generate_random_4d<ov::float16>(10, 3, 3, 3, -1, 1);
    set_values(weight_mem, weight_data);
    auto elt1_in1_layout = layout{ ov::PartialShape{1, 10, -1, -1}, data_types::f16, format::bfyx};

    topology.add(data("weights", weight_mem));
    topology.add(input_layout("conv_input", conv_in_layout));
    topology.add(input_layout("elt1_input", elt1_in1_layout));
    topology.add(convolution("conv", input_info("conv_input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltw1", { input_info("conv"), input_info("elt1_input") }, eltwise_mode::prod));
    topology.add(activation("act", input_info("eltw1"), activation_func::erf));
    topology.add(eltwise("elt2", { input_info("conv"), input_info("act") }, eltwise_mode::prod));
    topology.add(reorder("reorder", input_info("elt2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);
    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    // Valid
    auto conv_input_data = rg.generate_random_4d<ov::float16>(1, 3, 7, 7, -1, 1);
    auto conv_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 3, 7, 7}, data_types::f16, format::bfyx});
    set_values(conv_input_mem, conv_input_data);

    auto elt_input_data = rg.generate_random_4d<ov::float16>(1, 10, 5, 5, -10, 10);
    auto elt_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 10, 5, 5}, data_types::f16, format::bfyx});
    set_values(elt_input_mem, elt_input_data);

    net.set_input_data("conv_input", conv_input_mem);
    net.set_input_data("elt1_input", elt_input_mem);

    net.execute();
    const auto& conv_inst = net.get_primitive("conv");
    ASSERT_FALSE(conv_inst->has_unfused_subgraph());

    // Invalid => unfusion
    auto conv_input_data2 = rg.generate_random_4d<ov::float16>(1, 3, 3, 3, -1, 1);
    auto conv_input_mem2 = engine.allocate_memory(layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx});
    set_values(conv_input_mem2, conv_input_data2);
    net.set_input_data("conv_input", conv_input_mem2);
    net.set_input_data("elt1_input", elt_input_mem);
    net.execute();
    ASSERT_TRUE(conv_inst->has_unfused_subgraph());
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_removal_check) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), {1, 0}),
        reorder("reorder_dt", input_info("permute"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);
    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_TRUE(!has_node(*prog, "permute"));
    ASSERT_EQ(prog->get_node("weights").get_output_layout().format, format::fbyx);
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_accuracy_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_2d<ov::float16>(2, 32, -1, 1);
    auto weights_data = rg.generate_random_2d<float>(32, 2, -1, 1);

    set_values(input, flatten_2d(format::bfyx, input_data));
    set_values(weights, flatten_2d(format::bfyx, weights_data));

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        permute("permute", input_info("reorder_dt"), {1, 0}),
        fully_connected("fc", input_info("input"), { "permute" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto output = outputs.at("fc").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());

    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network_ref(engine, topology, config_ref);
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    auto output_ref = outputs_ref.at("fc").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr_ref(output_ref, get_test_stream());

    for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
        ASSERT_EQ(output_ptr[i], output_ptr_ref[i]);
    }
}
