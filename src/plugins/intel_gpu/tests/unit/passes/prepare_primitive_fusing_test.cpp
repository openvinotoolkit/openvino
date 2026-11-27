// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "convolution_inst.h"
#include "data_inst.h"
#include "depth_to_space_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gemm_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "pass_manager.h"
#include "program_wrapper.h"
#include "random_generator.hpp"
#include "reduce_inst.h"
#include "reshape_inst.h"
#include "test_utils.h"
#include "to_string_utils.h"

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 10}, data_types::i8, format::bfyx });
    auto extra_input_memory = engine.allocate_memory(layout{ ov::PartialShape{4, 4}, data_types::f32, format::bfyx });
    set_values<int8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

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
    auto weight_data = rg.generate_random_1d<ov::float16>(10 * 3 * 3 * 3, -1, 1);
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

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    // Valid
    auto conv_input_data = rg.generate_random_1d<ov::float16>(1 * 3 * 7 * 7, -1, 1);
    auto conv_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 3, 7, 7}, data_types::f16, format::bfyx});
    set_values(conv_input_mem, conv_input_data);

    auto elt_input_data = rg.generate_random_1d<ov::float16>(1 * 10 * 5 * 5, -10, 10);
    auto elt_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 10, 5, 5}, data_types::f16, format::bfyx});
    set_values(elt_input_mem, elt_input_data);

    net.set_input_data("conv_input", conv_input_mem);
    net.set_input_data("elt1_input", elt_input_mem);

    auto read_output = [&](const std::map<primitive_id, network_output>& outputs) {
        auto mem = outputs.at("reorder").get_memory();
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        return std::vector<float>(lock.data(), lock.data() + mem->get_layout().count());
    };

    const auto fused_ref = read_output(net.execute());
    const auto& conv_inst = net.get_primitive("conv");
    ASSERT_FALSE(conv_inst->has_unfused_subgraph());
    ASSERT_FALSE(conv_inst->use_unfused_subgraph());

    // Invalid => unfusion
    auto conv_input_data2 = rg.generate_random_1d<ov::float16>(1 * 3 * 3 * 3, -1, 1);
    auto conv_input_mem2 = engine.allocate_memory(layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx});
    set_values(conv_input_mem2, conv_input_data2);
    net.set_input_data("conv_input", conv_input_mem2);
    net.set_input_data("elt1_input", elt_input_mem);
    net.execute();
    ASSERT_TRUE(conv_inst->has_unfused_subgraph());
    ASSERT_TRUE(conv_inst->use_unfused_subgraph());

    // Valid again => fused primitive is used while the unfused subgraph stays cached
    net.set_input_data("conv_input", conv_input_mem);
    net.set_input_data("elt1_input", elt_input_mem);
    const auto fused_again = read_output(net.execute());
    ASSERT_TRUE(conv_inst->has_unfused_subgraph());
    ASSERT_FALSE(conv_inst->use_unfused_subgraph());
    ASSERT_EQ(fused_ref, fused_again);
}

// Reproducer for two issues found in the unfused-subgraph path (PR #39126 review):
//  - N1 (High): a non-output consumer reads a stale fused buffer on every valid -> invalid transition,
//    because the unfused execution swaps _outputs[0] without publishing MEMORY_CHANGED.
//  - F1 (Medium): one pooled fused output buffer is orphaned per fused -> unfused transition.
// Sequence: valid(b2,x1) -> INVALID(b1,x2) -> valid(b2,x1) -> INVALID(b1,x3) -> valid(b2,x1).
// Expected per step (fc weights all-ones [4,8], dep = [[1,1,1,1],[2,2,2,2]], pool = 1x1 identity):
//   valid:   fc = [2,4] (all 8), fused prod dep -> rows (8,16)
//   INVALID: fc = [1,4] (all 16), unfused broadcast -> rows (16,32)  (N1 stale-read gives (8,16))
//   INVALID x3: fc = [1,4] (all 24) -> rows (24,48)                  (N1 stale-read gives (8,16))
TEST(prepare_primitive_fusing, unfused_subgraph_stale_consumer_and_pool_growth) {
    auto& engine = get_test_engine();

    auto w_mem = engine.allocate_memory(layout{ov::PartialShape{4, 8}, data_types::f16, format::bfyx});
    set_values(w_mem, std::vector<ov::float16>(32, ov::float16(1.0f)));
    auto d_mem = engine.allocate_memory(layout{ov::PartialShape{2, 4}, data_types::f16, format::bfyx});
    {
        std::vector<ov::float16> dv(8, ov::float16(1.0f));
        for (size_t i = 4; i < 8; ++i) dv[i] = ov::float16(2.0f);
        set_values(d_mem, dv);
    }

    topology topo;
    topo.add(input_layout("fc_input", layout{ov::PartialShape{-1, 8}, data_types::f16, format::bfyx}));
    topo.add(data("weights", w_mem));
    topo.add(data("dep", d_mem));
    topo.add(fully_connected("fc", input_info("fc_input"), {"weights"}, "", data_types::f16));
    topo.add(eltwise("eltw", { input_info("fc"), input_info("dep") }, eltwise_mode::prod));
    topo.add(pooling("pool", input_info("eltw"), pooling_mode::max, {1, 1}, {1, 1}));
    topo.add(reorder("reorder", input_info("pool"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    // Full program pipeline: production memory restrictions are required for the F1 pool-growth
    // assertion to be meaningful (a partial build has no self-restriction and self-heals the pool).
    auto prog = program::build_program(engine, topo, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    const auto fc_inst = net.get_primitive("fc");
    ASSERT_NE(fc_inst, nullptr);

    auto make_in = [&](size_t batch, float v) {
        auto m = engine.allocate_memory(layout{ov::PartialShape{(int)batch, 8}, data_types::f16, format::bfyx});
        set_values(m, std::vector<ov::float16>(batch * 8, ov::float16(v)));
        return m;
    };
    auto read_out = [&](const std::map<primitive_id, network_output>& outputs) {
        auto om = outputs.at("reorder").get_memory();
        mem_lock<float, mem_lock_type::read> lock(om, get_test_stream());
        return std::vector<float>(lock.data(), lock.data() + 8);
    };
    auto ex = [&]() { return net.execute(); };

    net.set_input_data("fc_input", make_in(2, 1.0f));
    auto o1 = read_out(ex());
    const size_t pool_after_first_valid = net.get_memory_pool().get_non_padded_pool_size();

    net.set_input_data("fc_input", make_in(1, 2.0f));
    auto o2 = read_out(ex());
    ASSERT_TRUE(fc_inst->use_unfused_subgraph()) << "invalid step must run the unfused subgraph";

    net.set_input_data("fc_input", make_in(2, 1.0f));
    auto o3 = read_out(ex());
    ASSERT_FALSE(fc_inst->use_unfused_subgraph()) << "switch back to fused";

    net.set_input_data("fc_input", make_in(1, 3.0f));
    auto o4 = read_out(ex());

    net.set_input_data("fc_input", make_in(2, 1.0f));
    auto o5 = read_out(ex());
    const size_t pool_final = net.get_memory_pool().get_non_padded_pool_size();

    // N1: non-output consumer (pool) must see the fresh subgraph result on invalid steps
    EXPECT_EQ(o1[0], 8.0f); EXPECT_EQ(o1[4], 16.0f);
    EXPECT_EQ(o2[0], 16.0f); EXPECT_EQ(o2[4], 32.0f) << "N1: consumer read stale fused buffer";
    EXPECT_EQ(o3[0], 8.0f); EXPECT_EQ(o3[4], 16.0f);
    EXPECT_EQ(o4[0], 24.0f); EXPECT_EQ(o4[4], 48.0f) << "N1: consumer read stale fused buffer";
    EXPECT_EQ(o5[0], 8.0f); EXPECT_EQ(o5[4], 16.0f);

    // F1: pool must not grow across the valid <-> invalid transitions
    EXPECT_LE(pool_final, pool_after_first_valid) << "F1: pooled fused output buffer orphaned per transition";
}

TEST(prepare_primitive_fusing, unfused_subgraph_waits_for_deps_in_oooq) {
    // act -> conv(+eltw fused). When the fusion becomes invalid at runtime, conv is executed as an unfused subgraph
    // which must still receive the dependency events of its producers in out-of-order queue.
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    topology topology;
    auto conv_in_layout = layout{ov::PartialShape{1, 3, -1, -1}, data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape{10, 3, 3, 3}, data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);
    auto weight_data = rg.generate_random_4d<ov::float16>(10, 3, 3, 3, -1, 1);
    set_values(weight_mem, weight_data);
    auto elt_in_layout = layout{ov::PartialShape{1, 10, -1, -1}, data_types::f16, format::bfyx};

    topology.add(data("weights", weight_mem));
    topology.add(input_layout("conv_input", conv_in_layout));
    topology.add(input_layout("elt_input", elt_in_layout));
    topology.add(activation("act", input_info("conv_input"), activation_func::relu));
    topology.add(convolution("conv", input_info("act"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltw", {input_info("conv"), input_info("elt_input")}, eltwise_mode::prod));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    // conv output is 1x10x1x1 while eltwise input is 1x10x5x5 => invalid fusion
    auto conv_input_data = rg.generate_random_4d<ov::float16>(1, 3, 3, 3, -1, 1);
    auto conv_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx});
    set_values(conv_input_mem, conv_input_data);

    auto elt_input_data = rg.generate_random_4d<ov::float16>(1, 10, 5, 5, -10, 10);
    auto elt_input_mem = engine.allocate_memory(layout{ov::PartialShape{1, 10, 5, 5}, data_types::f16, format::bfyx});
    set_values(elt_input_mem, elt_input_data);

    net.set_input_data("conv_input", conv_input_mem);
    net.set_input_data("elt_input", elt_input_mem);
    net.execute();

    const auto& conv_inst = net.get_primitive("conv");
    ASSERT_TRUE(conv_inst->has_unfused_subgraph());
    ASSERT_FALSE(conv_inst->get_impl_params()->dep_events.empty());
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_removal_check) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), {1, 0}),
        reorder("reorder_dt", input_info("permute"), format::fbyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl} }));
    }

    auto prog = program::build_program(engine, topology, config, false, true);

    prog->get_layout_optimizer().set_implementation_forcing(config.get_force_implementations());
    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_TRUE(!has_node(*prog, "permute"));
    ASSERT_EQ(prog->get_node("weights").get_output_layout().format, format::fbyx);

    if (engine.get_device_info().supports_immad) {
        ASSERT_TRUE(has_node(*prog, "reorder_dt"));
        ASSERT_EQ(prog->get_node("reorder_dt").get_output_layout().format, format::bfyx);
    }
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

// Regression test for https://github.com/openvinotoolkit/openvino/issues/39700:
// a rank-3 [1, N, K] weights constant transposed by {0, 2, 1} must not be fused as a swap of its outer dims.
TEST(prepare_primitive_fusing, fuse_constant_transposes_3d_weights_accuracy_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "FC with non-transposed constant weights is only produced for oneDNN (XMX) devices";
    }

    const size_t T = 2, K = 32, N = 4;
    auto input = engine.allocate_memory({ { 1, T, K }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ { 1, N, K }, data_types::f16, format::bfyx });

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_1d<ov::float16>(T * K, -1, 1);
    auto weights_data = rg.generate_random_1d<ov::float16>(N * K, -1, 1);
    set_values(input, input_data);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), {0, 2, 1}),
        fully_connected("fc", input_info("input"), "permute", "", data_types::f16, 3, 3, false)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto output = outputs.at("fc").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), T * N);
    for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
            float ref = 0.f;
            for (size_t k = 0; k < K; ++k) {
                ref += static_cast<float>(input_data[t * K + k]) * static_cast<float>(weights_data[n * K + k]);
            }
            ASSERT_NEAR(static_cast<float>(output_ptr[t * N + n]), ref, 2e-2f) << "t=" << t << " n=" << n;
        }
    }
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_multiple_users) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "The weights format reorder is only inserted for oneDNN fully connected";
    }

    auto input = engine.allocate_memory({{2, 32}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({{32, 2}, data_types::f16, format::bfyx});

    topology topology(input_layout("input", input->get_layout()),
                      data("weights", weights),
                      permute("permute", input_info("weights"), {1, 0}),
                      fully_connected("fc0", input_info("input"), {"permute"}, "", data_types::f16),
                      fully_connected("fc1", input_info("input"), {"permute"}, "", data_types::f16),
                      fully_connected("fc2", input_info("input"), {"permute"}, "", data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc0", fc_impl}, {"fc1", fc_impl}, {"fc2", fc_impl}}));

    auto prog = program::build_program(engine, topology, config, false, true);
    prog->get_layout_optimizer().set_implementation_forcing(config.get_force_implementations());
    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_FALSE(has_node(*prog, "permute"));
    for (const auto& fc_id : {"fc0", "fc1", "fc2"}) {
        const auto& fc_weights = prog->get_node(fc_id).get_dependency(1);
        ASSERT_TRUE(fc_weights.is_type<reorder>()) << fc_id << " reads weights from " << fc_weights.id();
        ASSERT_EQ(fc_weights.get_output_layout().format, format::bfyx) << fc_id;
    }
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_multiple_users_accuracy_test) {
    auto& engine = get_test_engine();

    const int64_t m = 4, k = 64, n = 16;
    auto input = engine.allocate_memory({{m, k}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({{k, n}, data_types::f16, format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_2d<ov::float16>(m, k, -1, 1);
    auto weights_data = rg.generate_random_2d<ov::float16>(k, n, -1, 1);

    set_values(input, flatten_2d(format::bfyx, input_data));
    set_values(weights, flatten_2d(format::bfyx, weights_data));

    topology topology(input_layout("input", input->get_layout()),
                      data("weights", weights),
                      permute("permute", input_info("weights"), {1, 0}),
                      activation("act1", input_info("input"), activation_func::negative),
                      activation("act2", input_info("input"), activation_func::abs),
                      fully_connected("fc0", input_info("input"), {"permute"}, "", data_types::f16),
                      fully_connected("fc1", input_info("act1"), {"permute"}, "", data_types::f16),
                      fully_connected("fc2", input_info("act2"), {"permute"}, "", data_types::f16));

    auto run = [&](bool optimize) {
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(optimize));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cldnn::network network(engine, topology, config);
        network.set_input_data("input", input);
        auto outputs = network.execute();
        std::vector<std::vector<ov::float16>> results;
        for (const auto& id : {"fc0", "fc1", "fc2"}) {
            cldnn::mem_lock<ov::float16> ptr(outputs.at(id).get_memory(), get_test_stream());
            results.emplace_back(ptr.begin(), ptr.end());
        }
        return results;
    };

    auto results = run(true);
    auto results_ref = run(false);

    for (size_t fc = 0; fc < results_ref.size(); ++fc) {
        ASSERT_EQ(results[fc].size(), results_ref[fc].size());
        for (size_t i = 0; i < results_ref[fc].size(); ++i) {
            ASSERT_NEAR(results[fc][i], results_ref[fc][i], 1e-2f) << "fc" << fc << " index " << i;
        }
    }
}

// Same graph the plugin builds for MatMul(x, Constant) under f16 inference: data(f32) -> permute -> Convert -> FC.
// Here the transposed weight is shared by two FCs, as when one Linear layer is reused across unrolled steps.
TEST(prepare_primitive_fusing, fuse_constant_transposes_shared_weights_accuracy) {
    auto& engine = get_test_engine();
    const int64_t ifm = 29;
    const int64_t ofm = 64;

    tests::random_generator rg(GET_SUITE_NAME);
    auto weights_data = rg.generate_random_1d<float>(ifm * ofm, -1, 1);
    auto input0_data = rg.generate_random_1d<ov::float16>(ifm, -1, 1);
    auto input1_data = rg.generate_random_1d<ov::float16>(ifm, -1, 1);

    auto weights = engine.allocate_memory({ov::PartialShape{ifm, ofm}, data_types::f32, format::bfyx});
    auto input0 = engine.allocate_memory({ov::PartialShape{1, ifm}, data_types::f16, format::bfyx});
    auto input1 = engine.allocate_memory({ov::PartialShape{1, ifm}, data_types::f16, format::bfyx});
    set_values(weights, weights_data);
    set_values(input0, input0_data);
    set_values(input1, input1_data);

    topology topology(
        input_layout("input0", input0->get_layout()),
        input_layout("input1", input1->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), {1, 0}),
        reorder("convert", input_info("permute"), format::any, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        fully_connected("fc0", input_info("input0"), "convert", "", data_types::f16),
        fully_connected("fc1", input_info("input1"), "convert", "", data_types::f16),
        reorder("out0", input_info("fc0"), format::bfyx, data_types::f32),
        reorder("out1", input_info("fc1"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc0", fc_impl}, {"fc1", fc_impl} }));
    }

    cldnn::network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    auto check = [&](const primitive_id& out_id, const std::vector<ov::float16>& input_data) {
        cldnn::mem_lock<float> output_ptr(outputs.at(out_id).get_memory(), get_test_stream());
        ASSERT_EQ(output_ptr.size(), static_cast<size_t>(ofm));
        for (int64_t o = 0; o < ofm; ++o) {
            float ref = 0.f;
            for (int64_t i = 0; i < ifm; ++i)
                ref += static_cast<float>(input_data[i]) * weights_data[i * ofm + o];
            ASSERT_NEAR(output_ptr[o], ref, 5e-2f) << out_id << " output " << o;
        }
    };
    check("out0", input0_data);
    check("out1", input1_data);
}

// FC with a 3D [1, ifm, ofm] or 4D [1, 1, ifm, ofm] constant weight whose last two dims are transposed, as built for
// MatMul(x, Constant): f32 weights with an f16 hint, or FP16-compressed IR weights with an f32 hint.
static void test_fuse_constant_transposes_nd_weights(size_t rank, data_types weights_dt, data_types compute_dt) {
    auto& engine = get_test_engine();
    const int64_t rows = 4;
    const int64_t ifm = 16;
    const int64_t ofm = 8;
    const bool is_4d = rank == 4;

    tests::random_generator rg(GET_SUITE_NAME);
    auto weights_data = rg.generate_random_1d<float>(ifm * ofm, -1, 1);
    auto input_data = rg.generate_random_1d<float>(rows * ifm, -1, 1);

    auto weights = engine.allocate_memory({is_4d ? ov::PartialShape{1, 1, ifm, ofm} : ov::PartialShape{1, ifm, ofm}, weights_dt, format::bfyx});
    auto input = engine.allocate_memory({is_4d ? ov::PartialShape{1, 1, rows, ifm} : ov::PartialShape{1, rows, ifm}, compute_dt, format::bfyx});
    // Rounds f16-stored data in place so the reference uses the same values.
    auto fill = [](memory::ptr mem, std::vector<float>& values) {
        if (mem->get_layout().data_type == data_types::f16) {
            std::vector<ov::float16> values_f16(values.begin(), values.end());
            set_values(mem, values_f16);
            values.assign(values_f16.begin(), values_f16.end());
        } else {
            set_values(mem, values);
        }
    };
    fill(weights, weights_data);
    fill(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), is_4d ? std::vector<uint16_t>{0, 1, 3, 2} : std::vector<uint16_t>{0, 2, 1}),
        reorder("convert", input_info("permute"), format::any, compute_dt,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        fully_connected("fc", input_info("input"), "convert", "", compute_dt, rank, rank),
        reorder("out", input_info("fc"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl} }));
    }

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    const float tolerance = compute_dt == data_types::f16 ? 5e-2f : 1e-3f;
    cldnn::mem_lock<float> output_ptr(outputs.at("out").get_memory(), get_test_stream());
    ASSERT_EQ(output_ptr.size(), static_cast<size_t>(rows * ofm));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t o = 0; o < ofm; ++o) {
            float ref = 0.f;
            for (int64_t i = 0; i < ifm; ++i)
                ref += input_data[r * ifm + i] * weights_data[i * ofm + o];
            ASSERT_NEAR(output_ptr[r * ofm + o], ref, tolerance) << "row " << r << " output " << o;
        }
    }
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_3d_weights_accuracy) {
    test_fuse_constant_transposes_nd_weights(3, data_types::f32, data_types::f16);
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_4d_weights_accuracy) {
    test_fuse_constant_transposes_nd_weights(4, data_types::f32, data_types::f16);
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_3d_f16_weights_f32_accuracy) {
    test_fuse_constant_transposes_nd_weights(3, data_types::f16, data_types::f32);
}

TEST(prepare_primitive_fusing, fuse_constant_transposes_4d_f16_weights_f32_accuracy) {
    test_fuse_constant_transposes_nd_weights(4, data_types::f16, data_types::f32);
}

TEST(prepare_primitive_fusing, can_profiling_data_when_fuse_illegal) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ov::PartialShape{2, 10}, data_types::u8, format::bfyx});
    auto in_layout = layout{ov::PartialShape::dynamic(2), data_types::u8, format::bfyx};
    auto in_eltw_layout = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};

    set_values<uint8_t>(weights, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", in_eltw_layout));
    topology.add(fully_connected("fc", input_info("input"), {"weights"}, "", data_types::f32));
    topology.add(eltwise("eltw", {input_info("fc"), input_info("extra_input")}, eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(ov::intel_gpu::QueueTypes::in_order));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::enable_profiling(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<eltwise>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ov::PartialShape{1, 10}, data_types::u8, format::bfyx});
    auto extra_input_memory = engine.allocate_memory(layout{ov::PartialShape{2, 2}, data_types::f32, format::bfyx});
    set_values<uint8_t>(input_memory, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    set_values<float>(extra_input_memory, {10, 20, 30, 40});

    net.set_input_data("input", input_memory);
    net.set_input_data("extra_input", extra_input_memory);

    auto output = net.execute();
    for (auto& iter : output)
        ASSERT_NE(iter.second.get_event(), nullptr);
}

TEST(prepare_primitive_fusing, dont_fuse_eltwise_to_dyn_dts) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    auto in_layout = layout{ ov::PartialShape{-1, -1, -1, -1}, data_types::f32, format::bfyx };
    auto weight_layout = layout{ ov::PartialShape{32, 32, 3, 3}, data_types::f32, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);
    auto weight_data = rg.generate_random_4d<ov::float16>(32, 32, 3, 3, -1, 1);
    set_values(weight_mem, weight_data);
    auto scale_layout = layout{ ov::PartialShape{1, 2, 1, 1}, data_types::f32, format::bfyx };
    auto scale_mem = engine.allocate_memory(scale_layout);
    auto elt_layout = layout{ ov::PartialShape{1, 2, 32, 32}, data_types::f32, format::bfyx };
    auto elt_mem = engine.allocate_memory(elt_layout);

    topology topology;

    topology.add(data("weights", weight_mem));
    topology.add(input_layout("input", in_layout));
    topology.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(depth_to_space("depth_to_space", input_info("conv"), 4, depth_to_space_mode::blocks_first));
    topology.add(data("scale1_data", scale_mem));
    topology.add(eltwise("scale1", { input_info("depth_to_space"), input_info("scale1_data") }, eltwise_mode::prod, data_types::f32));
    topology.add(activation("actv1", input_info("scale1"), activation_func::relu));
    topology.add(data("eltw_data", elt_mem));
    topology.add(eltwise("eltw", { input_info("actv1"), input_info("eltw_data") }, eltwise_mode::sum, data_types::f32));
    topology.add(reorder("reorder_bfyx", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node(*prog, "scale1"));
}

static void move_next(program& p, primitive_id node_id, primitive_id key_id) {
    auto n = &p.get_node(node_id);
    p.get_processing_order().erase(&p.get_node(node_id));
    p.get_processing_order().insert_next(&p.get_node(key_id), n);
}

TEST(prepare_primitive_fusing, fuse_by_priotizing_to_parent_in_fusing_history) {
    //    in1    in2                   in1   in2
    //     |      |                     |     |
    //   conv1  conv2                   |   conv2
    //     |      |                     |    /
    //   actv1  eltw1                   |   /
    //    /   \   |                     |  /
    //  eltw2   eltw3      ----->     conv1 (actv1, eltw2, eltw3, eltw4, eltw5, eltw6)
    //    |       |                     |
    //  eltw4     |                     |
    //    |       |                     |
    //  eltw6 -- eltw5                  |
    //            |                     |
    //           eltw7                eltw7
    //
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    auto in_layout      = layout{ ov::PartialShape{32, 96, -1, -1}, data_types::f32, format::bfyx };
    auto weights_layout = layout{ ov::PartialShape{32, 96, 1, 1}, data_types::f32, format::bfyx};
    auto eltwise_layout = layout{ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx };

    auto weights_memory = engine.allocate_memory(weights_layout);
    auto eltwise_memory = engine.allocate_memory(eltwise_layout);

    auto weights_data   = rg.generate_random_4d<ov::float16>(32, 96, 1, 1, 1, 1);
    auto eltwise_data   = rg.generate_random_1d<ov::float16>(1, 1, 1, 1);

    topology topology(
        input_layout("input1", in_layout),
        input_layout("input2", in_layout),
        data("weight1", weights_memory),
        data("weight2", weights_memory),
        data("data1", eltwise_memory),
        data("data2", eltwise_memory),
        data("data3", eltwise_memory),
        data("data4", eltwise_memory),
        data("data5", eltwise_memory),
        convolution("conv1", input_info("input1"), "weight1", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        convolution("conv2", input_info("input2"), "weight2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        activation("actv1", input_info("conv1"), activation_func::hard_sigmoid),
        eltwise("eltw1", { input_info("conv2"), input_info("data1") }, eltwise_mode::sum, data_types::f32),
        eltwise("eltw2", { input_info("actv1"), input_info("data2") }, eltwise_mode::prod, data_types::f32),
        eltwise("eltw3", { input_info("actv1"), input_info("eltw1") }, eltwise_mode::sub, data_types::f32),
        eltwise("eltw4", { input_info("eltw2"), input_info("data3") }, eltwise_mode::prod, data_types::f32),
        eltwise("eltw5", { input_info("eltw3"), input_info("eltw6") }, eltwise_mode::prod, data_types::f32),
        eltwise("eltw6", { input_info("eltw4"), input_info("data4") }, eltwise_mode::sum, data_types::f32),
        eltwise("eltw7", { input_info("eltw5"), input_info("data5") }, eltwise_mode::prod, data_types::f32)
    );


    set_values(weights_memory, weights_data);
    set_values(eltwise_memory, eltwise_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto program = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(program, nullptr);

    int32_t process_number_conv1 = program->get_processing_order().get_processing_number(&program->get_node("conv1"));
    int32_t process_number_conv2 = program->get_processing_order().get_processing_number(&program->get_node("conv2"));

    if (process_number_conv1 > process_number_conv2) {
        // Change the processing order to create a expected test case,
        // where fused_idx < peer_idx and the processing order of parents[0] is lower than parents[1].
        //
        // index  : 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
        // before : w2| w1| i2| c2| i1| c1| a1| d5| d4| d3| d2 | e2 | e4 | e6 | d1 | e1 | e3 | e5 | e7 |
        // after  : w1| w2| i1| c1| a1| i2| c2| d5| d4| d3| d2 | d1 | e1 | e2 | e3 | e4 | e6 | e5 | e7 |
        //
        // w : weights
        // i : input
        // c : convolution
        // a : activation
        // d : data
        // e : eltwise
        //
        move_next(*program, "weight2", "weight1");
        move_next(*program, "input1" , "weight2");
        move_next(*program, "conv1"  , "input1" );
        move_next(*program, "actv1"  , "conv1"  );
        move_next(*program, "data1"  , "data2"  );
        move_next(*program, "eltw1"  , "data1"  );
        move_next(*program, "eltw3"  , "eltw2"  );
    }

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*program);

    ASSERT_FALSE(has_node(*program, "actv1"));
    ASSERT_FALSE(has_node(*program, "eltw3"));
    ASSERT_FALSE(has_node(*program, "eltw4"));
    ASSERT_FALSE(has_node(*program, "eltw5"));
    ASSERT_FALSE(has_node(*program, "eltw6"));
}

TEST(prepare_primitive_fusing, dont_fuse_bias_when_multiple_users) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx };
    auto weights = engine.allocate_memory({ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx });
    auto bias = engine.allocate_memory({ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx });

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(activation("act", input_info("conv"), activation_func::relu));
    topology.add(eltwise("eltw", { input_info("conv"), input_info("bias") }, eltwise_mode::sum));
    topology.add(reorder("reorder_act", input_info("act"), format::bfyx, data_types::f32));
    topology.add(reorder("reorder_eltw", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);

    ASSERT_TRUE(has_node(*prog, "eltw"));
    ASSERT_TRUE(has_node(*prog, "conv"));
}
