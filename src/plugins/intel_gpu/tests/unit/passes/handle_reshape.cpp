// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/permute.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "reorder_inst.h"
#include "broadcast_inst.h"
#include "fully_connected_inst.h"
#include "pass_manager.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(handle_reshape, dont_remove_reshape_that_changes_rank) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{}, data_types::f16, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f16, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(0), data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_layout));
    topology.add(data("data1", data1_layout));
    topology.add(eltwise("e1", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("e1"), false, {1}, {1}));
    topology.add(eltwise("e2", input_info("reshape"), input_info("data1"), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
}

TEST(handle_reshape, dont_remove_reshape_that_changes_rank_chain) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{1, 1}, data_types::f16, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1, 1, 1}, data_types::f16, format::bfyx });
    auto in_layout = layout{ ov::PartialShape{2, 4}, data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_layout));
    topology.add(data("data1", data1_layout));
    topology.add(eltwise("e1", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reshape("reshape1", input_info("e1"), false, {2, 1, 4}, {2, 1, 4})); // 2d -> 3d
    topology.add(reshape("reshape2", input_info("reshape1"), true, {1, 2, 0}, {1, 2, 0})); // swap 0 and 1 axis and keep 2nd axis as is
    topology.add(eltwise("e2", input_info("reshape2"), input_info("data1"), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<handle_reshape>(*prog);
    program_wrapper::apply_opt_pass<remove_redundant_reorders>(*prog);

    for (auto& n : prog->get_processing_order()) {
        n->recalc_output_layout(true);
    }

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node(*prog, "reshape1"));
    ASSERT_TRUE(has_node(*prog, "reshape2"));

    ASSERT_TRUE(prog->get_node("reshape1").can_be_optimized());
    ASSERT_TRUE(prog->get_node("reshape2").can_be_optimized());
}

TEST(handle_reshape, skip_reorder_node_to_split_when_onndnn_not_support) {
    // Onednn FC does not support fp32 input, fp16 weight. In such case, we need to ignore reorder_split from handle_reshape pass
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 9, 1, 1024} });
    auto data_01 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 9, 1, 1024} });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1024, 1, 1024} });
    auto bias = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 1, 1024} });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("data", data_01));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(eltwise("e1", input_info("input"), input_info("data"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("e1"), tensor(9, 1, 1, 1024), cldnn::reshape::reshape_mode::base));
    topology.add(reorder("convert_to_f32", input_info("reshape"), { data_types::f32, format::bfyx, { 9, 1, 1, 1024} }));
    topology.add(fully_connected("matmul", input_info("reshape"), "weights", "bias", 3, 2));
    topology.add(reorder("convert_to_f32_matmul", input_info("matmul"), { data_types::f32, format::bfyx, { 9, 1, 1, 1024} }));
    topology.add(eltwise("e2", input_info("convert_to_f32"), input_info("convert_to_f32_matmul"), eltwise_mode::sum));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    prog->get_layout_optimizer().set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);
    reorder_factory rf;

    program_wrapper::apply_opt_pass<reorder_inputs>(*prog, rf);
    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(prog->get_node("matmul").get_input_layout(0).data_type == data_types::f16);
}

TEST(handle_reshape, correct_parameters_propagation) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{}, data_types::f16, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1, 12}, data_types::f16, format::bfyx });
    auto in_layout = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_layout));
    topology.add(data("data1", data1_layout));
    topology.add(eltwise("e1", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("e1"), false, {2, 12}, {2, 12}));
    topology.add(eltwise("e2", input_info("reshape"), input_info("data1"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());

    auto out_shape0 = prog->get_node("e2").get_output_pshape();
    auto out_shape1 = prog->get_node("reorder").get_output_pshape();

    ov::PartialShape expected_out_shape{2, 12};

    // handle_reshape may do reshape split, so ensure that output shape on all branches is correct
    ASSERT_EQ(out_shape0, expected_out_shape);
    ASSERT_EQ(out_shape1, expected_out_shape);
}

TEST(handle_reshape, correct_parameters_propagation_2_inputs) {
    auto& engine = get_test_engine();
    auto data0_mem = engine.allocate_memory({ ov::PartialShape{}, data_types::f16, format::bfyx });
    auto data1_mem = engine.allocate_memory({ ov::PartialShape{1, 12}, data_types::f16, format::bfyx });
    auto shape_mem = engine.allocate_memory({ ov::PartialShape{2}, data_types::i32, format::bfyx });
    auto in_layout = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx };
    set_values<int32_t>(shape_mem, {2, 12});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_mem));
    topology.add(data("data1", data1_mem));
    topology.add(data("shape", shape_mem));
    topology.add(eltwise("e1", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("e1"), input_info("shape"), false, {-1, 12}));
    topology.add(eltwise("e2", input_info("reshape"), input_info("data1"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    auto& reshape_node = prog->get_node("reshape");
    ASSERT_TRUE(reshape_node.can_be_optimized());
    ASSERT_EQ(reshape_node.get_dependencies().size(),  2);

    auto& reshape_split_node = prog->get_node("reorder").get_dependency(0);
    ASSERT_TRUE(reshape_split_node.is_type<reshape>());
    ASSERT_EQ(reshape_split_node.get_dependencies().size(),  2);

    auto out_shape0 = prog->get_node("e2").get_output_pshape();
    auto out_shape1 = prog->get_node("reorder").get_output_pshape();

    ov::PartialShape expected_out_shape{2, 12};

    // handle_reshape may do reshape split, so ensure that output shape on all branches is correct
    ASSERT_EQ(out_shape0, expected_out_shape);
    ASSERT_EQ(out_shape1, expected_out_shape);
}

TEST(handle_reshape, reshape_input_reorder) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto shape_memory = engine.allocate_memory({ ov::PartialShape{5}, data_types::i32, format::bfyx });
    auto in0_layout = layout{ ov::PartialShape{1, -1, 16, 64, 64}, data_types::f16, format::bfzyx };
    auto in0_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 16, 64, 64}, data_types::f16, format::bfzyx });
    auto in1_layout = layout{ ov::PartialShape{-1, 16, 64, 64}, data_types::f16, format::bfyx };
    auto in1_memory = engine.allocate_memory({ ov::PartialShape{2, 16, 64, 64}, data_types::f16, format::bfyx });

    auto in0 = rg.generate_random_1d<ov::float16>(in0_memory->count(), -10, 10);
    auto in1 = rg.generate_random_1d<ov::float16>(in1_memory->count(), -10, 10);
    set_values<ov::float16>(in0_memory, in0);
    set_values<int32_t>(shape_memory, {1, 2, 16, 64, 64});
    set_values<ov::float16>(in1_memory, in1);

    topology topology;
    topology.add(input_layout("input0", in0_layout));
    topology.add(input_layout("target_shape", shape_memory->get_layout()));
    topology.add(broadcast("broadcast", input_info("input0"), input_info("target_shape"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    topology.add(reshape("reshape", input_info("broadcast"), true, {-1, 16, 64, 64}, {-1, 16, 64, 64}));
    topology.add(input_layout("input1", in1_layout));
    topology.add(eltwise("eltw", input_info("reshape"), input_info("input1"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltw"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
    auto reshape_layout_in = prog->get_node("reshape").get_input_layouts()[0];
    auto reshape_layout_out = prog->get_node("reshape").get_output_layout();

    // At this moment transfomations insert reorder before reshape which
    // converts tensor to default format with rank = reshape_out_rank
    // Likely in the future we'll update that reorder so it will use reshape_input_rank
    // After that expected in format will be bfzyx
    // [Updated] get_preferred_format() updated to use 'in_lay_rank' instead of 'out_lay_rank' for preferred input format
    ASSERT_EQ(reshape_layout_in.format, format::bfzyx);
    ASSERT_EQ(reshape_layout_out.format, format::bfyx);

    ov::PartialShape expected_out_shape{-1, 16, 64, 64};
    ASSERT_EQ(reshape_layout_out.get_partial_shape(), expected_out_shape);

    network net(prog);

    net.set_input_data("input0", in0_memory);
    net.set_input_data("input1", in1_memory);
    net.set_input_data("target_shape", shape_memory);
    auto output = net.execute();

    auto out_mem = output.at("reorder").get_memory();
    mem_lock<float> lock(out_mem, get_test_stream());

    for (size_t i = 0; i < out_mem->count(); i++) {
        float expected = static_cast<float>(in0[i]) + static_cast<float>(in1[i]);
        float actual = lock[i];
        ASSERT_EQ(expected, actual) << " i = " << i;
    }
}

TEST(handle_reshape, reshape_opt_out_layout_update) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::b_fs_yx_fsv16, { 1, 512, 30, 4 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 512, 512, 3, 3 } });
    auto weights2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 512, 512, 2, 2 } });

    topology topology;
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convolution("conv1", input_info("input"), "weights1", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, ov::op::PadType::SAME_UPPER));
    topology.add(activation("relu1", input_info("conv1"), activation_func::relu));
    topology.add(pooling("pool", input_info("relu1"), pooling_mode::max, { 2, 1 }, { 2, 1 }));
    topology.add(convolution("conv2", input_info("pool"), "weights2", "", 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false, ov::op::PadType::SAME_UPPER));
    topology.add(activation("relu2", input_info("conv2"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu2"), false, {1,512,30}, {1,512,30}));
    topology.add(permute("permute", input_info("reshape"), { 2, 0, 1 }));
    topology.add(reorder("reorder", input_info("permute"), format::b_fs_yx_fsv16, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    auto reshape_layout_in = prog->get_node("reshape").get_input_layouts()[0];
    auto reshape_layout_out = prog->get_node("reshape").get_output_layout();

    // The format should have default format(bfyx) for both input/output when properly handling reshape
    ASSERT_EQ(reshape_layout_in.format, format::bfyx);
    ASSERT_EQ(reshape_layout_out.format, format::bfyx);
}
