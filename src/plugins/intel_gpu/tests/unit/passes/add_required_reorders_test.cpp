// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gather_inst.h"
#include "pass_manager.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "convolution_inst.h"
#include "dft_inst.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(add_required_reorders, input_reorder_inside_shape_of_subgraph) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 32, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{1, 32, 32, 32}, data_types::f16, format::bfyx});
    auto data_0 = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{}, data_types::f32, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather0", input_info("shape_of"), input_info("data_0"), 0, {}, {}, 0, true));
    topology.add(eltwise("eltwise0", {input_info("gather0"), input_info("data_1")}, eltwise_mode::prod, data_types::f32));
    topology.add(reshape("reshape0", input_info("eltwise0"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(gather("gather1", input_info("shape_of"), input_info("data_0"), 0, {}, {}, 0, true));
    topology.add(eltwise("eltwise1", {input_info("gather1"), input_info("data_1")}, eltwise_mode::prod, data_types::f32));
    topology.add(reshape("reshape1", input_info("eltwise1"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(concatenation("concat0", {input_info("reshape0"), input_info("reshape1")}, 0, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    auto& eltwise_node = prog->get_node("eltwise0");
    auto eltwise_in_layout = eltwise_node.get_input_layout();

    ASSERT_EQ(eltwise_in_layout.data_type, data_types::f32);
}

TEST(add_required_reorders, prevent_input_dt_changing_for_convs) {
    auto& engine = get_test_engine();

    int input_b = 1, input_f = 16, input_y = 3, input_x = 3;
    int output_b = input_b, output_f = 16, output_y = 6, output_x = 6;

    auto input_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto input2_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto weights_mem = engine.allocate_memory({ {16, 16, 1, 1}, data_types::i8, format::bfyx });

    auto input = input_layout("input", input_mem->get_layout());
    auto input_const = data("input_const", input2_mem);
    auto weights = data("weights", weights_mem);
    auto eltwise1 = eltwise("eltwise1", input_info("input"), input_info("input_const"), eltwise_mode::sum);
    auto conv1 = convolution("conv1", input_info("eltwise1"), "weights", "", 1, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, false);
    auto output_reorder = reorder("reorder", input_info("conv1"), { data_types::f32, format::bfyx, { output_b, output_f, output_y, output_x } });

    topology topology_test(input, input_const, eltwise1, weights, conv1, output_reorder);

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv1_impl_test = { format::bfyx, "", impl_types::ocl };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv1", conv1_impl_test } }));

    auto prog = program::build_program(engine, topology_test, config_test, false, true);
    program_wrapper::apply_opt_pass<add_required_reorders>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("conv1"));
    ASSERT_EQ(prog->get_node("conv1").get_input_layout(0).data_type, data_types::u8);
}

TEST(add_required_reorders, prevent_users_invalidation) {
    auto& engine = get_test_engine();

    // Create padded input memory
    auto input_mem_padded = engine.allocate_memory({ {1, 16, 8, 8}, data_types::f16, format::bfyx, padding{{0, 0, 2, 2}, 0} });
    auto weights_mem = engine.allocate_memory({ {16, 16, 1, 1}, data_types::f16, format::bfyx });

    auto input = input_layout("input", input_mem_padded->get_layout());
    auto weights = data("weights", weights_mem);
    auto conv = convolution("conv", input_info("input"), "weights", "", 1, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, false);

    auto prog = program::build_program(engine,
                                       topology(input, weights, conv),
                                       get_test_default_config(engine),
                                       false,
                                       true);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("conv"));

    const auto& conv_node = prog->get_node("conv");

    // Force OneDNN impl type to insert padded_layout -> non_padded_layout reorder
    prog->get_node("conv").set_forced_impl_type(impl_types::onednn);

    program_wrapper::apply_opt_pass<add_required_reorders>(*prog);

    const auto& conv_input = conv_node.get_dependency(0);

    ASSERT_TRUE(conv_input.is_type<reorder>());
    ASSERT_TRUE(conv_input.is_valid_output_layout());
}

TEST(add_required_reorders, skip_adding_reorder_batch_axis_padding) {
    auto& engine = get_test_engine();

    auto in_input = engine.allocate_memory({ data_types::i32, format::bfzyx, tensor{ 3, 6, 2, 2, 2 } });

    layout reorder_layout(data_types::f32, format::bfyx, { 1, 6, 4, 2 });

    tests::set_random_values<int32_t>(in_input);

    topology topology;
    topology.add(input_layout("Input", in_input->get_layout()));
    topology.add(reorder("reorder_input", input_info("Input"), format::bfzyx, data_types::f32));
    topology.add(crop("crop1", input_info("reorder_input"), tensor{1, 6, 2, 2, 2}, tensor(1, 0, 0, 0, 0)));
    topology.add(reorder("crop1_reorder", input_info("crop1"), reorder_layout));
    topology.add(reshape("reshape1", input_info("crop1_reorder"), tensor(6, 2, 2, 2)));
    topology.add(crop("crop2", input_info("reorder_input"), tensor{1, 6, 2, 2, 2}, tensor(2, 0, 0, 0, 0)));
    topology.add(reorder("crop2_reorder", input_info("crop2"), reorder_layout));
    topology.add(reshape("reshape2", input_info("crop2_reorder"), tensor(6, 2, 2, 2)));
    topology.add(concatenation("concat", { input_info("reshape1"), input_info("reshape2") }, 1, data_types::f32));
    topology.add(reorder("reorder_output", input_info("concat"), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);

    network.set_input_data("Input", in_input);
    auto outputs = network.execute();
    auto output = outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    ExecutionConfig ref_config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);

    ref_network.set_input_data("Input", in_input);
    auto ref_outputs = ref_network.execute();
    auto ref_output = ref_outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> ref_output_ptr(ref_output, get_test_stream());

    int crop_batch_num = 6;
    int crop_feature_num = 2;
    int crop_y_size = 2;
    int crop_x_size = 2;
    for (int b = 0; b < crop_batch_num; ++b) {
        for (int f = 0; f < crop_feature_num; ++f) {
            for (int y = 0; y < crop_y_size; ++y) {
                for (int x = 0; x < crop_x_size; ++x) {
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    ASSERT_EQ(output_ptr[output_linear_id], ref_output_ptr[output_linear_id]);
                }
            }
        }
    }

    // optimized reorder, concate
    auto crop_prim = network.get_primitive("crop1");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    crop_prim = network.get_primitive("crop2");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    auto reorder_prim = network.get_primitive("crop1_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), true);
    reorder_prim = network.get_primitive("crop2_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), true);
    auto concate = network.get_primitive("concat");
    ASSERT_EQ(concate->can_be_optimized(), false);
}
