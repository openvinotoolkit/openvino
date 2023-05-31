// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "reorder_inst.h"
#include "softmax_inst.h"
#include "reduce_inst.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "activation_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(remove_redundant_reorders, remove_dep_dynamic) {
    // Topology:
    // convolution -> reorder -> softmax
    //
    // Expectation:
    // The preferred format of convolution should be selected as b_fs_yx_fsv16 (reorder_inputs)
    // A new reorder that converts to bfyx should be inserted after convolution (reorder_inputs)
    // In reorders, output format of dependency reorder should be saved as output_format of orginial reorder (remove_redundant_reorders)

    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 224, 224 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 64, 3, 7, 7 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder", input_info("conv"), format::any, data_types::f32));
    topology.add(softmax("softmax", input_info("reorder"), 1));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);
    auto& softmax_node = prog->get_node("softmax");
    auto softmax_layout = softmax_node.get_output_layout();

    ASSERT_EQ(softmax_layout.format.value, format::bfyx);
}

TEST(remove_redundant_reorders, optimize_fsv16_to_bfyx) {
    // Topology:
    // reorder(b_fs_yx_fsv16) -> reduce(b_fs_yx_fsv16) -> fully_connected(bfyx)
    //
    // Expectation:
    // Reorder that converts b_fs_yx_fsv16 to bfyx is added between reduce and fc (add_required_reorders)
    // If it is post_optimize_graph phase and the batch size of reorder output layout is not 1,
    // reorder optimization (b_fs_yx_fsv16->bfyx when spatials are eqaul to 1) is skipped (remove_redundant_reorders)
    // So there should be no upper padding for feature dim of FC's input layout

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1080, 7, 7 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1001, 1080, 1, 1 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), format::b_fs_yx_fsv16, data_types::f32));
    topology.add(reduce("reduce", input_info("reorder"), reduce_mode::min, {2, 3}, true));
    topology.add(fully_connected("fc", input_info("reduce"), "weights"));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);
    auto& fc_node = prog->get_node("fc");
    auto fc_in_layout = fc_node.get_input_layouts();
    ASSERT_EQ(fc_in_layout.front().data_padding.upper_size().feature[0], 0);
}

TEST(remove_redundant_reorders, skip_reorder_fusing_when_sibling_not_support_padding) {
    // Reorder fusing with padding in remove_redundant_reorders pass should check all sibiling nodes whether they support padding or not.
    // This test case has two reorders after convolution and one has padding. This reorder shouldn't be fused in the pass.
    // Reference model : Enhance3-lite

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 32, 480, 270 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 16, 32, 1, 1 } });
    auto weights_2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 64, 16, 3, 3 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(data("weights_2", weights_2));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convolution("convolution", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false, ov::op::PadType::EXPLICIT));
    topology.add(reorder("reorder_reshape_1", input_info("convolution"), { data_types::f16, format::bfwzyx, { 2, 16, 1, 1, 480, 270 } }));
    topology.add(permute("transpose_1", input_info("reorder_reshape_1"), { 0, 1, 2, 3, 5, 4 }));
    topology.add(reorder("convolution_reorder_1", input_info("convolution"),
                        { data_types::f16, format::fs_b_yx_fsv32, { 2, 16, 480, 270 }, padding({0, 0, 1, 1}, 0) }));
    topology.add(convolution("convolution_2", input_info("convolution_reorder_1"),
                             "weights_2", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false, ov::op::PadType::EXPLICIT, padding({0, 0, 1, 1}, 0)));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto prog = program::build_program(engine, topology, config, false, true);
    config.set_property(ov::intel_gpu::optimize_data(true));

    layout_optimizer lo(true);

    bool optimize_data = config.get_property(ov::intel_gpu::optimize_data);
    program_wrapper::apply_opt_pass<remove_redundant_reorders>(*prog, lo, optimize_data);

    ASSERT_NE(prog, nullptr);

    ASSERT_EQ(prog->get_node("convolution").get_output_layout().data_padding, padding());
}

TEST(remove_redundant_reorders, not_to_fuse_reshape_with_fused_prims) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{1, 32, 2, 2}, data_types::f16, format::bfyx });
    auto in_layout = layout{ ov::PartialShape{1, 32, 2, 2}, data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_layout));
    topology.add(eltwise("elt", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("elt"), { data_types::f16, format::bfzyx, {1, 1, 32, 2, 2}}));
    topology.add(reshape("reshape1", input_info("reorder"), {1, 4, 16, 2}));
    topology.add(reorder("reorder2", input_info("reshape1"), { data_types::f16, format::bfzyx, {1, 1, 32, 2, 2}}));
    topology.add(reshape("reshape2", input_info("reorder2"), {1, 32, 2, 2, 1}));
    topology.add(activation("activation", input_info("reshape2"), activation_func::relu));
    topology.add(reorder("reorder4", input_info("activation"), { data_types::f32, format::bfyx, {1, 4, 32, 1}}));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);
    bool optimize_data = config.get_property(ov::intel_gpu::optimize_data);
    program_wrapper::apply_opt_pass<remove_redundant_reorders>(*prog, lo, optimize_data);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));
    network network(engine, topology, config);

    auto input = engine.allocate_memory(in_layout);
    VVVVF<float> input_all_neg = generate_random_4d<float>(1, 32, 2, 2, -10.f, 0.f);
    set_values(input, input_all_neg);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_GE(output_ptr[i], 0);
    }
}

TEST(remove_redundant_reorders, not_to_fuse_permute) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f16, format::b_fs_zyx_fsv16, {2, 256, 2, 8, 8}});
    auto weight = engine.allocate_memory({data_types::f16, format::bfzyx, {1, 256, 1, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(
        reorder("reorder1", input_info("convolution"), {data_types::f16, format::b_fs_zyx_fsv16, {2, 256, 2, 8, 8}}));
    topology.add(reorder("reorder2", input_info("reorder1"), {data_types::f16, format::bfwzyx, {2, 2, 1, 8, 8, 256}}));
    topology.add(permute("permute", input_info("reorder2"), {0, 3, 2, 4, 5, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    layout_optimizer lo(true);
    bool opt_data = config.get_property(ov::intel_gpu::optimize_data);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);
    program_wrapper::apply_opt_pass<remove_redundant_reorders>(*prog, lo, opt_data);

    auto& node = prog->get_node("permute");
    auto in_layout = node.get_input_layouts()[0];
    ASSERT_EQ(in_layout.format.value, format::bfwzyx);

    network network(engine, topology, config);
}

TEST(remove_redundant_reorders, remove_fused) {
    auto& engine = get_test_engine();
    layout output_layout_fp16( data_types::f16, format::bfyx, { 1, 3, 2, 2 } );
    layout output_layout_fp32( data_types::f32, format::bfyx, { 2, 3, 1, 2 } );

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 2, 2 } });
    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), output_layout_fp16));
    topology.add(activation("act", input_info("reorder1"), activation_func::relu));
    topology.add(reorder("reorder2", input_info("input1"), output_layout_fp16));
    topology.add(eltwise("sum", input_info("reorder2"), input_info("act"), eltwise_mode::sum));
    topology.add(reorder("reorder3", input_info("sum"), output_layout_fp32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);
    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);
    bool optimize_data = config.get_property(ov::intel_gpu::optimize_data);
    program_wrapper::apply_opt_pass<remove_redundant_reorders>(*prog, lo, optimize_data);

    ASSERT_NE(prog, nullptr);
    network network(engine, topology, config);
    ASSERT_TRUE(has_node(*prog, "reorder2"));
}
