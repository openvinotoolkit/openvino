// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"

#include "data_inst.h"
#include "eltwise_inst.h"
#include "dft_inst.h"
#include "gather_inst.h"
#include "border_inst.h"
#include "reshape_inst.h"
#include "strided_slice_inst.h"
#include "batch_to_space_inst.h"
#include "permute_inst.h"
#include "concatenation_inst.h"
#include "fully_connected_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(reorder_inputs, propagation) {
    // Topology:
    // convolution -> pooling -> convolution
    //
    // Both convolutions have same parameters.
    //
    // Expectation:
    // Both convolutions should execute in the same, preferred format.
    // Format of convolutions should be propagated through pooling.
    // At most single reorder should be inserted before first convolution.

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 2, 32, 1, 1 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 32, 32, 1, 1 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convolution("conv1", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(pooling("pool", input_info("conv1"), pooling_mode::max, { 1, 1 }, { 1, 1 }));
    topology.add(convolution("conv2", input_info("pool"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    auto prog_impl = prog.get();

    size_t reorder_cnt = 0;
    for (auto node : prog_impl->get_processing_order()) {
        if (node->is_type<reorder>())
            reorder_cnt += 1;
    }
    ASSERT_LE(reorder_cnt, 1u);

    auto& conv1_node = prog_impl->get_node("conv1");
    auto& conv2_node = prog_impl->get_node("conv2");

    auto conv_pref = prog->get_layout_optimizer().get_preferred_format(conv1_node);

    ASSERT_EQ(conv1_node.get_output_layout().format.value, conv_pref);
    ASSERT_EQ(conv2_node.get_output_layout().format.value, conv_pref);

    auto& pool_node = prog_impl->get_node("pool");

    ASSERT_EQ(pool_node.get_output_layout().format.value, conv_pref);
}

TEST(reorder_inputs, mixed_ranks_irdft) {
    // Topology:
    // transpose -> (5d) -> irdft -> (4d) -> eltwise
    // Expected: (bfzyx) -> irdft -> (bfyx)

    auto& engine = get_test_engine();

    topology topology;
    topology.add(input_layout("input", layout{ { 1, 120, 2, 64, 33 }, data_types::f16, format::bfzyx }));
    topology.add(input_layout("eltw_input", layout{ { 1, 120, 64, 64 }, data_types::f16, format::bfyx }));
    topology.add(permute("permute", input_info("input"), { 0, 1, 3, 4, 2 }));
    topology.add(dft("dft", input_info("permute"), {2, 3}, {64, 64}, {1, 120, 64, 64}, dft_direction::inverse, dft_mode::real));
    topology.add(eltwise("eltwise", input_info("dft"), input_info("eltw_input"), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    program::ptr prog = nullptr;
    OV_ASSERT_NO_THROW(prog = program::build_program(engine, topology, config));
    ASSERT_NE(prog, nullptr);

    auto prog_impl = prog.get();

    auto& dft_node = prog_impl->get_node("dft");

    ASSERT_EQ(dft_node.get_input_layouts()[0].format, format::bfzyx);
    ASSERT_EQ(dft_node.get_output_layout().format, format::bfyx);
}

TEST(reorder_inputs, mixed_ranks_gather) {
    // Topology:
    // (4d) -> conv -> (4d) -> border -> (4d) -> gather -> (5d) -> gather -> (6d) -> permute (6d)
    // In case when preferred format for convolution is selected as byxf (in the test it's enforced)
    // it could be propagated to border and gathers, but dimensions are handled incorrectly
    // and the second gather may complain that axis >= rank
    // So here we expect that input format for gather is aligned with actual output rank and format

    auto& engine = get_test_engine();
    auto data1_mem = engine.allocate_memory(layout{ { 3, 128, 1, 1 }, data_types::i32, format::bfyx });
    auto data2_mem = engine.allocate_memory(layout{ { 3, 55, 1, 1 }, data_types::i32, format::bfyx });
    auto weights_mem = engine.allocate_memory(layout{ { 2, 256, 3, 3 }, data_types::f16, format::bfyx });

    topology topology;
    topology.add(input_layout("input", layout{ { 1, 256, 128, 55 }, data_types::f16, format::bfyx }));
    topology.add(data("weights", weights_mem));
    topology.add(data("data1", data1_mem));
    topology.add(data("data2", data2_mem));
    topology.add(convolution("conv",
                             input_info("input"),
                             "weights",
                             "",
                             1,
                             ov::Strides{1, 1},
                             ov::Strides{1, 1},
                             ov::CoordinateDiff{0, 0},
                             ov::CoordinateDiff{0, 0},
                             false));
    topology.add(border("pad", { input_info("conv") }, 0, ov::CoordinateDiff{0, 0, 1, 1}, ov::CoordinateDiff{0, 0, 1, 1}));
    topology.add(gather("gather1", input_info("pad"), input_info("data1"), 2, 4, { 1, 2, 3, 128, 57 }, 0, false));
    topology.add(gather("gather2", input_info("gather1"), input_info("data2"), 4, 5, { 1, 2, 3, 128, 3, 55 }, 0, false));
    topology.add(permute("permute", input_info("gather2"), {0, 1, 2, 4, 3, 5}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = { format::byxf, "" };
    ov::intel_gpu::ImplementationDesc permute_impl = { format::bfwzyx, "" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", conv_impl}, { "permute", permute_impl} }));

    program::ptr prog = nullptr;
    prog = program::build_program(engine, topology, config);
    ASSERT_NE(prog, nullptr);

    auto prog_impl = prog.get();

    auto& gather1_node = prog_impl->get_node("gather1");
    auto& gather2_node = prog_impl->get_node("gather2");

    ASSERT_EQ(gather1_node.get_input_layouts()[0].format, format::bfyx);
    ASSERT_EQ(gather1_node.get_output_layout().format, format::bfzyx);

    ASSERT_EQ(gather2_node.get_input_layouts()[0].format, format::bfzyx);
    ASSERT_EQ(gather2_node.get_output_layout().format, format::bfwzyx);
}

TEST(reorder_inputs, impl_forcing_basic_format) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pool", input_info("input"), pooling_mode::max, { 1, 2 }, { 1, 2 }));

    ov::intel_gpu::ImplementationDesc pool_impl = { format::yxfb, "", impl_types::ocl };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"pool", pool_impl} }));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);

    set_values(input, { 1.f, 2.f, 3.f, 2.f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    auto outputs = network.execute();

    const auto& prog = network.get_program();
    auto& pool_node = prog->get_node("pool");
    auto pool_layout = pool_node.get_output_layout();

    ASSERT_EQ(pool_layout.format.value, format::yxfb);

    auto out_mem = outputs.at("pool").get_memory();
    cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem_ptr.size(), 4u);

    ASSERT_EQ(out_mem_ptr[0], 2.f);
    ASSERT_EQ(out_mem_ptr[1], 7.f);
    ASSERT_EQ(out_mem_ptr[2], 3.f);
    ASSERT_EQ(out_mem_ptr[3], -1.f);
}

TEST(reorder_inputs, impl_forcing_not_existing) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pool", input_info("input"), pooling_mode::max, { 1, 2 }, { 1, 2 }));

    ov::intel_gpu::ImplementationDesc pool_impl = { format::any, "NOT_EXISTING", impl_types::ocl };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"pool", pool_impl} }));
    config.set_property(ov::intel_gpu::optimize_data(true));

    ASSERT_ANY_THROW(network network(engine, topology, config));
}

TEST(reorder_inputs, impl_forcing_basic_format_kernel) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("actv", input_info("input"), activation_func::relu));

    ov::intel_gpu::ImplementationDesc actv_impl = { format::yxfb, "activation_ref" };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"actv", actv_impl} }));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);

    set_values(input, { -1.f, 2.f, -3.f, 0.5f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto prog = network.get_program();
    auto& node = prog->get_node("actv");
    auto actv_layout = node.get_output_layout();
    ASSERT_NE(node.get_selected_impl(), nullptr);
    auto kernel_name = node.get_selected_impl()->get_kernel_name();

    ASSERT_EQ(actv_layout.format.value, format::yxfb);
    ASSERT_EQ(kernel_name, actv_impl.kernel_name);

    auto out_mem = outputs.at("actv").get_memory();
    cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem_ptr.size(), 8u);

    ASSERT_EQ(out_mem_ptr[0], 0.f);
    ASSERT_EQ(out_mem_ptr[1], 7.f);
    ASSERT_EQ(out_mem_ptr[2], 2.f);
    ASSERT_EQ(out_mem_ptr[3], 3.f);
    ASSERT_EQ(out_mem_ptr[4], 0.f);
    ASSERT_EQ(out_mem_ptr[5], 0.f);
    ASSERT_EQ(out_mem_ptr[6], 0.5f);
    ASSERT_EQ(out_mem_ptr[7], 0.f);
}

TEST(reorder_inputs, no_add_reorder_infront_of_reshape) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, 2, 7, 7, 384}, data_types::f32, format::bfwzyx };
    auto in_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 2, 7, 7, 384}, data_types::f32, format::bfwzyx });

    auto in = rg.generate_random_1d<float>(in_memory->count(), -10, 10);

    set_values<float>(in_memory, in);

    topology topology;
    topology.add(input_layout("input0", in_layout));
    topology.add(permute("permute", input_info("input0"), {0, 1, 3, 2, 4, 5}));
    topology.add(reshape("reshape", input_info("permute"), true, {1, 14, 14, 384}, {1, 14, 14, 384}));
    topology.add(eltwise("eltw", input_info("reshape"), input_info("reshape"), eltwise_mode::sum));
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

    ASSERT_EQ(reshape_layout_in.format, format::bfwzyx);
    ASSERT_EQ(reshape_layout_out.format, format::bfyx);

    auto dep_id_of_reshape = prog->get_node("reshape").get_dependencies_ids()[0];
    ASSERT_EQ(dep_id_of_reshape, "permute");

    ov::PartialShape expected_out_shape{1, 14, 14, 384};
    ASSERT_EQ(reshape_layout_out.get_partial_shape(), expected_out_shape);
}

TEST(reorder_inputs, no_need_of_reorder_for_strided_slice) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout1 = layout{ ov::PartialShape{1080, 1920, 1, 2}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ ov::PartialShape{4, 540, 960, 1, 1}, data_types::f32, format::bfzyx };
    auto data_1 = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx});
    auto data_2 = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx});
    auto data_3 = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx});

    topology topology(
        input_layout("input1", in_layout1),
        input_layout("input2", in_layout2),
        permute("permute1", input_info("input1"), {0, 1, 2, 3}),
        batch_to_space("batch_to_space1",
            input_info("input2"),
            tensor{1, 1, 4, 1, 1},
            tensor{0, 0, 1, 0, 0},
            tensor{0, 0, 1, 0, 0},
            tensor{1, 1080, 1920, 1, 2}),
        batch_to_space("batch_to_space2",
            input_info("input2"),
            tensor{1, 1, 4, 1, 1},
            tensor{0, 0, 1, 0, 0},
            tensor{0, 0, 1, 0, 0},
            tensor{1, 1080, 1920, 1, 2}),
        data("data_1", data_1),
        data("data_2", data_2),
        data("data_3", data_3),
        strided_slice("strided_slice1",
            input_info("batch_to_space1"),
            input_info("data_1"),
            input_info("data_2"),
            input_info("data_3"),
            {}, {}, {}, {}, {}, {1080, 1920, 1, 2}),
        strided_slice("strided_slice2",
            input_info("batch_to_space2"),
            input_info("data_1"),
            input_info("data_2"),
            input_info("data_3"),
            {}, {}, {}, {}, {}, {1080, 1920, 1, 2}),
        concatenation("concat", {input_info("permute1"), input_info("strided_slice1"), input_info("strided_slice2")}, 2),
        permute("result", input_info("concat"), {3, 0, 1, 2})
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<reorder_inputs>(*program, rf);

    ASSERT_NE(program, nullptr);

    auto& result = program->get_node("result");
    auto in_order = format::get_default_format(result.get_input_layout(0).get_rank()).order();
    auto out_shape = result.get_output_layout(0).get_shape();
    ASSERT_EQ(in_order.size(), out_shape.size());
}

TEST(reorder_inputs, no_need_of_reorder_to_change_input_rank_for_rdft) {
    // Topology:
    //
    // (4d)___conv___(4d)___rdft___(5d)
    //            \__(4d)___eltw___(4d)

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout1 = layout{ ov::PartialShape{1, 240, 96, 96}, data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2 = layout{ ov::PartialShape{1, 120, 96, 96}, data_types::f16, format::bfyx };
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, {120, 240, 1, 1} });
    auto bias = engine.allocate_memory({ data_types::f16, format::bfyx, {1, 120, 1, 1} });

    topology topology(
        input_layout("input1", in_layout1),
        input_layout("input2", in_layout2),
        data("weights", weights),
        data("bias", bias),
        convolution("conv", input_info("input1"), "weights", "bias", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        eltwise("eltwise", input_info("input2"), input_info("conv"), eltwise_mode::sum),
        dft("rdft", input_info("conv"), {1, 1}, {1, 1}, {1, 120, 96, 49, 2}, dft_direction::forward, dft_mode::real),
        reorder("reorder", input_info("rdft"), format::bfzyx, data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<reorder_inputs>(*program, rf);

    ASSERT_NE(program, nullptr);

    auto& dft_node = program->get_node("rdft");
    ASSERT_EQ(size_t(4), format::dimension(dft_node.get_input_layouts()[0].format));
}

TEST(reorder_inputs, add_reorder_between_single_output_type_node_and_multiple_users) {
    // Topology:
    //
    //         Add (single output)               Add
    //          |                                 |
    //  0->0 -------- 0->0    ------------>    Reorder
    //       |      |                           |   |
    //      FC      FC                         FC   FC
    //
    // Description :
    //     : Test the case where a node which doens't have muptiple output but have multiple users,
    //     : and port number to each user is same all.
    //     : In this case reorder should be inserted to each FC

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto in_layout1 = layout{ ov::PartialShape{1, 4096, 256}, data_types::i32, format::bfyx };
    auto weights = engine.allocate_memory({ data_types::i32, format::bfyx, {128, 256, 1, 1} });

    topology topology(
        input_layout("input1", in_layout1),
        input_layout("input2", in_layout1),
        data("weights1", weights),
        data("weights2", weights),
        eltwise("add", input_info("input1"), input_info("input2"), eltwise_mode::sum),
        fully_connected("fc1", input_info("add"), "weights1"),
        fully_connected("fc2", input_info("add"), "weights2")
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<reorder_inputs>(*program, rf);

    ASSERT_NE(program, nullptr);

    auto& add = program->get_node("add");
    for (auto& user : add.get_users()) {
        ASSERT_TRUE(user->is_type<reorder>());
    }

    auto& fc1 = program->get_node("fc1");
    auto& fc2 = program->get_node("fc2");

    ASSERT_TRUE(fc1.get_dependency(0).is_type<reorder>());
    ASSERT_TRUE(fc2.get_dependency(0).is_type<reorder>());
}

// TODO Not yet implemented
//TEST(reorder_inputs, impl_forcing_conv_format_kernel) {
//    auto& engine = get_test_engine();
//    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1, 2, 2, 2} });
//    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, {2, 2, 1, 1} });
//
//    topology topology;
//    topology.add(data("weights", weights));
//    topology.add(input_layout("input", input->get_layout()));
//    topology.add(convolution("conv", "input", { "weights" }));
//    topology.add(reorder("output", "conv", format::bfyx, data_types::f32));
//
//    graph graph(engine, topology);
//    auto possible_impls = graph.get_implementations("conv");
//
//    set_values(input, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f });
//    set_values(weights, { 1.f, 2.f, 3.f, 4.f });
//
//    for (auto impl : possible_impls) {
//        SCOPED_TRACE(to_string(impl));
//
//        ExecutionConfig config = get_test_default_config(engine);
//        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", impl} }));
//
//        network network(engine, topology, config);
//
//        network.set_input_data("input", input);
//        network.execute();
//
//
//        auto network = api_cast(network.get());
//        auto& prog = network->get_program();
//        auto& conv_node = prog.get_node("conv");
//        auto conv_sel_impl = conv_node.get_selected_impl();
//        auto conv_layout = conv_node.get_output_layout();
//
//        ASSERT_EQ(conv_layout.format.value, impl.format);
//        ASSERT_EQ(conv_sel_impl->get_kernel_name(), impl.kernel);
//
//        auto out_mem = network.get_output("output").get_memory();
//        cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());
//
//        ASSERT_EQ(out_mem_ptr.size(), 8);
//
//        ASSERT_EQ(out_mem_ptr[0], 11.f);
//        ASSERT_EQ(out_mem_ptr[1], 14.f);
//        ASSERT_EQ(out_mem_ptr[2], 17.f);
//        ASSERT_EQ(out_mem_ptr[3], 20.f);
//
//        ASSERT_EQ(out_mem_ptr[4], 23.f);
//        ASSERT_EQ(out_mem_ptr[5], 30.f);
//        ASSERT_EQ(out_mem_ptr[6], 37.f);
//        ASSERT_EQ(out_mem_ptr[7], 44.f);
//    }
//}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(reorder_inputs, has_reshape_user) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
       return;

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 4, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights)),
    topology.add(data("biases", biases)),
    topology.add(convolution("conv", input_info("input"), "weights", "biases", 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, false));
    topology.add(reshape("reshape1", input_info("conv"), false, { 1, 1, 3, 3, 3 }, { 1, 1, 3, 3, 3 }));
    topology.add(permute("permute", input_info("reshape1"), { 0, 1, 2, 3, 4 }));
    topology.add(reshape("reshape2", input_info("permute"), false, { 1, 3, 3, 3 }, { 1, 3, 3, 3 }));
    topology.add(reorder("output", input_info("reshape2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input", input);

    primitive_id out_id = "output";
    auto output = network.execute();
    auto out_l = network.get_output_layout(out_id);
    auto out_mem = output.at(out_id).get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    std::vector<int> ref_output = {
        3, 2, 2, 6,  5,  6, 9, 4, 6,
        5, 2, 5, 10, 9,  5, 7, 5, 4,
        3, 4, 6, 6,  5, 10, 9, 4, 1
    };

    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(static_cast<float>(ref_output[x]), output_ptr[x]);
    }
}

TEST(reorder_inputs, two_connections_with_different_format) {
    // Topology:
    // convolution(fsv16) ___ convolution
    //                    \__ deformable_conv
    //                     \_ reshape
    // Purpose:
    // When convolution has reshape as a user, its layout may be chosen in a confusing way from get_preferred_format.
    // This test mimics the behavior.
    //
    // Expectation:
    // Reorder should be added only to deformable_conv as deformable_conv supports bfyx only.

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 128, 128 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 32, 32, 1, 1 } });
    auto trans = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 2, 128, 128 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(data("trans", trans));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convolution("conv1", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv2", input_info("conv1"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("deform_conv", {input_info("conv1"), input_info("trans")}, "weights", "", true, 1, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}));
    topology.add(reshape("reshape", input_info("conv1"), tensor(2, 16, 128, 128), cldnn::reshape::reshape_mode::base));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    auto& node = prog->get_node("deform_conv");
    ASSERT_NE(node.get_selected_impl(), nullptr);
    auto kernel_name = node.get_selected_impl()->get_kernel_name();
    ASSERT_EQ(kernel_name, "deformable_convolution_gpu_bfyx_opt");
}
#endif
