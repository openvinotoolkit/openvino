// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"

#include "data_inst.h"
#include "eltwise_inst.h"
#include "dft_inst.h"
#include "gather_inst.h"
#include "border_inst.h"
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

    layout_optimizer lo;
    auto conv_pref = lo.get_preferred_format(conv1_node);

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
    ASSERT_NO_THROW(prog = program::build_program(engine, topology, config));
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
    topology.add(gather("gather1", input_info("pad"), input_info("data1"), 2, { 1, 2, 3, 128, 57 }, 0, false));
    topology.add(gather("gather2", input_info("gather1"), input_info("data2"), 4, { 1, 2, 3, 128, 3, 55 }, 0, false));
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

    ASSERT_EQ(gather1_node.get_input_layouts()[0].format, format::bfzyx);
    ASSERT_EQ(gather1_node.get_output_layout().format, format::bfzyx);

    ASSERT_EQ(gather2_node.get_input_layouts()[0].format, format::bfwzyx);
    ASSERT_EQ(gather2_node.get_output_layout().format, format::bfwzyx);
}

TEST(reorder_inputs, impl_forcing_basic_format) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pool", input_info("input"), pooling_mode::max, { 1, 2 }, { 1, 2 }));

    ov::intel_gpu::ImplementationDesc pool_impl = { format::yxfb, "" };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"pool", pool_impl} }));

    network network(engine, topology, config);

    set_values(input, { 1.f, 2.f, 3.f, 2.f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    network.execute();

    const auto& prog = network.get_program();
    auto& pool_node = prog->get_node("pool");
    auto pool_layout = pool_node.get_output_layout();

    ASSERT_EQ(pool_layout.format.value, format::yxfb);

    auto out_mem = network.get_output("pool").get_memory();
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

    ov::intel_gpu::ImplementationDesc pool_impl = { format::any, "NOT_EXISTING" };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"pool", pool_impl} }));

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

    network network(engine, topology, config);

    set_values(input, { -1.f, 2.f, -3.f, 0.5f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    network.execute();

    auto prog = network.get_program();
    auto& node = prog->get_node("actv");
    auto actv_layout = node.get_output_layout();
    ASSERT_NE(node.get_selected_impl(), nullptr);
    auto kernel_name = node.get_selected_impl()->get_kernel_name();

    ASSERT_EQ(actv_layout.format.value, format::yxfb);
    ASSERT_EQ(kernel_name, actv_impl.kernel_name);

    auto out_mem = network.get_output("actv").get_memory();
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
