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
    topology.add(convolution("conv1", "input", { "weights" }));
    topology.add(pooling("pool", "conv1", pooling_mode::max, { 1, 1 }, { 1, 1 }));
    topology.add(convolution("conv2", "pool", { "weights" }));

    build_options build_opts;
    build_opts.set_option(build_option::optimize_data(true));
    auto prog = program::build_program(engine, topology, build_opts);

    auto prog_impl = prog.get();

    size_t reorder_cnt = 0;
    for (auto node : prog_impl->get_processing_order()) {
        if (node->is_type<reorder>())
            reorder_cnt += 1;
    }
    EXPECT_LE(reorder_cnt, 1u);

    auto& conv1_node = prog_impl->get_node("conv1");
    auto& conv2_node = prog_impl->get_node("conv2");

    layout_optimizer lo;
    auto conv_pref = lo.get_preferred_format(conv1_node);

    ASSERT_EQ(conv1_node.get_output_layout().format.value, conv_pref);
    ASSERT_EQ(conv2_node.get_output_layout().format.value, conv_pref);

    auto& pool_node = prog_impl->get_node("pool");

    EXPECT_EQ(pool_node.get_output_layout().format.value, conv_pref);
}

TEST(reorder_inputs, impl_forcing_basic_format) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pool", "input", pooling_mode::max, { 1, 2 }, { 1, 2 }));

    implementation_desc pool_impl = { format::yxfb, "" };

    build_options build_opts;
    build_opts.set_option(build_option::force_implementations({ {"pool", pool_impl} }));

    network network(engine, topology, build_opts);

    set_values(input, { 1.f, 2.f, 3.f, 2.f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    network.execute();

    const auto& prog = network.get_program();
    auto& pool_node = prog->get_node("pool");
    auto pool_layout = pool_node.get_output_layout();

    EXPECT_EQ(pool_layout.format.value, format::yxfb);

    auto out_mem = network.get_output("pool").get_memory();
    cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem_ptr.size(), 4u);

    EXPECT_EQ(out_mem_ptr[0], 2.f);
    EXPECT_EQ(out_mem_ptr[1], 7.f);
    EXPECT_EQ(out_mem_ptr[2], 3.f);
    EXPECT_EQ(out_mem_ptr[3], -1.f);
}

TEST(reorder_inputs, impl_forcing_not_existing) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pool", "input", pooling_mode::max, { 1, 2 }, { 1, 2 }));

    implementation_desc pool_impl = { format::any, "NOT_EXISTING" };

    build_options build_opts;
    build_opts.set_option(build_option::force_implementations({ {"pool", pool_impl} }));

    ASSERT_ANY_THROW(network network(engine, topology, build_opts));
}

TEST(reorder_inputs, impl_forcing_basic_format_kernel) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("actv", "input", activation_func::relu));

    implementation_desc actv_impl = { format::yxfb, "activation_ref" };

    build_options build_opts;
    build_opts.set_option(build_option::force_implementations({ {"actv", actv_impl} }));

    network network(engine, topology, build_opts);

    set_values(input, { -1.f, 2.f, -3.f, 0.5f,
                        7.f, 3.f, -2.f, -1.f });

    network.set_input_data("input", input);
    network.execute();

    auto prog = network.get_program();
    auto& node = prog->get_node("actv");
    auto actv_layout = node.get_output_layout();
    auto kernel_name = node.get_selected_impl()->get_kernel_name();

    EXPECT_EQ(actv_layout.format.value, format::yxfb);
    EXPECT_EQ(kernel_name, actv_impl.kernel_name);

    auto out_mem = network.get_output("actv").get_memory();
    cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());

    ASSERT_EQ(out_mem_ptr.size(), 8u);

    EXPECT_EQ(out_mem_ptr[0], 0.f);
    EXPECT_EQ(out_mem_ptr[1], 7.f);
    EXPECT_EQ(out_mem_ptr[2], 2.f);
    EXPECT_EQ(out_mem_ptr[3], 3.f);
    EXPECT_EQ(out_mem_ptr[4], 0.f);
    EXPECT_EQ(out_mem_ptr[5], 0.f);
    EXPECT_EQ(out_mem_ptr[6], 0.5f);
    EXPECT_EQ(out_mem_ptr[7], 0.f);
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
//        build_options build_opts;
//        build_opts.set_option(build_option::force_implementations({ {"conv", impl} }));
//
//        network network(engine, topology, build_opts);
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
//        EXPECT_EQ(conv_layout.format.value, impl.format);
//        EXPECT_EQ(conv_sel_impl->get_kernel_name(), impl.kernel);
//
//        auto out_mem = network.get_output("output").get_memory();
//        cldnn::mem_lock<float> out_mem_ptr(out_mem, get_test_stream());
//
//        EXPECT_EQ(out_mem_ptr.size(), 8);
//
//        EXPECT_EQ(out_mem_ptr[0], 11.f);
//        EXPECT_EQ(out_mem_ptr[1], 14.f);
//        EXPECT_EQ(out_mem_ptr[2], 17.f);
//        EXPECT_EQ(out_mem_ptr[3], 20.f);
//
//        EXPECT_EQ(out_mem_ptr[4], 23.f);
//        EXPECT_EQ(out_mem_ptr[5], 30.f);
//        EXPECT_EQ(out_mem_ptr[6], 37.f);
//        EXPECT_EQ(out_mem_ptr[7], 44.f);
//    }
//}
