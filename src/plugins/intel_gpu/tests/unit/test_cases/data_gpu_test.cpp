// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "test_utils.h"
#include "random_generator.hpp"
#include "program_wrapper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(data_gpu, attach_host_buffer) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{6, 4}, data_types::f32, format::bfyx};
    auto weights_layout = layout{ov::PartialShape{2, 4}, data_types::f32, format::bfyx};
    std::vector<float> weights_data = {1.5f, 1.0f, -1.0f, 0.0f,
                                       0.5f, -0.5f, -0.5f, 1.0f };
    auto input_memory = engine.allocate_memory(in_layout);
    auto weights_memory = engine.attach_memory(weights_layout, weights_data.data());

    set_values(input_memory, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                              -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                              -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});

    cldnn::topology topology{
        input_layout("input", in_layout),
        data("weights", weights_memory),
        fully_connected("fc", input_info("input"), "weights", "", in_layout.get_partial_shape().size()),
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_memory);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ov::Shape expected_shape = {6, 2};
    ASSERT_EQ(output_prim_mem->get_layout().get_shape(), expected_shape);

    std::vector<float> expected_output = { 0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5 };

    cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

    for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
        ASSERT_EQ(expected_output[i], output_ptr[i]);
    }
}

TEST(data_gpu, usm_device_buffer) {
    auto& engine = get_test_engine();
    if (!engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP();

    auto in_layout = layout{ov::PartialShape{6, 4}, data_types::f32, format::bfyx};
    auto weights_layout = layout{ov::PartialShape{2, 4}, data_types::f32, format::bfyx};
    std::vector<float> weights_data = {1.5f, 1.0f, -1.0f, 0.0f,
                                       0.5f, -0.5f, -0.5f, 1.0f };
    auto input_memory = engine.allocate_memory(in_layout);
    auto weights_memory = engine.allocate_memory(weights_layout, allocation_type::usm_device, false);

    weights_memory->copy_from(get_test_stream(), weights_data.data(), true);

    set_values(input_memory, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                              -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                              -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});

    cldnn::topology topology{
        input_layout("input", in_layout),
        data("weights", weights_memory),
        fully_connected("fc", input_info("input"), "weights", "", in_layout.get_partial_shape().size()),
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_memory);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ov::Shape expected_shape = {6, 2};
    ASSERT_EQ(output_prim_mem->get_layout().get_shape(), expected_shape);

    std::vector<float> expected_output = { 0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5 };

    cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

    for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
        ASSERT_EQ(expected_output[i], output_ptr[i]);
    }
}

TEST(data_gpu, transfer_memory_padding_mismatch_forced) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().has_separate_cache) {
        GTEST_SKIP() << "Device doesn't have separate cache, test not applicable";
    }

    if (!engine.supports_allocation(allocation_type::usm_device)) {
        GTEST_SKIP() << "USM device allocation not supported, skipping test";
    }

    layout mem_layout_no_pad{data_types::f32, format::bfyx, {1, 1, 1, 91}};
    auto const_mem = engine.allocate_memory(mem_layout_no_pad, allocation_type::usm_host, false);

    std::vector<float> const_data(mem_layout_no_pad.count(), 1.0f);
    const_mem->copy_from(get_test_stream(), const_data.data());
    get_test_stream().finish();

    cldnn::topology topology;
    topology.add(input_layout("input", layout{data_types::f32, format::bfyx, {1, 1, 1, 91}}));
    topology.add(data("const_data", const_mem));
    topology.add(eltwise("output", {input_info("input"), input_info("const_data")}, eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto prog = std::make_shared<program>(engine, topology, config, nullptr, nullptr, false, false, false);

    auto& data_node = prog->get_node("const_data");
    padding node_padding;
    node_padding._lower_size = {0, 0, 0, 0};
    node_padding._upper_size = {0, 16, 0, 1};
    layout node_layout_with_pad{data_types::f32, format::bfyx, {1, 1, 1, 91}, node_padding};
    data_node.set_output_layout(node_layout_with_pad, false);

    bool exception_thrown = false;
    std::string exception_msg;

    try {
        program_wrapper::init_graph(*prog);
        auto& prog_config = const_cast<ExecutionConfig&>(prog->get_config());
        prog_config.finalize(prog->get_engine());
        prog->get_engine().set_enable_large_allocations(prog_config.get_enable_large_allocations());
        program_wrapper::pre_optimize_graph(*prog, false);
        program_wrapper::run_graph_compilation(*prog);
        program_wrapper::post_optimize_graph(*prog, false);
        program_wrapper::prepare_memory_dependencies(*prog);
        program_wrapper::apply_opt_pass<build_implementations>(*prog);

        if (prog->get_engine().get_device_info().has_separate_cache) {
            program_wrapper::transfer_memory_to_device(*prog);
        }
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
        exception_msg = e.what();
    }
    ASSERT_FALSE(exception_thrown) << "Padding fix failed! Exception: " << exception_msg;
}
