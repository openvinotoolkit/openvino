// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

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
