// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "quantize_inst.h"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                        -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(output_low,  { -1.0f });
    set_values(output_high, {  1.0f });

    // 0 1 1 0  0 0 0 0  0 0 0 0  0 1 1 1
    // 1 1 1 1  0 1 0 0  0 0 1 1  0 1 1 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    std::vector<float> ref_data = { -1,  1,  1,  1,
                                     1,  1,  1,  1,
                                     1,  1,  1,  1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                     1,  1,  1,  1,
                                     1,  1, -1, -1,
                                     1,  1,  1,  1 };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 2, data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1_ch8) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 8, 2, 2}});
    auto input_thresh = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 8, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_thresh,  { 0.0f, 1.0f, 2.0f, 3.0f,
                                4.0f, 5.0f, 6.0f, 7.0f });

    set_values(output_low,  { -1.0f });
    set_values(output_high, {  1.0f });

    // 0 1 1 0  0 0 0 0  0 0 0 0  0 1 1 1
    // 1 1 1 1  0 1 0 0  0 0 1 1  0 1 1 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    std::vector<float> ref_data = { -1,  1,  1,  1,
                                     1,  1,  1,  1,
                                     1,  1,  1,  1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1 };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_thresh),
        data("input_high", input_thresh),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 2, data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)32);
    ASSERT_EQ(output->get_layout().count(), (size_t)32);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_2) {
    cldnn::engine& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                        -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 4.0f });
    set_values(input_high, { 4.0f });
    set_values(output_low,  { -1.0f });
    set_values(output_high, {  1.0f });

    std::vector<float> ref_data = { -1, -1, -1, -1,
                                     1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1, -1, -1, -1,
                                     1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1,  1, -1, -1,
                                    -1, -1, -1, -1 };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 2, data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(float));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_3) {
    cldnn::engine& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                        -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 0.0f, 4.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(output_low,  { 0.0f });
    set_values(output_high, { 1.0f });

    std::vector<float> ref_data = {
            0.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 0.5f, 0.5f, 0.5f,
            1.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
    };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 3, data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), ref_data.size());
    ASSERT_EQ(output->get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output->size(), ref_data.size() * sizeof(float));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}

TEST(quantize_gpu, quantize_levels_256_2d_unsigned) {
    cldnn::engine& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.1f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                         1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 10.0f, 21.0f, 32.0f, 43.0f,
                             54.0f, 65.0f, 76.0f, 87.0f,
                             87.0f, 76.0f, 65.0f, 54.0f,
                             43.0f, 32.0f, 21.0f, 10.0f });

    set_values(output_low,  { 0.0f });
    set_values(output_high, { 255.0f });

    std::vector<uint8_t> ref_data = {
            0, 54, 77, 102,
            51, 13, 13, 26,
            17, 34, 8, 8,
            0, 13, 0, 0,

            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 5, 0, 0,

            0, 0, 0, 0,
            17, 34, 8, 8,
            26, 51, 0, 0,
            26, 26, 26, 26
    };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, data_types::u8)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), ref_data.size());
    ASSERT_EQ(output->get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}

TEST(quantize_gpu, quantize_levels_256_3d_unsigned) {
    cldnn::engine& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, {1, 16, 2, 1, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { -1.0f, 2.1f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                         1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 10.0f, 21.0f, 32.0f, 43.0f,
                             54.0f, 65.0f, 76.0f, 87.0f,
                             87.0f, 76.0f, 65.0f, 54.0f,
                             43.0f, 32.0f, 21.0f, 10.0f });

    set_values(output_low,  { 0.0f });
    set_values(output_high, { 255.0f });

    std::vector<uint8_t> ref_data = {
            0, 54, 77, 102,
            51, 13, 13, 26,
            17, 34, 8, 8,
            0, 13, 0, 0,

            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 5, 0, 0,

            0, 0, 0, 0,
            17, 34, 8, 8,
            26, 51, 0, 0,
            26, 26, 26, 26
    };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, data_types::u8),
        reorder("out", input_info("quantize"), format::bfzyx, data_types::u8)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), ref_data.size());
    ASSERT_EQ(output->get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}

TEST(quantize_gpu, eltwise_quantize_fs_b_yx_fsv32) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    // conv to enable 'fs_b_yx_fsv32_network'
    const int batch_num = 2;
    const int input_xy = 5;
    const int input_f = 32;
    const int output_f = 32;
    const int filter_xy = 1;
    const int pad = filter_xy / 2;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = engine.allocate_memory({ data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    topology topology(
        input_layout("input_conv", input_mem->get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to fs_byx_fsv32
    topology.add(reorder("input_fsv", input_info("input_conv"), { data_types::f16, format::fs_b_yx_fsv32, input_size }));

    topology.add(convolution("conv0", input_info("input_fsv"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv1", input_info("conv0"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv2", input_info("conv1"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv3", input_info("conv2"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv4", input_info("conv3"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv5", input_info("conv4"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv6", input_info("conv5"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv7", input_info("conv6"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv8", input_info("conv7"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv9", input_info("conv8"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv10", input_info("conv9"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));
    topology.add(convolution("conv11", input_info("conv10"), "weights_fsv", "", 1, {1, 1}, {1, 1}, { pad, pad }, { pad, pad }, false));

    topology.add(reorder("reorder_conv", input_info("conv11"), format::b_fs_yx_fsv16, data_types::f32));

    // eltwise + quantize pattern
    auto in_layout = layout{ ov::PartialShape{2, 16, 1, 2}, data_types::f16, format::b_fs_yx_fsv16 };
    auto input = engine.allocate_memory(in_layout);
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    set_values<ov::float16>(input, {-1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                        -1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(output_low,  { -1.0f });
    set_values(output_high, {  1.0f });

    std::vector<float> ref_data = { 1, 1, 1, 1,     1, -1, -1, 1,    1, 1, 1, 1,
                                    1, 1, -1, 1,    1, -1, -1, -1,   1, 1, 1, 1,
                                    1, 1, -1, -1,  -1, -1, -1, 1,    1, 1, 1, 1,
                                    1, -1, -1, 1,   1, 1, 1, 1,      1, 1, -1, 1,
                                    1, -1, -1, -1,  1, 1, 1, 1,      1, 1, -1, -1,
                                    -1, -1, -1, 1 };

    topology.add(
        input_layout("input1", in_layout),
        input_layout("input2", in_layout),
        eltwise("multiply", input_info("input1"), input_info("input2"), eltwise_mode::prod),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("multiply"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 2, data_types::f32),
        reorder("reorder", input_info("quantize"), format::b_fs_yx_fsv16, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc quantize_impl = { format::b_fs_yx_fsv16, "quantize_gpu_ref" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "quantize", quantize_impl } }));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input_conv", input_mem);
    network.set_input_data("input1", input);
    network.set_input_data("input2", input);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, dynamic) {
    auto& engine = get_test_engine();

    auto input       = engine.allocate_memory({ { 1, 16, 2, 2 }, data_types::f32, format::bfyx });
    auto input_low   = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32, format::bfyx });
    auto input_high  = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32, format::bfyx });
    auto output_low  = engine.allocate_memory({ { 1, 1,  1, 1 }, data_types::f32, format::bfyx });
    auto output_high = engine.allocate_memory({ { 1, 1,  1, 1 }, data_types::f32, format::bfyx });

    layout in_dyn_layout { ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };

    set_values(input, { -1.0f, 2.1f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                         1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 10.0f, 21.0f, 32.0f, 43.0f,
                             54.0f, 65.0f, 76.0f, 87.0f,
                             87.0f, 76.0f, 65.0f, 54.0f,
                             43.0f, 32.0f, 21.0f, 10.0f });

    set_values(output_low,  { 0.0f });
    set_values(output_high, { 255.0f });

    std::vector<uint8_t> ref_data = {
            0, 54, 77, 102,
            51, 13, 13, 26,
            17, 34, 8, 8,
            0, 13, 0, 0,

            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 5, 0, 0,

            0, 0, 0, 0,
            17, 34, 8, 8,
            26, 51, 0, 0,
            26, 26, 26, 26
    };

    topology topology;
    topology.add(
        input_layout("input", in_dyn_layout),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 255, data_types::u8)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("quantize");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_NEAR(output_ptr[i], ref_data[i], 1) << " index = " << i;
    }
}

TEST(quantize_gpu, dynamic_fsv16) {
    auto& engine = get_test_engine();

    auto input       = engine.allocate_memory({ { 1, 16, 2, 2 }, data_types::f32, format::bfyx });
    auto input_low   = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32, format::bfyx });
    auto input_high  = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32, format::bfyx });
    auto output_low  = engine.allocate_memory({ { 1, 1,  1, 1 }, data_types::f32, format::bfyx });
    auto output_high = engine.allocate_memory({ { 1, 1,  1, 1 }, data_types::f32, format::bfyx });

    layout in_dyn_layout { ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };

    set_values(input, { -1.0f, 2.1f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f,

                         1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 2.0f, 2.0f, 3.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,

                         1.0f, 1.0f, 1.0f, 1.0f,
                         4.0f, 6.0f, 3.0f, 3.0f,
                         3.0f, 5.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f });

    set_values(input_low,  { 0.0f, 1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f, 7.0f,
                             7.0f, 6.0f, 5.0f, 4.0f,
                             3.0f, 2.0f, 1.0f, 0.0f });
    set_values(input_high, { 10.0f, 21.0f, 32.0f, 43.0f,
                             54.0f, 65.0f, 76.0f, 87.0f,
                             87.0f, 76.0f, 65.0f, 54.0f,
                             43.0f, 32.0f, 21.0f, 10.0f });

    set_values(output_low,  { 0.0f });
    set_values(output_high, { 255.0f });

    std::vector<uint8_t> ref_data = {
            0, 54, 77, 102,
            51, 13, 13, 26,
            17, 34, 8, 8,
            0, 13, 0, 0,

            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 4, 0, 0,
            0, 5, 0, 0,

            0, 0, 0, 0,
            17, 34, 8, 8,
            26, 51, 0, 0,
            26, 26, 26, 26
    };

    topology topology;
    topology.add(
        input_layout("input", in_dyn_layout),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        reorder("reorder", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
        quantize("quantize", input_info("reorder"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 255, data_types::u8),
        reorder("output_reorder", input_info("quantize"), format::bfyx, data_types::u8)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("quantize");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("output_reorder").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_NEAR(output_ptr[i], ref_data[i], 1) << " index = " << i;
    }
}

struct quantize_random_test_params {
    data_types  input_type;
    data_types  output_type;

    tensor      input_size;

    format::type in_format;
    format::type out_format;

    int32_t inputs_num;  // 5: ref
};

struct quantize_random_test : testing::TestWithParam<quantize_random_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_typed(memory::ptr src, memory::ptr dst) {
        auto l = dst->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        mem_lock<T> data{src, get_test_stream()};
        mem_lock<T> ptr{dst, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = dst->get_layout().get_linear_offset(coords);
                        auto src_offset = src->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[src_offset];
                    }
                }
            }
        }
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        mem_lock<T, mem_lock_type::read> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T, mem_lock_type::read> opt_ptr{out_opt, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto opt_out_val = opt_ptr[opt_out_offset];
                        ASSERT_NEAR(opt_out_val, ref_out_val, 1) << " index = " << opt_out_offset;
                    }
                }
            }
        }
    }

    void execute_compare(const quantize_random_test_params& params, bool check_result, bool is_caching_test) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);
        auto input = engine.allocate_memory(in_layout);
        fill_random(input);

        auto input_low   = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto input_high  = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto output_low  = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto output_high = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

        // For quantize_gpu_scale_shift_opt
        auto input_scale   = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto input_shift   = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto output_scale  = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
        auto output_shift  = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

        set_values(input_low,  { 0.0f });
        set_values(input_high, { 40.0f });
        set_values(output_low,  { 0.0f });
        set_values(output_high, { 255.0f });

        set_values(input_scale, { 2.0f });
        set_values(input_shift, { 4.0f });
        set_values(output_scale, { 2.0f });
        set_values(output_shift, { 4.0f });

        // Execute quantize_gpu_ref
        cldnn::topology topo;
        if (params.inputs_num == 5) {
            topo.add(
                input_layout("input", input->get_layout()),
                data("input_low", input_low),
                data("input_high", input_high),
                data("output_low", output_low),
                data("output_high", output_high),
                quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, params.output_type)
            );
        } else {
            FAIL() << "Not supported inputs number: " << params.inputs_num;
        }

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"quantize"}));
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input", input);

        auto result = net->execute();
        auto output = result.at("quantize").get_memory();

        auto input_opt = engine.allocate_memory(in_layout);
        if (params.input_type == data_types::f32) {
            fill_typed<float>(input, input_opt);
        } else if (params.input_type == data_types::f16) {
            fill_typed<ov::float16>(input, input_opt);
        } else if (params.input_type == data_types::i8) {
            fill_typed<int8_t>(input, input_opt);
        } else if (params.input_type == data_types::u8) {
            fill_typed<uint8_t>(input, input_opt);
        } else {
            FAIL() << "Not supported input data type: " << static_cast<size_t>(params.input_type);
        }

        cldnn::topology topo_opt;
        if (params.inputs_num == 5) {
            topo_opt.add(
                input_layout("input_opt", input_opt->get_layout()),
                reorder("input_re", input_info("input_opt"), format::bfyx, params.input_type),
                data("input_low", input_low),
                data("input_high", input_high),
                data("output_low", output_low),
                data("output_high", output_high),
                quantize("quantize_opt", input_info("input_re"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, params.output_type),
                reorder("out", input_info("quantize_opt"), params.out_format, params.output_type)
            );
        } else {
            FAIL() << "Not supported inputs number: " << params.inputs_num;
        }

        network net_opt(engine, topo_opt, get_test_default_config(engine));
        net_opt.set_input_data("input_opt", input_opt);

        auto result_opt = net_opt.execute();
        auto output_opt = result_opt.at("out").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.output_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.output_type == data_types::f16) {
                compare_outputs<ov::float16>(output, output_opt);
            } else if (params.output_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (params.output_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported output data type: " << static_cast<size_t>(params.output_type);
            }
        }
    }
};

struct quantize_random_test_param_generator : std::vector<quantize_random_test_params> {
    quantize_random_test_param_generator& simple_params(data_types input_type, data_types output_type, format::type input_format, format::type output_format, int32_t inputs_num) {
        push_back(quantize_random_test_params{ input_type, output_type, {1, 32, 2, 2}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {1, 16, 10, 10}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {64, 32, 10, 10}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {1, 17, 10, 10}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {17, 17, 10, 10}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {1, 1, 1029, 85}, input_format, output_format, inputs_num});
        push_back(quantize_random_test_params{ input_type, output_type, {1, 1, 81, 5}, input_format, output_format, inputs_num});
        return *this;
    }
};

TEST_P(quantize_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true, false);
}

INSTANTIATE_TEST_SUITE_P(quantize_smoke,
                        quantize_random_test,
                        testing::ValuesIn(
                            quantize_random_test_param_generator()
                            .simple_params(data_types::f32, data_types::u8, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, 5)
                            .simple_params(data_types::f32, data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, 5)
                            .simple_params(data_types::f32, data_types::u8, format::bfyx, format::bfyx, 5)
                            .simple_params(data_types::f16, data_types::u8, format::bs_fs_yx_bsv16_fsv32, format::bs_fs_yx_bsv16_fsv32, 5)
                        ));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(quantize_random_test, random_cached) {
    auto param = GetParam();
    execute_compare(param, true, true);
}
#else
using quantize_random_test_cached = quantize_random_test;

TEST_P(quantize_random_test_cached, random) {
    auto param = GetParam();
    execute_compare(param, true, true);
}

INSTANTIATE_TEST_SUITE_P(quantize_smoke,
                        quantize_random_test_cached,
                        testing::Values(
                            quantize_random_test_params{ data_types::f32, data_types::u8, {1, 16, 10, 10}, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, 5}
                        ));
#endif
