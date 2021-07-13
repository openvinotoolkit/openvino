// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/quantize.hpp>
#include <cldnn/primitives/data.hpp>

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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2, data_types::f32)
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2, data_types::f32)
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)32);
    ASSERT_EQ(output->get_layout().count(), (size_t)32);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1_ch8_binary_pack) {
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2, data_types::bin),
        reorder("reorder", "quantize", layout{data_types::f32, format::bfyx, tensor{1,8,2,2}})
    );

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)32);
    ASSERT_EQ(output->get_layout().count(), (size_t)32);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2, data_types::f32)
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("quantize").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), (size_t)64);
    ASSERT_EQ(output->get_layout().count(), (size_t)64);

    ASSERT_EQ(output->size(), ref_data.size() * sizeof(float));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 3, data_types::f32)
    );

    network network(engine, topology);
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
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 256, data_types::u8)
    );

    network network(engine, topology);
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
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
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
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 256, data_types::u8),
        reorder("out", "quantize", format::bfzyx, data_types::u8)
    );

    network network(engine, topology);
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
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
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
    template <typename T>
    void fill_typed(memory::ptr src, memory::ptr dst) {
        auto size = dst->get_layout().size;
        size_t b = size.batch[0];
        size_t f = size.feature[0];
        size_t x = size.spatial[0];
        size_t y = size.spatial[1];

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
        auto size = mem->get_layout().size;
        size_t b = size.batch[0];
        size_t f = size.feature[0];
        size_t x = size.spatial[0];
        size_t y = size.spatial[1];

        auto data = generate_random_4d<T>(b, f, y, x, min, max, k);
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
            fill_random_typed<FLOAT16>(mem, -127, 127, 2);
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
    bool compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.size.batch[0];
        size_t f = output_lay.size.feature[0];
        size_t x = output_lay.size.spatial[0];
        size_t y = output_lay.size.spatial[1];
        mem_lock<T> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T> opt_ptr{out_opt, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto opt_out_val = opt_ptr[opt_out_offset];

                        EXPECT_EQ(opt_out_val, ref_out_val);
                    }
                }
            }
        }

        return true;
    }

    void execute_compare(const quantize_random_test_params& params, bool check_result) {
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
                quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 256, params.output_type)
            );
        } else {
            FAIL() << "Not supported inputs number: " << params.inputs_num;
        }

        auto build_ops = build_options();
        build_ops.set_option(build_option::outputs({"quantize"}));

        auto net = network(engine, topo, build_ops);
        net.set_input_data("input", input);

        auto result = net.execute();
        auto output = result.at("quantize").get_memory();

        auto input_opt = engine.allocate_memory(in_layout);
        if (params.input_type == data_types::f32) {
            fill_typed<float>(input, input_opt);
        } else if (params.input_type == data_types::f16) {
            fill_typed<FLOAT16>(input, input_opt);
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
                reorder("input_re", "input_opt", format::bfyx, params.input_type),
                data("input_low", input_low),
                data("input_high", input_high),
                data("output_low", output_low),
                data("output_high", output_high),
                quantize("quantize_opt", "input_re", "input_low", "input_high", "output_low", "output_high", 256, params.output_type),
                reorder("out", "quantize_opt", params.out_format, params.output_type)
            );
        } else {
            FAIL() << "Not supported inputs number: " << params.inputs_num;
        }

        auto buildops_opt = build_options();

        auto net_opt = network(engine, topo_opt, buildops_opt);
        net_opt.set_input_data("input_opt", input_opt);

        auto result_opt = net_opt.execute();
        auto output_opt = result_opt.at("out").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.output_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.output_type == data_types::f16) {
                compare_outputs<FLOAT16>(output, output_opt);
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
        return *this;
    }
};

TEST_P(quantize_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true);
}

INSTANTIATE_TEST_SUITE_P(quantize_smoke,
                        quantize_random_test,
                        testing::ValuesIn(
                            quantize_random_test_param_generator()
                            .simple_params(data_types::f32, data_types::u8, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, 5)
                            .simple_params(data_types::f32, data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, 5)
                        ));
