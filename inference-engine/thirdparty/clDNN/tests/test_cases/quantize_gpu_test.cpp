// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/engine.hpp>
#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/quantize.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include "test_utils/test_utils.h"

#include <cstddef>
#include <api/data.hpp>
#include <src/include/to_string_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<float>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), (size_t)64);
    ASSERT_EQ(output.get_layout().count(), (size_t)64);

    ASSERT_EQ(output.size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1_ch8) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 8, 2, 2}});
    auto input_thresh = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 8, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<float>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), (size_t)32);
    ASSERT_EQ(output.get_layout().count(), (size_t)32);

    ASSERT_EQ(output.size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_1_ch8_binary_pack) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 8, 2, 2}});
    auto input_thresh = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 8, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<float>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), (size_t)32);
    ASSERT_EQ(output.get_layout().count(), (size_t)32);

    ASSERT_EQ(output.size(), ref_data.size() * sizeof(uint32_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_2_output_broadcast_inputs_2) {
    const cldnn::engine& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto input_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<float>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), (size_t)64);
    ASSERT_EQ(output.get_layout().count(), (size_t)64);

    ASSERT_EQ(output.size(), ref_data.size() * sizeof(float));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " index = " << i;
    }
}

TEST(quantize_gpu, quantize_levels_3) {
    const cldnn::engine& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<float>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), ref_data.size());
    ASSERT_EQ(output.get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output.size(), ref_data.size() * sizeof(float));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}

TEST(quantize_gpu, quantize_levels_256_2d_unsigned) {
    const cldnn::engine& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<uint8_t>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), ref_data.size());
    ASSERT_EQ(output.get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output.size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}

TEST(quantize_gpu, quantize_levels_256_3d_unsigned) {
    const cldnn::engine& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfzyx, {1, 16, 2, 1, 2}});
    auto input_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

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
        input_layout("input", input.get_layout()),
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
    auto output_ptr = output.pointer<uint8_t>();

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output.count(), ref_data.size());
    ASSERT_EQ(output.get_layout().count(), ref_data.size());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output.size(), ref_data.size() * sizeof(uint8_t));

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(output_ptr[i], ref_data[i]) << " i=" << i;
    }
}
