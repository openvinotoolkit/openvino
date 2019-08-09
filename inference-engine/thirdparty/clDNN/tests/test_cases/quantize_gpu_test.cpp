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

#include <api/CPP/engine.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/quantize.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include "test_utils/test_utils.h"

#include <cstddef>
#include <api/CPP/data.hpp>
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
    set_values(output_low,  { 0.0f });
    set_values(output_high, { 1.0f });

    // 0 1 1 0  0 0 0 0  0 0 0 0  0 1 1 1
    // 1 1 1 1  0 1 0 0  0 0 1 1  0 1 1 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    // 1 1 1 0  0 0 0 0  0 0 0 0  0 1 0 1
    std::vector<float> ref_data = { 0, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    1, 1, 1, 1,
                                    1, 1, 0, 0,
                                    1, 1, 1, 1 };

    topology topology;
    topology.add(
        input_layout("input", input.get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high)
    );
    topology.add(
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2)
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
    set_values(output_low,  { 0.0f });
    set_values(output_high, { 1.0f });

    std::vector<float> ref_data = { 0, 0, 0, 0,
                                    1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0 };

    topology topology;
    topology.add(
        input_layout("input", input.get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high)
    );
    topology.add(
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 2)
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
        data("output_high", output_high)
    );
    topology.add(
        quantize("quantize", "input", "input_low", "input_high", "output_low", "output_high", 3)
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
