/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include <api/CPP/input_layout.hpp>
#include "api/CPP/strided_slice.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>


using namespace cldnn;
using namespace tests;


TEST(strided_slice_gpu_f32, test_2x2x2x2) {
    // Input (BFYX): 2x2x2x2
    // Begin (BFYX): 0x0x0x0
    // End (BFYX): 2x2x2x2
    // Stride (BFYX): 1x1x1x1
    // Output (BFYX): 2x2x2x2

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    });
    set_values(begin, {
            0, 0, 0, 0
    });
    set_values(end, {
            2, 2, 2, 2
    });
    set_values(strides, {
            1, 1, 1, 1
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, {}, {}, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(strided_slice_gpu_f32, test_2x2x2x2_2) {
    // Input (BFYX): 2x2x2x2
    // Begin (BFYX): 1x1x1x1
    // End (BFYX): 2x2x2x2
    // Stride (BFYX): 1x1x1x1
    // Output (BFYX): 1x1x1x1

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    });
    set_values(begin, {
            1, 1, 1, 1
    });
    set_values(end, {
            2, 2, 2, 2
    });
    set_values(strides, {
            1, 1, 1, 1
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, {}, {}, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = { 15.f };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(strided_slice_gpu_f32, test_2x2x4x3) {
    // Input (BFYX): 2x2x4x3
    // Begin (BFYX): 0x0x0x0
    // End (BFYX): 2x2x4x3
    // Stride (BFYX): 1x1x2x1
    // Output (BFYX): 2x2x2x3

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 4 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
            9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
            18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f,
            27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f,
            36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f,
            45.f, 46.f, 47.f
    });
    set_values(begin, {
            0, 0, 0, 0
    });
    set_values(end, {
            2, 2, 4, 3
    });
    set_values(strides, {
            1, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, {}, {}, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = {
            0.f, 1.f, 2.f, 6.f, 7.f, 8.f, 12.f, 13.f, 14.f, 18.f, 19.f, 20.f,
            24.f, 25.f, 26.f, 30.f, 31.f, 32.f, 36.f, 37.f, 38.f, 42.f, 43.f, 44.f
    };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(strided_slice_gpu_f32, test_2x2x4x4) {
    // Input (BFYX): 2x2x1x1
    // Begin (BFYX): 1x0x0x1
    // End (BFYX): 2x2x4x4
    // Stride (BFYX): 1x1x1x2
    // Output (BFYX): 1x2x2x3

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 4, 4 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
            50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
            60.0f, 61.0f, 62.0f, 63.0f
    });
    set_values(begin, {
            1, 0, 0, 1
    });
    set_values(end, {
            2, 2, 4, 4
    });
    set_values(strides, {
            1, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, {}, {}, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = {
            33.f, 34.f, 35.f, 41.f, 42.f, 43.f, 49.f, 50.f, 51.f, 57.f, 58.f, 59.f
    };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(strided_slice_gpu_f32, test_2x2x4x1_new_axis_mask) {
    // Input (BFYX): 2x2x4x1
    // New_axis_mask: 1
    // Output (BFYX): 1x2x2x4

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 4 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    });
    set_values(begin, {
            1, 0, 1, 0
    });
    set_values(end, {
            2, 2, 4, 4
    });
    set_values(strides, {
            1, 1, 1, 2
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, { 1 }, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(strided_slice_gpu_f32, test_2x2x1x1_new_axis_mask_2) {
    // Input (BFYX): 2x2x1x1
    // New_axis_mask: 101
    // Output (BFYX): 1x2x1x2

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f
    });
    set_values(begin, {
            1, 0, 1, 0
    });
    set_values(end, {
            2, 2, 4, 4
    });
    set_values(strides, {
            1, 1, 1, 2
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "input", "input2", "input3", "input4", {}, {}, { 1, 0, 1 }, {}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "strided_slice");

    auto output = outputs.at("strided_slice").get_memory();

    std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f
    };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < answers.size(); ++i)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}
