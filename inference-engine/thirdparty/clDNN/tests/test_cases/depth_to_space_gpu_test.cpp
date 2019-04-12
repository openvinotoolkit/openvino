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

#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/depth_to_space.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(depth_to_space_fp16_gpu, d1411_bs2) {
    //  Input  : 1x4x1x1
    //  Block size : 2
    //  Output : 1x1x2x2
    //  Input values in fp16

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 4, 1, 1 } });
    size_t block_size = 2;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f)
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
        depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(depth_to_space_fp16_gpu, d1421_bs2) {
    //  Input  : 1x4x2x1
    //  Block size : 2
    //  Output : 1x1x4x2
    //  Input values in fp16

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 4, 1, 2 } });
    size_t block_size = 2;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f),
        FLOAT16(6.0f), FLOAT16(7.0f)
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
        depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.0f, 2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f, 7.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(depth_to_space_fp16_gpu, d1933_bs3) {
    //  Input  : 1x9x3x3
    //  Block size : 3
    //  Output : 1x1x9x9
    //  Input values in fp16

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 9, 3, 3 } });
    size_t block_size = 3;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f), FLOAT16(9.0f),
        FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f),
        FLOAT16(15.0f), FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
        FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f),
        FLOAT16(30.0f), FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f),
        FLOAT16(35.0f), FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f),
        FLOAT16(40.0f), FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f),
        FLOAT16(45.0f), FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f), FLOAT16(49.0f),
        FLOAT16(50.0f), FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),
        FLOAT16(55.0f), FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f), FLOAT16(59.0f),
        FLOAT16(60.0f), FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),
        FLOAT16(65.0f), FLOAT16(66.0f), FLOAT16(67.0f), FLOAT16(68.0f), FLOAT16(69.0f),
        FLOAT16(70.0f), FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),
        FLOAT16(75.0f), FLOAT16(76.0f), FLOAT16(77.0f), FLOAT16(78.0f), FLOAT16(79.0f),
        FLOAT16(80.0f)
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
            depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.0f, 9.0f, 18.0f, 1.0f, 10.0f, 19.0f, 2.0f, 11.0f, 20.0f, 27.0f,
        36.0f, 45.0f, 28.0f, 37.0f, 46.0f, 29.0f, 38.0f, 47.0f, 54.0f, 63.0f,
        72.0f, 55.0f, 64.0f, 73.0f, 56.0f, 65.0f, 74.0f, 3.0f, 12.0f, 21.0f,
        4.0f, 13.0f, 22.0f, 5.0f, 14.0f, 23.0f, 30.0f, 39.0f, 48.0f, 31.0f,
        40.0f, 49.0f, 32.0f, 41.0f, 50.0f, 57.0f, 66.0f, 75.0f, 58.0f, 67.0f,
        76.0f, 59.0f, 68.0f, 77.0f, 6.0f, 15.0f, 24.0f, 7.0f, 16.0f, 25.0f,
        8.0f, 17.0f, 26.0f, 33.0f, 42.0f, 51.0f, 34.0f, 43.0f, 52.0f, 35.0f,
        44.0f, 53.0f, 60.0f, 69.0f, 78.0f, 61.0f, 70.0f, 79.0f, 62.0f, 71.0f,
        80.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(depth_to_space_fp32_gpu, d1411_bs2) {
    //  Input  : 1x4x1x1
    //  Block size : 2
    //  Output : 1x1x2x2
    //  Input values in fp32

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } });
    size_t block_size = 2;

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
        depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(depth_to_space_fp32_gpu, d1421_bs2) {
    //  Input  : 1x4x2x1
    //  Block size : 2
    //  Output : 1x1x4x2
    //  Input values in fp32

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 2 } });
    size_t block_size = 2;

    set_values(input1, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
        depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 2.f, 4.f, 6.f, 1.f, 3.f, 5.f, 7.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(depth_to_space_fp32_gpu, d1933_bs3) {
    //  Input  : 1x9x3x3
    //  Block size : 3
    //  Output : 1x1x9x9
    //  Input values in fp32

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 9, 3, 3 } });
    size_t block_size = 3;

    set_values(input1, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
        30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
        40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
        50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
        60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
        70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
        80.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input1.get_layout()));
    topology.add(
        depth_to_space("depth_to_space", "Input0", block_size)
    );

    network network(engine, topology);

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.0f, 9.0f, 18.0f, 1.0f, 10.0f, 19.0f, 2.0f, 11.0f, 20.0f, 27.0f,
        36.0f, 45.0f, 28.0f, 37.0f, 46.0f, 29.0f, 38.0f, 47.0f, 54.0f, 63.0f,
        72.0f, 55.0f, 64.0f, 73.0f, 56.0f, 65.0f, 74.0f, 3.0f, 12.0f, 21.0f,
        4.0f, 13.0f, 22.0f, 5.0f, 14.0f, 23.0f, 30.0f, 39.0f, 48.0f, 31.0f,
        40.0f, 49.0f, 32.0f, 41.0f, 50.0f, 57.0f, 66.0f, 75.0f, 58.0f, 67.0f,
        76.0f, 59.0f, 68.0f, 77.0f, 6.0f, 15.0f, 24.0f, 7.0f, 16.0f, 25.0f,
        8.0f, 17.0f, 26.0f, 33.0f, 42.0f, 51.0f, 34.0f, 43.0f, 52.0f, 35.0f,
        44.0f, 53.0f, 60.0f, 69.0f, 78.0f, 61.0f, 70.0f, 79.0f, 62.0f, 71.0f,
        80.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}
