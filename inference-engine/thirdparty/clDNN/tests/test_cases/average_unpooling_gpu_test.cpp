/*
// Copyright (c) 2018 Intel Corporation
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
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/average_unpooling.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/reorder.hpp>
#include <api/data.hpp>
#include <api/mutable_data.hpp>
#include <api/pooling.hpp>
#include "test_utils/float16.h"

using namespace cldnn;
using namespace tests;

TEST(average_unpooling_gpu, basic_in2x2x2x1) {
    //  Input  : 2x2x2x1
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    set_values(input, {
        2.5f, -4.5f,
        6.f, 4.f,
        0.f, -6.75f,
        7.0f, 4.75f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.625f, -0.5f, -1.125,
        0.625f, -0.5f, -1.125,
        1.5f, 2.5f, 1.f,
        1.5f, 2.5f, 1.f,
        0.f, -1.6875f, -1.6875f,
        0.f, -1.6875f, -1.6875f,
        1.75f, 2.9375f, 1.1875f,
        1.75f, 2.9375f, 1.1875f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x3x2_with_average_pooling_unpooling) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  3    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17
    //
    //  Input:
    //  f0: b0:  2.5 -12  b1:   0    -13
    //  f1: b0:  6    1   b1:   7     2
    //
    //  Output:
    //  f0: b0:  0.625   0.625  -6   b1:   0      0     -6.5
    //  f0: b0:  0.625   0.625  -6   b1:   0      0     -6.5
    //  f1: b0:  1.5     1.5    0.5  b1:   1.75   1.75   1
    //  f1: b0:  1.5     1.5    0.5  b1:   1.75   1.75   1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        3.f, 6.f, -12.f,
        7.f, 8.f, 14.f,
        0.f, 0.f, -11.f,
        1.f, -1.f, -15.f,
        1.5f, 5.5f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(pooling("pooling", "input", pooling_mode::average_no_padding, { 1, 1, 2, 2 }, { 1, 1, 2, 2 }));
    topology.add(average_unpooling("average_unpooling", "pooling", input.get_layout().size, { 1, 1, 2, 2 }, { 1, 1, 2, 2 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.625f, 0.625f, -6,
        0.625f, 0.625f, -6,
        1.5f, 1.5f, 0.5f,
        1.5f, 1.5f, 0.5f,
        0.f, 0, -6.5f,
        0.f, 0, -6.5f,
        1.75f, 1.75f, 1.f,
        1.75f, 1.75f, 1.f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x2x1_output_padding) {
    //  Input  : 2x2x2x1
    //  Output Padding : 0x0x1x1
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        2.5f, -4.5f,
        6.f, 4.f,
        0.f, -6.75f,
        7.0f, 4.75f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.625f, -0.5f, -1.125, 0.f,
        0.f, 0.625f, -0.5f, -1.125, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 1.5f, 2.5f, 1.f, 0.f,
        0.f, 1.5f, 2.5f, 1.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, -1.6875f, -1.6875f, 0.f,
        0.f, 0.f, -1.6875f, -1.6875f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 1.75f, 2.9375f, 1.1875f, 0.f,
        0.f, 1.75f, 2.9375f, 1.1875f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f
    };
    std::vector<float> out;
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        out.push_back(output_ptr[i]);
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x2x1_fp16) {
    //  Input  : 2x2x2x1
    //  Output Padding : 0x0x1x1
    //  Output : 2x2x3x2
    //  Input values in fp16

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        FLOAT16(2.5f), FLOAT16(-4.5f),
        FLOAT16(6.f), FLOAT16(4.f),
        FLOAT16(0.f), FLOAT16(-6.75f),
        FLOAT16(7.0f), FLOAT16(4.75f)
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    auto output_ptr = output.pointer<uint16_t>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.625f, -0.5f, -1.125,
        0.625f, -0.5f, -1.125,
        1.5f, 2.5f, 1.f,
        1.5f, 2.5f, 1.f,
        0.f, -1.6875f, -1.6875f,
        0.f, -1.6875f, -1.6875f,
        1.75f, 2.9375f, 1.1875f,
        1.75f, 2.9375f, 1.1875f
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}