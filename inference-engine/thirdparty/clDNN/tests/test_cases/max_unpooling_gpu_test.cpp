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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/max_unpooling.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/reorder.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/mutable_data.hpp>
#include <api/CPP/pooling.hpp>
#include "test_utils/float16.h"

using namespace cldnn;
using namespace tests;

TEST(max_unpooling_gpu, basic_in2x3x2x2) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_output_padding) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Output Padding : 0x0x1x1

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 4.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 8.f, 16.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.5f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 12.f, 0.f, 17.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
    };
    std::vector<float> out;
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        out.push_back(output_ptr[i]);
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_output_size) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Output size explicitly provided

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", {2, 2, 3, 2}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_fp16) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Input values in fp16

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        FLOAT16(4.0f), FLOAT16(4.0f),
        FLOAT16(8.0f), FLOAT16(16.0f),
        FLOAT16(0.5f), FLOAT16(0.0f),
        FLOAT16(12.0f), FLOAT16(17.0f)
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    auto output_ptr = output.pointer<uint16_t>();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(max_unpooling_gpu, basic_in2x2x3x2_max_with_argmax_pooling_unpooling) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Forward Max Pooling Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Unpooling output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17


    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling_max_with_argmax", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));
    topology.add(max_unpooling("max_unpooling", "pooling_max_with_argmax", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();
    auto argmax_ptr = arg_max.pointer<float>();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 2);
    EXPECT_EQ(output_layout.size.spatial[0], 3);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }

    for (size_t i = 0; i < expected_argmax_vec.size(); ++i) {
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}