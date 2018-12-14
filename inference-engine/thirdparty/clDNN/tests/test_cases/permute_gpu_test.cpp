/*
// Copyright (c) 2016 Intel Corporation
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
#include "api/CPP/permute.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>

#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(permute_gpu_f32, basic_bfyx_permute_0_1_3_2)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Input:
    //  f0: b0:  1    3  b1:   0    0.5
    //  f0: b0:  2    4  b1:   0    -0.5
    //  f0: b0:  -15 -15 b1:   -15  -15
    //  f1: b0:  5    7  b1:   1.5  12
    //  f1: b0:  6    8  b1:   5.2   8
    //  f1: b0:  -15 -15 b1:   -15   -15
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.5f, -0.5f, -15.f,

        1.5f,  5.2f, -15.f,
        12.0f, 8.0f, -15.f,
    });

    topology topology(
        input_layout("input", input.get_layout()),
        permute("permute", "input", { 0, 1, 3, 2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.5f,
        0.0f, -0.5f,
        -15.0f,  -15.0f,

        1.5f,  12.0f,
        5.2f, 8.0f,
        -15.0f,  -15.0f,
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 24; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_yxfb_permute_3_2_0_1)
{
    //  Input               : yxfb:2x2x2x2
    //  Permute order       : { 3,2,0,1 }
    //  Output padding      : 0x1
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Output:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        permute("permute", "input", { 3, 2, 0, 1 }, { { 0, 0, 1, 0}, 0 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[32] = {
        0.0f, 0.0f, 0.0f, 0.0f, 
        1.0f,  2.0f, 3.0f,  4.0f,
        5.0f,  6.0f, 7.0f,  8.0f, 
        0.0f, 0.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, -0.5f,
        1.5f, 5.2f, 12.0f, 8.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 32; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_bfyx_permute_0_1_3_2_input_padding)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //  Input padding       : 2x1
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Input:
    //  f0: b0:  1    3  b1:   0    0.5
    //  f0: b0:  2    4  b1:   0    -0.5
    //  f0: b0:  -15 -15 b1:   -15  -15
    //  f1: b0:  5    7  b1:   1.5  12
    //  f1: b0:  6    8  b1:   5.2   8
    //  f1: b0:  -15 -15 b1:   -15   -15
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.5f, -0.5f, -15.f,

        1.5f,  5.2f, -15.f,
        12.0f, 8.0f, -15.f,
    });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })),
        permute("permute", "reorder", { 0, 1, 3, 2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.5f,
        0.0f, -0.5f,
        -15.0f,  -15.0f,

        1.5f,  12.0f,
        5.2f, 8.0f,
        -15.0f,  -15.0f,
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 24; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

template<data_types DType>
void permute_test_with_reorder()
{
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.0f,  0.0f, -15.f,

        1.0f,  5.0f, -15.f,
        12.0f, 8.0f, -15.f,
    });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", { DType, format::bfyx,{ 2, 2, 3, 2 } }),
        permute("permute", "reorder", { 0, 1, 3, 2 }),
        reorder("reorder_out", "permute", { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_out");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.0f,
        0.0f,  0.0f,
        -15.0f,  -15.0f,

        1.0f,  12.0f,
        5.0f, 8.0f,
        -15.0f,  -15.0f,
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 24; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_i8, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i8>();
}

TEST(permute_gpu_i32, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i32>();
}

TEST(permute_gpu_i64, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i64>();
}