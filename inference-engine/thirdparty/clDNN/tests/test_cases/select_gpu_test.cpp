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
#include "api/select.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

// select_gpu_f32
TEST(select_gpu_f32, select_basic) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
         5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        1.f,   1.f,  1.f,  1.f,
        0.f,   1.f,  0.f,  1.f,
        1.f,   0.f,  1.f,  0.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {  0.5f,  2.5f,   0.5f,  2.5f,
                           2.f,   0.f,    6.f,   5.2f,
                          15.f,   0.5f,   8.f,  12.f,
                           4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_negative) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        -0.f,   -0.f,  -0.f,  -0.f,
        -1.f,   -1.f,  -1.f,  -1.f,
        -0.f,   -1.f,  -0.f,  -1.f,
        -1.f,   -0.f,  -1.f,  -0.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        0.1f,   0.3f,  0.5f,  0.7f,
        -0.f,   -0.1f,  -0.f,  -0.5f,
        -0.7f,   -0.f,  -1.5f,  -0.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_error_input_sizes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 3, 4, 5, 6 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    EXPECT_ANY_THROW(network(engine, topology));
}

TEST(select_gpu_f32, select_basic_error_mask_sizes) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 3, 4, 5, 6 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    EXPECT_ANY_THROW(network(engine, topology));
}

TEST(select_gpu_f32, select_basic_error_input_types) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));
    EXPECT_ANY_THROW(network(engine, topology));
}

TEST(select_gpu_f32, select_basic_error_input_formats) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    EXPECT_ANY_THROW(network(engine, topology));
}

TEST(select_gpu_f32, select_basic_byxf) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        1.f,   1.f,  1.f,  1.f,
        0.f,   1.f,  0.f,  1.f,
        1.f,   0.f,  1.f,  0.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_f16) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f16, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<uint16_t>(mask, {
        0,   0,  0,  0,
        1,   1,  1,  1,
        0,   1,  0,  1,
        1,   0,  1,  0 });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_i8) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<char>(mask, {
        0,   0,  0,  0,
        1,   1,  1,  1,
        0,   1,  0,  1,
        1,   0,  1,  0 });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_u8) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::u8, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<unsigned char>(mask, {
        0,   0,  0,  0,
        128,   210,  150,  177,
        0,   211,  0,  255,
        199,   0,  160,  0 });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,    0.f,    2.f,    0.f
    });

    set_values(input2, {
        0.5f,    2.5f,    5.f,    7.f
    });

    set_values(mask, {
        0.f,    0.f,    1.f,    1.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = { 
        0.5f,    2.5f,    2.f,    0.f
    };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f,
        2.f,   0.f
    });

    set_values(input2, {
        0.5f,   2.5f,
        5.f,   7.f
    });

    set_values(mask, {
        0.f,   0.f,
        1.f,   1.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = {
        0.5f,  2.5f,
        2.f,   0.f
    };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_byxf_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values(input, {
        1.f,   0.f,
        2.f,   0.f
    });

    set_values(input2, {
        0.5f,   2.5f,
        5.f,   7.f
    });

    set_values(mask, {
        0.f,   0.f,
        1.f,   1.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = {
        0.5f,  2.5f,
        2.f,   0.f
    };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

// select_gpu_f16
TEST(select_gpu_f16, select_basic_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<uint16_t>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    auto output_ptr = output.pointer<uint16_t>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_mask_f32_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,   0.4f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    auto output_ptr = output.pointer<uint16_t>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_mask_i8_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<char>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    auto output_ptr = output.pointer<uint16_t>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_mask_u8_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<unsigned char>(mask, {
        0,   0,
        128,   255
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    auto output_ptr = output.pointer<uint16_t>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

// select_gpu_i8
TEST(select_gpu_i8, select_basic_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<char>(mask, {
        0,   0,
        3,   5
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    auto output_ptr = output.pointer<char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_mask_f32_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,  0.4f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    auto output_ptr = output.pointer<char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_mask_f16_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<uint16_t>(mask, {
        0,   0,
        3,   5
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    auto output_ptr = output.pointer<char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_mask_u8_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<unsigned char>(mask, {
        0,   0,
        128,   255
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    auto output_ptr = output.pointer<char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

// select_gpu_u8
TEST(select_gpu_u8, select_basic_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<unsigned char>(mask, {
        0,   0,
        128,   255
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    unsigned char answers[4] = {
        0,  255,
        255,  0
    };

    auto output_ptr = output.pointer<unsigned char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_mask_f32_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,  0.4f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  255,
        255,  0
    };

    auto output_ptr = output.pointer<unsigned char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_mask_f16_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<uint16_t>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    unsigned char answers[4] = {
        0,  255,
        255,  0
    };

    auto output_ptr = output.pointer<unsigned char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_mask_i8_1x1x2x2) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = memory::allocate(engine, { data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("mask", mask.get_layout()));
    topology.add(cldnn::select("select", "input", "input2", "mask"));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<char>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    unsigned char answers[4] = {
        0,  255,
        255,  0
    };

    auto output_ptr = output.pointer<unsigned char>();

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
}
