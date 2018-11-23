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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/CPP/engine.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/border.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include "test_utils/test_utils.h"
#include "test_utils/uniform_quantized_real_distribution.hpp"

#include <cstddef>


using namespace cldnn;
using namespace ::tests;


template<typename T>
static std::vector<T> generate_rnd_real_input(
    const std::size_t b, const std::size_t f, const std::size_t y, const std::size_t x,
    const T min = static_cast<T>(0), const T max = static_cast<T>(1), const unsigned rnd_bits = 9)
{
    static std::default_random_engine rnd_gen(random_seed);
    cldnn::tests::distributions::uniform_quantized_real_distribution<T> rnd_dist(min, max, rnd_bits);

    std::vector<T> data;
    data.reserve(b * f * y * x);
    for (size_t i = 0; i < b * f * y * x; ++i)
        data.push_back(rnd_dist(rnd_gen));

    return data;
}


TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_zero) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::zero)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   1, -2,  3,  -4, 0, 0, 0, 0,
        0, 0,   5,  6,  7,   8, 0, 0, 0, 0,
        0, 0, -10, 12, 13, -13, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_mirror) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
         6,   5,   5,  6,  7,   8,   8,  7,  6,   5,
        12, -10, -10, 12, 13, -13, -13, 13, 12, -10,
        12, -10, -10, 12, 13, -13, -13, 13, 12, -10,
         6,   5,   5,  6,  7,   8,   8,  7,  6,   5,
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_mirror_101) {
    //  Input (XY) : 5x4
    //  Output (XY): 11x8

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 4;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror_101)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,  4,
          5,  6,  7,   8, -8,
        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
    };
    std::vector<float> out_data = {
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
         3, -2,   1, -2,  3,  -4,  4,  -4,  3, -2,   1,
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
        13, 12, -10, 12, 13, -13, 10, -13, 13, 12, -10,
        23, 22, -20, 22, 23, -23, 20, -23, 23, 22, -20,
        13, 12, -10, 12, 13, -13, 10, -13, 13, 12, -10,
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
         3, -2,   1, -2,  3,  -4,  4,  -4,  3, -2,   1,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}
TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_zero) {
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 3;
    constexpr auto in_size_y = 5;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::zero)
    );

    std::vector<float> input_data = generate_rnd_real_input<float>(in_size_b, in_size_f, in_size_y, in_size_x, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    if (b < blt_size_b || b >= out_size_b - brb_size_b ||
                        f < blt_size_f || f >= out_size_f - brb_size_f ||
                        y < blt_size_y || y >= out_size_y - brb_size_y ||
                        x < blt_size_x || x >= out_size_x - brb_size_x)
                    {
                        EXPECT_EQ(output_ptr[output_off], 0.0f);
                    }
                    else
                    {
                        auto input_off  = (((b - blt_size_b) * in_size_f + f - blt_size_f) * in_size_y + y - blt_size_y) * in_size_x + x - blt_size_x; // BFYX
                        EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_mirror) {
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 3;
    constexpr auto in_size_y = 5;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror)
    );

    std::vector<float> input_data = generate_rnd_real_input<float>(in_size_b, in_size_f, in_size_y, in_size_x, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - 1 - b : in_size_b + out_size_b - brb_size_b - 1 - b);
                    auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - 1 - f : in_size_f + out_size_f - brb_size_f - 1 - f);
                    auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - 1 - y : in_size_y + out_size_y - brb_size_y - 1 - y);
                    auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - 1 - x : in_size_x + out_size_x - brb_size_x - 1 - x);

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX


                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_mirror_101) {
    constexpr auto in_size_b = 3;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_y = 6;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror_101)
    );

    std::vector<float> input_data = generate_rnd_real_input<float>(in_size_b, in_size_f, in_size_y, in_size_x, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - b : in_size_b + out_size_b - brb_size_b - 2 - b);
                    auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - f : in_size_f + out_size_f - brb_size_f - 2 - f);
                    auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - y : in_size_y + out_size_y - brb_size_y - 2 - y);
                    auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - x : in_size_x + out_size_x - brb_size_x - 2 - x);

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX


                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}
