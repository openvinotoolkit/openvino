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
#include <api/CPP/broadcast.hpp>
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


TEST(broadcast_gpu, basic_yxfb_1x1x2x3_to_1x2x2x9) {
    //  Input (BF:XY) :  1x1:3x2
    //  Output (BF:XY):  1x2:9x2

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 3;

    constexpr auto bc_scale_b = 1;
    constexpr auto bc_scale_f = 2;
    constexpr auto bc_scale_y = 1;
    constexpr auto bc_scale_x = 3;

    constexpr auto out_size_b = bc_scale_b * in_size_b;
    constexpr auto out_size_f = bc_scale_f * in_size_f;
    constexpr auto out_size_y = bc_scale_y * in_size_y;
    constexpr auto out_size_x = bc_scale_x * in_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        broadcast("output", "input", {out_size_b, out_size_f, out_size_x, out_size_y})
    );

    std::vector<float> input_data = {
         41, -11, 13,
        107, -66,  0,
    };
    std::vector<float> out_data = {
         41,  41,   -11, -11,   13, 13,    41,  41,   -11, -11,   13, 13,    41,  41,   -11, -11,   13, 13,
        107, 107,   -66, -66,    0,  0,   107, 107,   -66, -66,    0,  0,   107, 107,   -66, -66,    0,  0,
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

TEST(broadcast_gpu, basic_bfyx_4x2x2x2_to_8x2x6x4) {
    //  Input (BF:XY) :  4x2:2x2
    //  Output (BF:XY):  8x2:6x4

    constexpr auto in_size_b = 4;
    constexpr auto in_size_f = 2;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 2;

    constexpr auto bc_scale_b = 2;
    constexpr auto bc_scale_f = 1;
    constexpr auto bc_scale_y = 3;
    constexpr auto bc_scale_x = 2;

    constexpr auto out_size_b = bc_scale_b * in_size_b;
    constexpr auto out_size_f = bc_scale_f * in_size_f;
    constexpr auto out_size_y = bc_scale_y * in_size_y;
    constexpr auto out_size_x = bc_scale_x * in_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        broadcast("output", "input", {out_size_b, out_size_f, out_size_x, out_size_y})
    );

    std::vector<float> input_data = {
         11,  12,
         21,  22,

        -11, -12,
        -21, -22,


         13,  14,
         23,  24,

        -13, -14,
        -23, -24,


         15,  16,
         25,  26,

        -15, -16,
        -25, -26,


         17,  18,
         27,  28,

        -17, -18,
        -27, -28,
    };
    std::vector<float> out_data = {
         11,  12,  11,  12,
         21,  22,  21,  22,
         11,  12,  11,  12,
         21,  22,  21,  22,
         11,  12,  11,  12,
         21,  22,  21,  22,

        -11, -12, -11, -12,
        -21, -22, -21, -22,
        -11, -12, -11, -12,
        -21, -22, -21, -22,
        -11, -12, -11, -12,
        -21, -22, -21, -22,


         13,  14,  13,  14,
         23,  24,  23,  24,
         13,  14,  13,  14,
         23,  24,  23,  24,
         13,  14,  13,  14,
         23,  24,  23,  24,

        -13, -14, -13, -14,
        -23, -24, -23, -24,
        -13, -14, -13, -14,
        -23, -24, -23, -24,
        -13, -14, -13, -14,
        -23, -24, -23, -24,


         15,  16,  15,  16,
         25,  26,  25,  26,
         15,  16,  15,  16,
         25,  26,  25,  26,
         15,  16,  15,  16,
         25,  26,  25,  26,

        -15, -16, -15, -16,
        -25, -26, -25, -26,
        -15, -16, -15, -16,
        -25, -26, -25, -26,
        -15, -16, -15, -16,
        -25, -26, -25, -26,


         17,  18,  17,  18,
         27,  28,  27,  28,
         17,  18,  17,  18,
         27,  28,  27,  28,
         17,  18,  17,  18,
         27,  28,  27,  28,

        -17, -18, -17, -18,
        -27, -28, -27, -28,
        -17, -18, -17, -18,
        -27, -28, -27, -28,
        -17, -18, -17, -18,
        -27, -28, -27, -28,


         11,  12,  11,  12,
         21,  22,  21,  22,
         11,  12,  11,  12,
         21,  22,  21,  22,
         11,  12,  11,  12,
         21,  22,  21,  22,

        -11, -12, -11, -12,
        -21, -22, -21, -22,
        -11, -12, -11, -12,
        -21, -22, -21, -22,
        -11, -12, -11, -12,
        -21, -22, -21, -22,


         13,  14,  13,  14,
         23,  24,  23,  24,
         13,  14,  13,  14,
         23,  24,  23,  24,
         13,  14,  13,  14,
         23,  24,  23,  24,

        -13, -14, -13, -14,
        -23, -24, -23, -24,
        -13, -14, -13, -14,
        -23, -24, -23, -24,
        -13, -14, -13, -14,
        -23, -24, -23, -24,


         15,  16,  15,  16,
         25,  26,  25,  26,
         15,  16,  15,  16,
         25,  26,  25,  26,
         15,  16,  15,  16,
         25,  26,  25,  26,

        -15, -16, -15, -16,
        -25, -26, -25, -26,
        -15, -16, -15, -16,
        -25, -26, -25, -26,
        -15, -16, -15, -16,
        -25, -26, -25, -26,


         17,  18,  17,  18,
         27,  28,  27,  28,
         17,  18,  17,  18,
         27,  28,  27,  28,
         17,  18,  17,  18,
         27,  28,  27,  28,

        -17, -18, -17, -18,
        -27, -28, -27, -28,
        -17, -18, -17, -18,
        -27, -28, -27, -28,
        -17, -18, -17, -18,
        -27, -28, -27, -28,
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
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(broadcast_gpu, basic_byxf_2x3x4x5_to_10x12x12x10) {
    //  Input (BF:XY) :    2x3:5x4
    //  Output (BF:XY):  10x12:10x12

    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 3;
    constexpr auto in_size_y = 4;
    constexpr auto in_size_x = 5;

    constexpr auto bc_scale_b = 5;
    constexpr auto bc_scale_f = 4;
    constexpr auto bc_scale_y = 3;
    constexpr auto bc_scale_x = 2;

    constexpr auto out_size_b = bc_scale_b * in_size_b;
    constexpr auto out_size_f = bc_scale_f * in_size_f;
    constexpr auto out_size_y = bc_scale_y * in_size_y;
    constexpr auto out_size_x = bc_scale_x * in_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::byxf, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        broadcast("output", "input", {out_size_b, out_size_f, out_size_x, out_size_y})
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
                    auto output_off = ((b * out_size_y + y) * out_size_x + x) * out_size_f + f; // BYXF

                    auto in_b = b % in_size_b;
                    auto in_f = f % in_size_f;
                    auto in_y = y % in_size_y;
                    auto in_x = x % in_size_x;

                    auto input_off  = ((in_b * in_size_y + in_y) * in_size_x + in_x) * in_size_f + in_f; // BYXF


                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(broadcast_gpu, basic_bfyx_2x1x1x5_to_2x13x11x5) {
    //  Input (BF:XY) :   2x1:5x1
    //  Output (BF:XY):  2x13:5x11

    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 1;
    constexpr auto in_size_x = 5;

    constexpr auto bc_scale_b = 1;
    constexpr auto bc_scale_f = 13;
    constexpr auto bc_scale_y = 11;
    constexpr auto bc_scale_x = 1;

    constexpr auto out_size_b = bc_scale_b * in_size_b;
    constexpr auto out_size_f = bc_scale_f * in_size_f;
    constexpr auto out_size_y = bc_scale_y * in_size_y;
    constexpr auto out_size_x = bc_scale_x * in_size_x;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        broadcast("output", "input", {out_size_b, out_size_f, out_size_x, out_size_y})
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

                    auto in_b = b % in_size_b;
                    auto in_f = f % in_size_f;
                    auto in_y = y % in_size_y;
                    auto in_x = x % in_size_x;

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX


                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(broadcast_gpu, basic_error_on_nondiv_bc_size) {
    //  Input (BF:XY) :   2x1:5x1
    //  Output (BF:XY):  2x13:5x11

    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 1;
    constexpr auto in_size_x = 5;

    constexpr auto out_size_b = in_size_b;
    constexpr auto out_size_f = in_size_f;
    constexpr auto out_size_y = in_size_y;
    constexpr auto out_size_x = 7;

    engine engine;
    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        broadcast("output", "input", {out_size_b, out_size_f, out_size_x, out_size_y})
    );

    std::vector<float> input_data = generate_rnd_real_input<float>(in_size_b, in_size_f, in_size_y, in_size_x, -8.0f, 8.0f);
    set_values(input, input_data);

    EXPECT_ANY_THROW(network(engine, topology));
}

