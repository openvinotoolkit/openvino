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
#include "api/CPP/eltwise.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include <api/CPP/reorder.hpp>
#include "test_utils/test_utils.h"

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

template <typename T>
T eltwise_execute(cldnn::eltwise_mode mode, T x, T y) {
    switch (mode) {
    case eltwise_mode::sum:
        return x + y;
    case eltwise_mode::sub:
        return x - y;
    case eltwise_mode::max:
        return std::max(x, y);
    case eltwise_mode::prod:
        return x * y;
    case eltwise_mode::div:
        return x / y;
    case eltwise_mode::min:
        return std::min(x, y);
    case eltwise_mode::pow:
        return std::pow((float)x, (float)y);
    case eltwise_mode::mod:
        return std::fmod((float)x, (float)y);
    default:
        return (T)0;
    }
}

template <typename T>
VVVVF<T> eltwise_reference(VVVVF<T> &input1, VVVVF<T> &input2,
    cldnn::eltwise_mode mode, bool relu = false, T slope = 0.0f,
    int input_padding_y = 0, int input_padding_x = 0,
    int output_padding_y = 0, int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t output_b = input1.size();
    size_t output_f = input1[0].size();
    size_t output_y = input1[0][0].size() + 2 * padding_y;
    size_t output_x = input1[0][0][0].size() + 2 * padding_x;
    VVVVF<T> output(output_b, VVVF<T>(output_f, VVF<T>(output_y, VF<T>(output_x))));

    T res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t f = 0; f < output_f; ++f) {
            for (size_t y = 0; y < input1[0][0].size(); ++y) {
                for (size_t x = 0; x < input1[0][0][0].size(); ++x) {
                    res = eltwise_execute<T>(mode, input1[b][f][y][x], input2[b][f][y][x]);
                    if (relu && res < (T)0)
                        res *= slope;
                    output[b][f][y + padding_y][x + padding_x] = res;
                }
            }
        }
    }
    return output;
}

template <typename T>
void generic_eltwise_test(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, cldnn::eltwise_mode mode,
    bool relu, T slope,    int input_padding_y, int input_padding_x, int output_padding_y, int output_padding_x) {

    int min_random = -2, max_random = 2;
    VVVVF<T> input1_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VVVVF<T> input2_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input1_rnd_vec = flatten_4d<T>(test_input_fmt, input1_rnd);
    VF<T> input2_rnd_vec = flatten_4d<T>(test_input_fmt, input2_rnd);

    engine engine;
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input1 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto input2 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", input1.get_layout().with_padding({{ 0, 0, input_padding_x, input_padding_y }, 0 })));
    topology.add(eltwise("eltwise", {"reorder1", "input2"}, mode, relu, slope, { { 0, 0, output_padding_x, output_padding_y }, 0 }));

    network network(engine, topology);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output_memory = outputs.at("eltwise").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    VVVVF<T> output_cpu = eltwise_reference<T>(input1_rnd, input2_rnd, mode, relu, slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    EXPECT_EQ(output_layout.format.value, test_input_fmt.value);
    tensor output_tensor = output_layout.get_buffer_size();
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(y_size, (int)output_cpu[0][0].size());
    EXPECT_EQ(x_size, (int)output_cpu[0][0][0].size());
    EXPECT_EQ(f_size, (int)output_cpu[0].size());
    EXPECT_EQ(b_size, (int)output_cpu.size());

    bool test_is_correct = true;
    VF<T> output_cpu_vec = flatten_4d<T>(test_input_fmt, output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (!floating_point_equal(output_cpu_vec[i], output_ptr[i]) && !(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i]))) {
            test_is_correct = false;
            break;
        }
    }
    EXPECT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "eltwise_mode = " << (int)mode << std::endl
        << "relu = " << relu << std::endl
        << "slope = " << (float)slope << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl
        << "type = " << (sizeof(T) == 2 ? "float16" : "float32") << std::endl;
}

TEST(eltwise_gpu_f32, add_basic_in4x4x2x2) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1.5  5.2     
    //  f1: b0:  7    8  b1:   12   8       
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7 
    //  f0: b0: 15  -2   b1: 17   6.5
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Output:
    //  f0: b0:   1.5  7    b1:  2.5   7      
    //  f0: b0:   18   2    b1:  17.5  6     
    //  f1: b0:   5.5  8    b1:   4    9.2     
    //  f1: b0:   15  16.5  b1:  22    16.5     
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,   5.5f,    4.f,
                          7.f,   7.f,    8.f,   9.2f,
                          18.f,17.5f,   15.f,   22.f,
                          2.f,   6.f,   7.5f,  5.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, max_basic_in4x4x4x4) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1.5  5.2     
    //  f1: b0:  7    8  b1:   12   8       
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7 
    //  f0: b0: 15   6   b1: 17   8
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Output:
    //  f0: b0:    1   5    b1:  2.5   7       
    //  f0: b0:   15   6    b1:  17    8    
    //  f1: b0:    5   6    b1:  2.5   5.2     
    //  f1: b0:    8   8    b1:  12    8     
    //
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::max));

    set_values(input, {
        1.f,   0.f,  5.f,  1.5f,
        2.f,   0.f,  6.f,  5.2f,
        3.f,   0.5f, 7.f, 12.f,
        4.f,  -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,  0.5f,  2.5f, 
         5.f,   7.f,   2.f,   4.f,
        15.f,  17.f,   8.f,  10.f,
         6.f,   8.f, -0.5f, -2.5f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
         1.f,   2.5f,  5.f,   2.5f,
         5.f,   7.f,   6.f,   5.2f,
        15.f,  17.f,   8.f,  12.f,
         6.f,   8.f,   8.f,   8.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, sub_basic_in4x4x4x4) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1.5  5.2     
    //  f1: b0:  7    8  b1:   12   8       
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7 
    //  f0: b0: 15   6   b1: 17   8
    //  f1: b0: 0.5  2   b1: -1   2
    //  f1: b0: 8   -0.5 b1: 8.5  10.5
    //
    //  Output:
    //  f0: b0:   0.5  -3    b1:  -2.5  -7       
    //  f0: b0:   -12  -2    b1:  -16.5 -8.5    
    //  f1: b0:   4.5   4    b1:  2.5    3.2     
    //  f1: b0:   -1    8.5  b1:  3.5   -2.5     
    //

    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sub));

    set_values(input, {
        1.f,   0.f,  5.f,  1.5f,
        2.f,   0.f,  6.f,  5.2f,
        3.f,   0.5f, 7.f,  12.f,
        4.f,  -0.5f, 8.f,   8.f
    });

    set_values(input2, {
        0.5f,  2.5f,  0.5f, -1.f,
        5.f,   7.f,   2.f,   2.f,
       15.f,  17.f,   8.f,   8.5f,
        6.f,   8.f, -0.5f,  10.5f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
         0.5f,  -2.5f,   4.5f,   2.5f,
        -3.f,   -7.f,    4.f,    3.2f,
       -12.f,  -16.5f,  -1.f,    3.5f,
        -2.f,   -8.5f,   8.5f,  -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_int, basic_in4x4x4x4) {
    //  Same params as in eltwise_gpu_f32, sub_basic_in4x4x4x4 but using int types instead

    std::vector<data_types> data_types_to_test = { data_types::i8, data_types::i32, data_types::i64 };
    std::vector<eltwise_mode> eltwise_ops_to_test = { eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::div, eltwise_mode::prod };

    for (auto& data_type : data_types_to_test)
    {
        for (auto& mode : eltwise_ops_to_test)
        {
            engine engine;
            auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
            auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

            topology topology;
            topology.add(input_layout("input", input.get_layout()));
            topology.add(input_layout("input2", input2.get_layout()));
            topology.add(reorder("input_reorder", "input", { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(reorder("input2_reorder", "input2", { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(eltwise("eltwise", { "input_reorder", "input2_reorder" }, mode));
            topology.add(reorder("eltwise_reorder", "eltwise", { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } }));

            std::vector<float> input_1_vec = {
                1.f,   0.f,  5.f,  1.f,
                2.f,   0.f,  6.f,  5.f,
                3.f,   0.f, 7.f,  12.f,
                4.f,   0.f, 8.f,   8.f
            };
            set_values(input, input_1_vec);

            std::vector<float> input_2_vec = {
                0.f,  2.f,  0.f, -1.f,
                5.f,   7.f,   2.f,   2.f,
                15.f,  17.f,   8.f,   8.f,
                6.f,   8.f, 0.f,  10.f };
            set_values(input2, input_2_vec);

            network network(engine, topology);
            network.set_input_data("input", input);
            network.set_input_data("input2", input2);
            auto outputs = network.execute();

            ASSERT_EQ(outputs.size(), size_t(1));
            EXPECT_EQ(outputs.begin()->first, "eltwise_reorder");

            auto output = outputs.at("eltwise_reorder").get_memory();

            auto output_ptr = output.pointer<float>();

            for (int i = 0; i < 16; i++)
            {
                float expected = 0.f;
                if (mode == eltwise_mode::sum)
                    expected = input_1_vec[i] + input_2_vec[i];
                else if (mode == eltwise_mode::sub)
                    expected = input_1_vec[i] - input_2_vec[i];
                else if (mode == eltwise_mode::prod)
                    expected = input_1_vec[i] * input_2_vec[i];
                else if (mode == eltwise_mode::div)
                    expected = input_1_vec[i] / input_2_vec[i];
                else if (mode == eltwise_mode::min)
                    expected = std::min(input_1_vec[i], input_2_vec[i]);
                else if (mode == eltwise_mode::max)
                    expected = std::max(input_1_vec[i], input_2_vec[i]);

                EXPECT_TRUE(are_equal(std::floor(expected), output_ptr[i]));
            }
        }
    }
}

TEST(eltwise_gpu_f32, prod_basic_in4x4x4x4) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1    5.2     
    //  f1: b0:  7    8  b1:   12   7.5       
    //
    //  Input2
    //  f0: b0: 0.5  0.5   b1: 5  2 
    //  f0: b0: 2.5  2.5   b1: 7  4
    //  f1: b0: 15   8     b1: 6  -0.5
    //  f1: b0: 17   10    b1: 8  -2.5
    //
    //  Output:
    //  f0: b0:   0.5  1     b1:  0      0       
    //  f0: b0:   7.5  10    b1:  3.5   -2     
    //  f1: b0:   75   48    b1:  6     -2.6     
    //  f1: b0:   119  80    b1:  96   -18.75     
    //


    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::prod));
    
    set_values(input, {
        1.f,   0.f,  5.f,  1.f,
        2.f,   0.f,  6.f,  5.2f,
        3.f,   0.5f, 7.f, 12.f,
        4.f,  -0.5f, 8.f,  7.5f
    });

    set_values(input2, {
        0.5f,   5.f,  15.f,    6.f,
        0.5f,   2.f,   8.f,   -0.5f,
        2.5f,   7.f,  17.f,    8.f,
        2.5f,   4.f,  10.f,   -2.5f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        0.5f,   0.0f,    75.f,    6.0f,
        1.0f,   0.0f,    48.f,   -2.6f,
        7.5f,   3.5f,   119.f,   96.0f,
       10.0f,  -2.0f,    80.f, -18.75f };
    
    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, max_basic_in4x4x4x4_input_padding) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2
    //  Input Padding: 2x1 (with reorder)

    //  Input:
    //  f0: b0:  1    2  b1:   0    0       
    //  f0: b0:  3    4  b1:   0.5 -0.5     
    //  f1: b0:  5    6  b1:   1.5  5.2     
    //  f1: b0:  7    8  b1:   12   8       
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7 
    //  f0: b0: 15   6   b1: 17   8
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Output:
    //  f0: b0:    1   5    b1:  2.5   7       
    //  f0: b0:   15   6    b1:  17    8    
    //  f1: b0:    5   6    b1:  2.5   5.2     
    //  f1: b0:    8   8    b1:  12    8     
    //
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder2", "input2", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
    topology.add(eltwise("eltwise", {"reorder", "reorder2"}, eltwise_mode::max));

    set_values(input, {
        1.f,   0.f,  5.f,  1.5f,
        2.f,   0.f,  6.f,  5.2f,
        3.f,   0.5f, 7.f, 12.f,
        4.f,  -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,  0.5f,  2.5f,
        5.f,   7.f,   2.f,   4.f,
        15.f,  17.f,   8.f,  10.f,
        6.f,   8.f, -0.5f, -2.5f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.f,   2.5f,  5.f,   2.5f,
        5.f,   7.f,   6.f,   5.2f,
        15.f,  17.f,   8.f,  12.f,
        6.f,   8.f,   8.f,   8.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_in4x4x2x2_with_coefficients) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7
    //  f0: b0: 15  -2   b1: 17   6.5
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Output:
    //  f0: b0:   0.75  3.5  b1:  1.25  3.5
    //  f0: b0:   9     1    b1:  8.75  3
    //  f1: b0:   2.75  4    b1:   2    4.6
    //  f1: b0:   7.5  8.25  b1:  11    8.25
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum, {0.5f, 0.5f}));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 0.75f, 1.25f, 2.75f,  2.f,
                          3.5f,  3.5f,   4.f, 4.6f,
                          9.f, 8.75f,  7.5f, 11.f,
                          1.f,   3.f, 3.75f, 2.75f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, coefficients_count_check) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));

    std::vector<float> coeffs0 = {};
    std::vector<float> coeffs1 = {0.5f};
    std::vector<float> coeffs2 = {0.5f, 0.5f};
    std::vector<float> coeffs3 = {0.5f, 0.5f, 0.5f};

    EXPECT_THROW(topology.add(eltwise("eltwise1", {"input", "input2"}, eltwise_mode::sum, coeffs1)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise2", {"input", "input2"}, eltwise_mode::sum, coeffs3)), std::invalid_argument);

    EXPECT_THROW(topology.add(eltwise("eltwise3", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs1)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise4", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs2)), std::invalid_argument);

    EXPECT_NO_THROW(topology.add(eltwise("eltwise5", {"input", "input2"}, eltwise_mode::sum, coeffs0)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise6", {"input", "input2"}, eltwise_mode::sum, coeffs2)));

    EXPECT_NO_THROW(topology.add(eltwise("eltwise7", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs0)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise8", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs3)));
}

TEST(eltwise_gpu_f32, add_basic_in4x4x2x2_with_coefficients_3inputs) {
    //  Input3   : 2x2x2
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7
    //  f0: b0: 15  -2   b1: 17   6.5
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Input3
    //  f0: b0: 8  7 b1: 0 1
    //  f0: b0: 6  5 b1: 0 1
    //  f1: b0: 4  3 b1: 0 1
    //  f1: b0: 2  1 b1: 0 1
    //
    //  Output:
    //  f0: b0:   4.75  7      b1:  1.25  4
    //  f0: b0:   12    3.5    b1:  8.75  3.5
    //  f1: b0:   4.75  5.5    b1:   2    5.1
    //  f1: b0:   8.5   8.75   b1:  11    8.75
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::sum, {0.5f, 0.5f, 0.5f}));

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

    set_values(input3, {
            8.f,  0.f,  4.f, 0.f,
            7.f,  1.f,  3.f, 1.f,
            6.f,  0.f,  2.f, 0.f,
            5.f,  1.f,  1.f, 1.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.75f, 1.25f, 4.75f,  2.f,
                          7.0f,  4.0f,  5.5f, 5.1f,
                          12.f, 8.75f,  8.5f, 11.f,
                          3.5f,  3.5f, 4.25f, 3.25f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, max_3inputs_in4x4x4x4_input_padding) {
    //  Input  : 2x2x2x2
    //  Input2 : 2x2x2x2
    //  Input3 : 2x2x2x2
    //  Output : 2x2x2x2
    //  Input Padding: 2x1 (with reorder)

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7
    //  f0: b0: 15   6   b1: 17   8
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Input3
    //  f0: b0: 1.1  1   b1: 4  0
    //  f0: b0: 15  -1   b1: 3  6
    //  f1: b0: 1.5  2   b1: 2  7
    //  f1: b0: 9   0.5  b1: 1  8
    //
    //  Output:
    //  f0: b0:   1.1  5    b1:  4   7
    //  f0: b0:   15   6    b1:  17    8
    //  f1: b0:    5   6    b1:  2.5   7
    //  f1: b0:    9   8    b1:  12    8
    //
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder2", "input2", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder3", "input3", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
    topology.add(eltwise("eltwise", {"reorder", "reorder2", "reorder3"}, eltwise_mode::max));

    set_values(input, {
         1.f,  0.f,  5.f,  1.5f,
         2.f,  0.f,  6.f,  5.2f,
         3.f,  0.5f, 7.f,  12.f,
         4.f, -0.5f, 8.f,   8.f
    });

    set_values(input2, {
        0.5f, 2.5f,  0.5f,  2.5f,
         5.f,  7.f,   2.f,   4.f,
        15.f, 17.f,   8.f,  10.f,
         6.f,  8.f, -0.5f, -2.5f });

    set_values(input3, {
        1.1f,  4.f, 1.5f, 2.f,
         1.f,  0.f,  2.f, 7.f,
        15.f,  3.f,  9.f, 1.f,
        -1.f,  6.f, 0.5f, 8.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.1f,   4.f,  5.f,   2.5f,
        5.f,   7.f,   6.f,   7.f,
        15.f,  17.f,   9.f,  12.f,
        6.f,   8.f,   8.f,   8.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}


void run_eltwise_generic_test(cldnn::eltwise_mode mode)
{
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    std::pair<int, int> input_size = { 227, 227 };

    engine engine;
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    generic_eltwise_test<float>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, false, 0.f, 0, 0, 0, 0);
    if (f16_supported)
        generic_eltwise_test<FLOAT16>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, false, (FLOAT16)0.f, 0, 0, 0, 0);
}

TEST(eltwise_gpu, eltwise_div) {
    run_eltwise_generic_test(cldnn::eltwise_mode::div);
}

TEST(eltwise_gpu, eltwise_min) {
    run_eltwise_generic_test(cldnn::eltwise_mode::min);
}

TEST(eltwise_gpu, eltwise_pow) {
    run_eltwise_generic_test(cldnn::eltwise_mode::pow);
}

TEST(eltwise_gpu, eltwise_mod) {
    run_eltwise_generic_test(cldnn::eltwise_mode::mod);
}

TEST(DISABLED_eltwise_gpu, generic_random) {
    VF<cldnn::format> test_inputs_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<cldnn::eltwise_mode> modes = { cldnn::eltwise_mode::sum, cldnn::eltwise_mode::sub, cldnn::eltwise_mode::max, cldnn::eltwise_mode::prod };
    VF<bool> relu = { true, false };
    VF<float> slopes = { 0.0f, -0.0f, -17.19f, 1028.8f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
    std::vector<std::pair<int, int>> input_sizes = { { 100, 100 },{ 227, 227 },{ 400, 600 } };

    engine engine;
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (cldnn::format test_input_fmt : test_inputs_fmts) {
        for (int input_b = 1; input_b <= 16; input_b *= 2) {
            for (int input_f = 1; input_f <= 1; ++input_f) {
                for (std::pair<int, int> &input_yx : input_sizes) {
                    for (cldnn::eltwise_mode mode : modes) {
                        for (bool relu_activated : relu) {
                            for (float slope : slopes) {
                                for (int input_padding_y = 0; input_padding_y <= 0; ++input_padding_y) {
                                    for (int input_padding_x = 0; input_padding_x <= 0; ++input_padding_x) {
                                        for (int output_padding_y = 0; output_padding_y <= 1; ++output_padding_y) {
                                            for (int output_padding_x = 0; output_padding_x <= 1; ++output_padding_x) {
                                                generic_eltwise_test<float>(test_input_fmt, input_b, input_f, input_yx.first, input_yx.second, mode, relu_activated, slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
                                                if (!f16_supported) continue;
                                                generic_eltwise_test<FLOAT16>(test_input_fmt, input_b, input_f, input_yx.first, input_yx.second, mode, relu_activated, (FLOAT16)slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
