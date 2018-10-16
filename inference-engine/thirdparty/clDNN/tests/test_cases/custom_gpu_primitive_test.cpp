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
#include "api/CPP/custom_gpu_primitive.hpp"
#include <api/CPP/engine.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"


namespace cldnn
{
	template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

TEST(custom_gpu_primitive_f32, add_basic_in2x2x2x2) {
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

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, const __global float* input1, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + input1[idx];
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<cldnn_arg> parameters = { {arg_input, 0}, {arg_input, 1 }, {arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel", 
        { "input", "input2" },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable",
        output_layout,
        gws));

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
    EXPECT_EQ(outputs.begin()->first, "user_kernel");

    auto output = outputs.at("user_kernel").get_memory();

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

TEST(custom_gpu_primitive_f32, eltwise_add_basic_in2x2x2x2) {
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

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + 1;
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<cldnn_arg> parameters = { { arg_input, 0 },{ arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { "eltwise" },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable",
        output_layout,
        gws));

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
    EXPECT_EQ(outputs.begin()->first, "user_kernel");

    auto output = outputs.at("user_kernel").get_memory();

    float answers[16] = 
      { 2.5f,  3.5f,  6.5f,   5.f,
         8.f,   8.f,   9.f, 10.2f,
        19.f, 18.5f,  16.f,  23.f,
         3.f,   7.f,  8.5f,  6.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(custom_gpu_primitive_f32, add_eltwise_basic_in2x2x2x2) {
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

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + SCALAR;
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<cldnn_arg> parameters = { { arg_input, 0 },{ arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { "input" },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=1",
        output_layout,
        gws));
    topology.add(eltwise("eltwise", {"user_kernel", "input2"}, eltwise_mode::sum));

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

    float answers[16] =
    { 2.5f,  3.5f,  6.5f,   5.f,
        8.f,   8.f,   9.f, 10.2f,
        19.f, 18.5f,  16.f,  23.f,
        3.f,   7.f,  8.5f,  6.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(custom_gpu_primitive_f32, two_kernels_with_same_entry_point_basic_in2x2x2x2) {
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

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code1 =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + SCALAR;
            }
        )__krnl";
    
    std::string kernel_code2 =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + 2*SCALAR;
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<cldnn_arg> parameters = { { arg_input, 0 },{ arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel1",
        { "input" },
        { kernel_code1 },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=1",
        output_layout,
        gws));
    topology.add(custom_gpu_primitive(
        "user_kernel2",
        { "user_kernel1" },
        { kernel_code2 },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=3",
        output_layout,
        gws));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "user_kernel2");

    auto output = outputs.at("user_kernel2").get_memory();

    auto output_ptr = output.pointer<float>();
    auto input_ptr = input.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(input_ptr[i] + 7, output_ptr[i]));
    }
}
