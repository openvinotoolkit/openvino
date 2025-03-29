// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/custom_gpu_primitive.hpp>

using namespace cldnn;
using namespace ::tests;

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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, const __global float* input1, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + input1[idx];
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        {custom_gpu_primitive::arg_input, 0},
        {custom_gpu_primitive::arg_input, 1 },
        {custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("input"), input_info("input2") },
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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "user_kernel");

    auto output = outputs.at("user_kernel").get_memory();

    float answers[16] = { 1.5f, 2.5f,   5.5f,    4.f,
                          7.f,   7.f,    8.f,   9.2f,
                          18.f,17.5f,   15.f,   22.f,
                          2.f,   6.f,   7.5f,  5.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

template<data_types DType>
void add_basic_in2x2x2x2_with_reorder()
{
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string data_type_string = "float";
    switch (DType)
    {
    case data_types::i32:
        data_type_string = "int";
        break;
    case data_types::i64:
        data_type_string = "long";
        break;
    default:
        throw std::runtime_error("Test does not support this data format!");
    }

    std::string kernel_code =
        "__kernel void add_kernel(const __global " + data_type_string + "* input0, const __global " + data_type_string + "* input1, __global " + data_type_string + "* output)\n" +
        "   {\n" +
        "       const unsigned idx = get_global_id(0);\n" +
        "       output[idx] = input0[idx] + input1[idx];\n" +
        "   }\n";
    std::string entry_point = "add_kernel";
    std::vector<custom_gpu_primitive::arg_desc> parameters = { { custom_gpu_primitive::arg_input, 0 },{ custom_gpu_primitive::arg_input, 1 },{ custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { DType, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("to_int1", input_info("input"), { DType, format::yxfb,{ 2,2,2,2 } }));
    topology.add(reorder("to_int2", input_info("input2"), { DType, format::yxfb,{ 2,2,2,2 } }));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("to_int1"), input_info("to_int2") },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable",
        output_layout,
        gws));
    topology.add(reorder("to_float", input_info("user_kernel"), { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } }));

    set_values(input, {
        1.f,   0.f, 5.f, 1.f,
        2.f,   0.f, 6.f, 5.f,
        3.f,  0.f, 7.f, 12.f,
        4.f,  0.f, 8.f,  8.f
    });

    set_values(input2, {
        0.f,   2.f,  0.f,  2.f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.f,  0.f, -2.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "to_float");

    auto output = outputs.at("to_float").get_memory();

    float answers[16] = { 1.f, 2.f,   5.f,    3.f,
        7.f,   7.f,    8.f,   9.f,
        18.f,17.f,   15.f,   22.f,
        2.f,   6.f,   8.f,  6.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(custom_gpu_primitive_int32, add_basic_in2x2x2x2) {
    add_basic_in2x2x2x2_with_reorder<data_types::i32>();
}

TEST(custom_gpu_primitive_int64, add_basic_in2x2x2x2) {
    add_basic_in2x2x2x2_with_reorder<data_types::i64>();
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + 1;
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<custom_gpu_primitive::arg_desc> parameters = { { custom_gpu_primitive::arg_input, 0 },{ custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("eltwise") },
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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "user_kernel");

    auto output = outputs.at("user_kernel").get_memory();

    float answers[16] =
      { 2.5f,  3.5f,  6.5f,   5.f,
         8.f,   8.f,   9.f, 10.2f,
        19.f, 18.5f,  16.f,  23.f,
         3.f,   7.f,  8.5f,  6.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + SCALAR;
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<custom_gpu_primitive::arg_desc> parameters = { { custom_gpu_primitive::arg_input, 0 },{ custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("input") },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=1",
        output_layout,
        gws));
    topology.add(eltwise("eltwise", { input_info("user_kernel"), input_info("input2") }, eltwise_mode::sum));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] =
    { 2.5f,  3.5f,  6.5f,   5.f,
        8.f,   8.f,   9.f, 10.2f,
        19.f, 18.5f,  16.f,  23.f,
        3.f,   7.f,  8.5f,  6.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

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
    std::vector<custom_gpu_primitive::arg_desc> parameters = { { custom_gpu_primitive::arg_input, 0 },{ custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel1",
        { input_info("input") },
        { kernel_code1 },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=1",
        output_layout,
        gws));
    topology.add(custom_gpu_primitive(
        "user_kernel2",
        { input_info("user_kernel1") },
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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "user_kernel2");

    auto output = outputs.at("user_kernel2").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(input_ptr[i] + 7, output_ptr[i]));
    }
}

template <typename T>
void test_custom_gpu_primitive_u8_add_basic_in2x2x2x2(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::u8, format::yxfb,{ 2, 2, 2, 2 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global uchar* input0, const __global uchar* input1, __global uchar* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + input1[idx];
            }
        )__krnl";
    std::string entry_point = "add_kernel";
    std::vector<custom_gpu_primitive::arg_desc> parameters = { { custom_gpu_primitive::arg_input, 0 },{ custom_gpu_primitive::arg_input, 1 },{ custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::u8, format::yxfb,{ 2, 2, 2, 2 } };
    std::vector<size_t> gws = { output_layout.count() };
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("input"), input_info("input2") },
        { kernel_code },
        entry_point,
        parameters,
        "-cl-mad-enable",
        output_layout,
        gws));

    set_values<T>(input, {
          1,   0,   5,    1,
        200, 100, 160,  150,
        130,   0, 175,   12,
          4, 100,   8,  180
    });

    set_values<T>(input2, {
         0,  2,  0,  2,
        55, 75, 20,  4,
        15, 17, 80, 10,
         2, 60,  0, 20
    });

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "user_kernel");

    auto output = outputs.at("user_kernel").get_memory();

    T answers[16] = {
          1,   2,   5,   3,
        255, 175, 180, 154,
        145,  17, 255,  22,
          6, 160,   8, 200
    };

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(custom_gpu_primitive_u8, add_basic_in2x2x2x2) {
    test_custom_gpu_primitive_u8_add_basic_in2x2x2x2<unsigned char>(false);
}

TEST(export_import_custom_gpu_primitive_u8, add_basic_in2x2x2x2) {
    test_custom_gpu_primitive_u8_add_basic_in2x2x2x2<unsigned char>(true);
}
