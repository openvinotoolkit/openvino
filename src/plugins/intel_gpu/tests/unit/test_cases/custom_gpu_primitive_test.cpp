// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph/include/primitive_inst.h"
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/custom_gpu_primitive.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/activation.hpp>

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
        { output_layout },
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

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

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
        { output_layout },
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

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

// custom_gpu_primitive_impl enqueues an OpenCL kernel, so it must not report itself as a
// CPU impl: that places its own output, and every producer feeding it, in usm_host.
TEST(custom_gpu_primitive_f32, impl_is_not_cpu_and_producer_stays_in_device_memory) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 4 } });

    std::string kernel_code =
        R"__krnl(
            __kernel void copy_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 1, 1, 4, 4 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(eltwise("producer", { input_info("input"), input_info("input") }, eltwise_mode::sum));
    topology.add(custom_gpu_primitive(
        "user_kernel",
        { input_info("producer") },
        { kernel_code },
        "copy_kernel",
        parameters,
        "-cl-mad-enable",
        { output_layout },
        { output_layout.count() }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.execute();

    auto custom_inst = network.get_primitive("user_kernel");
    ASSERT_NE(custom_inst->get_impl(), nullptr);
    EXPECT_FALSE(custom_inst->get_impl()->is_cpu());
    EXPECT_FALSE(custom_inst->get_impl()->requires_lockable_input());

    if (engine.supports_allocation(allocation_type::usm_device)) {
        auto producer_inst = network.get_primitive("producer");
        ASSERT_NE(producer_inst->output_memory_ptr(), nullptr);
        EXPECT_NE(producer_inst->output_memory_ptr()->get_allocation_type(),
                  allocation_type::usm_host);
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
        { output_layout },
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

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

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
        { output_layout },
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

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

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
        { output_layout },
        gws));
    topology.add(custom_gpu_primitive(
        "user_kernel2",
        { input_info("user_kernel1") },
        { kernel_code2 },
        entry_point,
        parameters,
        "-cl-mad-enable -DSCALAR=3",
        { output_layout },
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

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
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
        { output_layout },
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


// INPUT0_OFFSET is documented as "the number of elements from the start of the tensor to
// the first valid element, bypassing the lower padding", with the padding and pitch arrays
// always ordered as BFYX. Logical b1 f1 y4 x3 with lower padding y=1 x=2 is padded to
// y5 x5, giving pitches {25,25,5,1} and an offset of 5*1 + 1*2 = 7. Indexing the padding
// array in the wrong axis order yields 11 instead.
TEST(custom_gpu_primitive_f32, input_offset_matches_spatial_lower_padding) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ data_types::f32, format::bfyx, tensor{ 1, 1, 3, 4 },
                             padding({0, 0, 1, 2}, {0, 0, 0, 0}) };
    auto input = engine.allocate_memory(in_layout);

    std::vector<float> vals(in_layout.get_linear_size());
    for (size_t i = 0; i < vals.size(); i++)
        vals[i] = static_cast<float>(i);
    set_values(input, vals);

    std::string kernel_code =
        R"__krnl(
            __kernel void probe_kernel(const __global float* input0, __global float* output)
            {
                output[0] = (float)(INPUT0_OFFSET);
                output[1] = input0[INPUT0_OFFSET];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 1, 1, 2, 1 } };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(custom_gpu_primitive("user_kernel", { input_info("input") }, { kernel_code },
        "probe_kernel", parameters, "-cl-mad-enable", { output_layout }, { 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("user_kernel").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> ptr(output, get_test_stream());
    ASSERT_EQ(ptr[0], 7.f) << "INPUT0_OFFSET does not match the lower padding";
    ASSERT_EQ(ptr[1], 7.f) << "a kernel honouring INPUT0_OFFSET read the wrong element";
}

// An in-place crop keeps the parent buffer and expresses the crop as padding on the view.
// A kernel that honours INPUT0_OFFSET must therefore still read the cropped data, and the
// crop must remain optimized out. Feature axis first.
TEST(custom_gpu_primitive_f32, conforming_kernel_reads_in_place_crop_on_feature_axis) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 2 } });
    set_values(input, { 0.f, 1.f, 10.f, 11.f, 20.f, 21.f, 30.f, 31.f });

    std::string kernel_code =
        R"__krnl(
            __kernel void copy_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[INPUT0_OFFSET + idx];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 1, 2, 1, 2 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    // features [2, 4) of the parent tensor
    topology.add(crop("crop", input_info("input"), tensor(1, 2, 1, 2), tensor(0, 2, 0, 0)));
    topology.add(custom_gpu_primitive("user_kernel", { input_info("crop") }, { kernel_code },
        "copy_kernel", parameters, "-cl-mad-enable", { output_layout }, { output_layout.count() }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_TRUE(network.get_primitive("crop")->can_be_optimized())
        << "a custom layer input should not block in-place cropping";

    auto output = outputs.at("user_kernel").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<float> expected = { 20.f, 21.f, 30.f, 31.f };
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(output_ptr[i], expected[i]) << "element " << i;
}

// The same on a spatial axis, where the offset covers a non-zero Y padding and so depends
// on the padding array being read in the documented BFYX order.
TEST(custom_gpu_primitive_f32, conforming_kernel_reads_in_place_crop_on_spatial_axis) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 1, 3, 4 } });
    std::vector<float> vals(12);
    for (size_t i = 0; i < vals.size(); i++)
        vals[i] = static_cast<float>(i);
    set_values(input, vals);

    std::string kernel_code =
        R"__krnl(
            __kernel void copy_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[INPUT0_OFFSET + idx];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, tensor{ 1, 1, 3, 3 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    // rows [1, 4) of the parent tensor
    topology.add(crop("crop", input_info("input"), tensor(1, 1, 3, 3), tensor(0, 0, 0, 1)));
    topology.add(custom_gpu_primitive("user_kernel", { input_info("crop") }, { kernel_code },
        "copy_kernel", parameters, "-cl-mad-enable", { output_layout }, { output_layout.count() }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_TRUE(network.get_primitive("crop")->can_be_optimized())
        << "a custom layer input should not block in-place cropping";

    auto output = outputs.at("user_kernel").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < 9; i++)
        ASSERT_EQ(output_ptr[i], static_cast<float>(i + 3)) << "element " << i;
}

// An in-place concatenation writes offset padding onto its predecessors output layouts,
// which a kernel would have to honour through OUTPUT0_OFFSET. custom_gpu_primitive is not
// listed in available_pred, so that does not happen today; this pins the current behaviour.
TEST(custom_gpu_primitive_f32, custom_layer_producer_blocks_in_place_concat) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 2 } });
    set_values(input, { 0.f, 1.f, 10.f, 11.f });

    std::string kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + ADDEND;
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 1, 2, 1, 2 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(custom_gpu_primitive("k1", { input_info("input") }, { kernel_code },
        "add_kernel", parameters, "-cl-mad-enable -DADDEND=100.0f", { output_layout }, { output_layout.count() }));
    topology.add(custom_gpu_primitive("k2", { input_info("input") }, { kernel_code },
        "add_kernel", parameters, "-cl-mad-enable -DADDEND=200.0f", { output_layout }, { output_layout.count() }));
    topology.add(concatenation("concat", { input_info("k1"), input_info("k2") }, 1));
    // the concat must not be the network output, or it is excluded from fusing for that
    // reason alone and the test would pass without exercising anything
    topology.add(activation("act", input_info("concat"), activation_func::abs));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_FALSE(network.get_primitive("concat")->can_be_optimized());

    auto output = outputs.at("act").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<float> expected = { 100.f, 101.f, 110.f, 111.f, 200.f, 201.f, 210.f, 211.f };
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(output_ptr[i], expected[i]) << "element " << i;
}

// Batch axis, completing the b/f/y/x coverage of the INPUT0_OFFSET terms.
TEST(custom_gpu_primitive_f32, conforming_kernel_reads_in_place_crop_on_batch_axis) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 1, 2, 1 } });
    set_values(input, { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f });

    std::string kernel_code =
        R"__krnl(
            __kernel void copy_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[INPUT0_OFFSET + idx];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 2, 1, 2, 1 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    // batches [2, 4) of the parent tensor
    topology.add(crop("crop", input_info("input"), tensor(2, 1, 2, 1), tensor(2, 0, 0, 0)));
    topology.add(custom_gpu_primitive("user_kernel", { input_info("crop") }, { kernel_code },
        "copy_kernel", parameters, "-cl-mad-enable", { output_layout }, { output_layout.count() }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_TRUE(network.get_primitive("crop")->can_be_optimized())
        << "a custom layer input should not block in-place cropping";

    auto output = outputs.at("user_kernel").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<float> expected = { 4.f, 5.f, 6.f, 7.f };
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(output_ptr[i], expected[i]) << "element " << i;
}

// A feature crop with batch > 1 leaves the cropped region NON-CONTIGUOUS: each batch slice
// starts at a different offset into the parent. Adding INPUT0_OFFSET to a flat index is not
// enough here, so this uses the full pitch-based addressing the CustomLayer documentation
// prescribes, on both the input and the output. This is the case that decides whether the
// emitted macros are actually sufficient to consume an in-place view.
TEST(custom_gpu_primitive_f32, documented_pitch_indexing_reads_non_contiguous_in_place_crop) {
    auto& engine = get_test_engine();

    // b2 f4 y1 x2, value == linear bfyx index
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 4, 2, 1 } });
    std::vector<float> vals(16);
    for (size_t i = 0; i < vals.size(); i++)
        vals[i] = static_cast<float>(i);
    set_values(input, vals);

    std::string kernel_code =
        R"__krnl(
            __kernel void copy_kernel(const __global float* input0, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                const unsigned x_size = OUTPUT0_DIMS[3];
                const unsigned y_size = OUTPUT0_DIMS[2];
                const unsigned f_size = OUTPUT0_DIMS[1];

                const unsigned x = idx % x_size;
                const unsigned y = (idx / x_size) % y_size;
                const unsigned f = (idx / (x_size * y_size)) % f_size;
                const unsigned b = idx / (x_size * y_size * f_size);

                const unsigned in_id = b * INPUT0_PITCHES[0] + f * INPUT0_PITCHES[1] +
                                       y * INPUT0_PITCHES[2] + x * INPUT0_PITCHES[3] +
                                       INPUT0_OFFSET;
                const unsigned out_id = b * OUTPUT0_PITCHES[0] + f * OUTPUT0_PITCHES[1] +
                                        y * OUTPUT0_PITCHES[2] + x * OUTPUT0_PITCHES[3] +
                                        OUTPUT0_OFFSET;
                output[out_id] = input0[in_id];
            }
        )__krnl";
    std::vector<custom_gpu_primitive::arg_desc> parameters = {
        { custom_gpu_primitive::arg_input, 0 },
        { custom_gpu_primitive::arg_output, 0 } };
    layout output_layout = { data_types::f32, format::bfyx, { 2, 2, 2, 1 } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    // features [2, 4) of both batches
    topology.add(crop("crop", input_info("input"), tensor(2, 2, 2, 1), tensor(0, 2, 0, 0)));
    topology.add(custom_gpu_primitive("user_kernel", { input_info("crop") }, { kernel_code },
        "copy_kernel", parameters, "-cl-mad-enable", { output_layout }, { output_layout.count() }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_TRUE(network.get_primitive("crop")->can_be_optimized())
        << "a custom layer input should not block in-place cropping";

    auto output = outputs.at("user_kernel").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    // b0 -> features 2,3 == parent 4,5,6,7 ; b1 -> parent 12,13,14,15
    std::vector<float> expected = { 4.f, 5.f, 6.f, 7.f, 12.f, 13.f, 14.f, 15.f };
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_EQ(output_ptr[i], expected[i]) << "element " << i;
}
