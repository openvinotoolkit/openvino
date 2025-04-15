// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "eltwise_inst.h"
#include "reshape_inst.h"

using namespace cldnn;
using namespace ::tests;

#define DEFAULT_BROADCAST_SPEC ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)

namespace {
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
    case eltwise_mode::eq:
        return (float)((float)x == (float)y);
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

    tests::random_generator rg(GET_SUITE_NAME);
    int min_random = -2, max_random = 2;
    VVVVF<T> input1_rnd = rg.generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VVVVF<T> input2_rnd = rg.generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input1_rnd_vec = flatten_4d<T>(test_input_fmt, input1_rnd);
    VF<T> input2_rnd_vec = flatten_4d<T>(test_input_fmt, input2_rnd);

    auto& engine = get_test_engine();
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input1 = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    auto input2 = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), input1->get_layout().with_padding(padding{{ 0, 0, input_padding_y, input_padding_x }, 0 })));
    auto eltwise_prim = eltwise("eltwise", { input_info("reorder1"), input_info("input2") }, mode, DEFAULT_BROADCAST_SPEC);
    eltwise_prim.output_paddings = { padding{ { 0, 0, output_padding_y, output_padding_x }, 0 } };
    topology.add(eltwise_prim);
    primitive_id out_id = "eltwise";
    if (relu)
    {
        topology.add(activation("out", out_id, activation_func::relu, { slope, 0.0f }));
        out_id = "out";
    }
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    VVVVF<T> output_cpu = eltwise_reference<T>(input1_rnd, input2_rnd, mode, relu, slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    ASSERT_EQ(output_layout.format.value, test_input_fmt.value);
    auto output_tensor = output_layout.get_padded_dims();
    int x_size = output_tensor[3];
    int y_size = output_tensor[2];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(y_size, (int)output_cpu[0][0].size());
    ASSERT_EQ(x_size, (int)output_cpu[0][0][0].size());
    ASSERT_EQ(f_size, (int)output_cpu[0].size());
    ASSERT_EQ(b_size, (int)output_cpu.size());

    bool test_is_correct = true;
    VF<T> output_cpu_vec = flatten_4d<T>(test_input_fmt, output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (!floating_point_equal(output_cpu_vec[i], output_ptr[i]) && !(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i]))) {
            test_is_correct = false;
            break;
        }
    }
    ASSERT_EQ(test_is_correct, true) << std::endl
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

void run_eltwise_generic_test(cldnn::eltwise_mode mode) {
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    std::pair<int, int> input_size = { 227, 227 };

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    generic_eltwise_test<float>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, false, 0.f, 0, 0, 0, 0);
    if (f16_supported)
        generic_eltwise_test<ov::float16>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, false, (ov::float16)0.f, 0, 0, 0, 0);

}

template <typename T, typename TOut>
TOut eltwise_int_execute(cldnn::eltwise_mode mode, T x, T y) {
    switch (mode) {
    case eltwise_mode::eq:
        return x == y;
    case eltwise_mode::ne:
        return x != y;
    case eltwise_mode::lt:
        return x < y;
    case eltwise_mode::le:
        return x <= y;
    case eltwise_mode::gt:
        return x > y;
    case eltwise_mode::ge:
        return x >= y;
    case eltwise_mode::logic_and:
        return x && y;
    case eltwise_mode::logic_or:
        return x || y;
    case eltwise_mode::right_shift:
        return x >> y;
    case eltwise_mode::left_shift:
        return x << y;
    case eltwise_mode::bitwise_and:
        return x & y;
    case eltwise_mode::bitwise_or:
        return x | y;
    case eltwise_mode::bitwise_xor:
        return x ^ y;
    default:
        return (TOut)0;
    }
}

template <typename T, typename TOut>
VVVVF<TOut> eltwise_int_reference(VVVVF<T> &input1, VVVVF<T> &input2,
    cldnn::eltwise_mode mode, int input_padding_y = 0,
    int input_padding_x = 0, int output_padding_y = 0,
    int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t output_b = input1.size();
    size_t output_f = input1[0].size();
    size_t output_y = input1[0][0].size() + 2 * padding_y;
    size_t output_x = input1[0][0][0].size() + 2 * padding_x;
    VVVVF<TOut> output(output_b, VVVF<TOut>(output_f, VVF<TOut>(output_y, VF<TOut>(output_x))));

    T res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t f = 0; f < output_f; ++f) {
            for (size_t y = 0; y < input1[0][0].size(); ++y) {
                for (size_t x = 0; x < input1[0][0][0].size(); ++x) {
                    res = eltwise_int_execute<T, TOut>(mode, input1[b][f][y][x], input2[b][f][y][x]);
                    output[b][f][y + padding_y][x + padding_x] = res;
                }
            }
        }
    }
    return output;
}

template <typename T, typename TOut>
void generic_eltwise_int_test(cldnn::format test_input_fmt,
                              int input_b,
                              int input_f,
                              int input_y,
                              int input_x,
                              cldnn::eltwise_mode mode,
                              int input_padding_y,
                              int input_padding_x,
                              int output_padding_y,
                              int output_padding_x,
                              int input1_min_val,
                              int input1_max_val,
                              int input2_min_val,
                              int input2_max_val) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    static_assert(std::is_integral<TOut>::value, "TOut must be an integral type");
    
    tests::random_generator rg(GET_SUITE_NAME);

    VVVVF<T> input1_rnd = rg.generate_random_4d<T>(input_b, input_f, input_y, input_x, input1_min_val, input1_max_val);
    VVVVF<T> input2_rnd = rg.generate_random_4d<T>(input_b, input_f, input_y, input_x, input2_min_val, input2_max_val);
    VF<T> input1_rnd_vec = flatten_4d<T>(test_input_fmt, input1_rnd);
    VF<T> input2_rnd_vec = flatten_4d<T>(test_input_fmt, input2_rnd);

    auto& engine = get_test_engine();
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input1 = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    auto input2 = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), input1->get_layout().with_padding(padding{{ 0, 0, input_padding_y, input_padding_x }, 0 })));
    auto eltwise_prim = eltwise("eltwise", { input_info("reorder1"), input_info("input2") }, mode, DEFAULT_BROADCAST_SPEC);
    eltwise_prim.output_paddings = { padding{ { 0, 0, output_padding_y, output_padding_x }, 0 } };
    topology.add(eltwise_prim);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output_memory = outputs.at("eltwise").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<TOut> output_ptr(output_memory, get_test_stream());

    VVVVF<TOut> output_cpu = eltwise_int_reference<T, TOut>(input1_rnd, input2_rnd, mode, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    ASSERT_EQ(output_layout.format.value, test_input_fmt.value);
    auto output_tensor = output_layout.get_padded_dims();
    int x_size = output_tensor[3];
    int y_size = output_tensor[2];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];

    ASSERT_EQ(y_size, (int)output_cpu[0][0].size());
    ASSERT_EQ(x_size, (int)output_cpu[0][0][0].size());
    ASSERT_EQ(f_size, (int)output_cpu[0].size());
    ASSERT_EQ(b_size, (int)output_cpu.size());

    bool test_is_correct = true;
    VF<TOut> output_cpu_vec = flatten_4d<TOut>(test_input_fmt, output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        const TOut cpu_val = output_cpu_vec[i]; 
        const TOut gpu_val = output_ptr[i];
        if (cpu_val != gpu_val) {
            test_is_correct = false;
            break;
        }
    }
    ASSERT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "eltwise_mode = " << (int)mode << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl
        << "type = " << (sizeof(T) == 1 ? "int8" : "int32") << std::endl;
}

void run_eltwise_bool_generic_test(cldnn::eltwise_mode mode)
{
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    std::pair<int, int> input_size = { 227, 227 };

    generic_eltwise_int_test<int32_t, int8_t>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, 0, 0, 0, 0, -2, 2, -2, 2);
    generic_eltwise_int_test<int8_t, int8_t>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, 0, 0, 0, 0, -2, 2, -2, 2);
}

void run_eltwise_int_shift_generic_test(cldnn::eltwise_mode mode) {
    OPENVINO_ASSERT(mode == eltwise_mode::right_shift || mode == eltwise_mode::left_shift,
                    "Only right_shift amd left_shift mode is supported for this test");
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    const int dim_size = 227;

#define ELTWISE_INT_TEST_CASES(type)                                                             \
    generic_eltwise_int_test<type, type>(test_inputs_fmt,                                        \
                                         1,                                                      \
                                         1,                                                      \
                                         dim_size,                                               \
                                         dim_size,                                               \
                                         mode,                                                   \
                                         0,                                                      \
                                         0,                                                      \
                                         0,                                                      \
                                         0,                                                      \
                                         0,                                                      \
                                         static_cast<int>(std::numeric_limits<type>::max()) /16, \
                                         0,                                                      \
                                         ((sizeof(type) * 8) - 1) / 2);

    ELTWISE_INT_TEST_CASES(int8_t);
    ELTWISE_INT_TEST_CASES(uint8_t);
    ELTWISE_INT_TEST_CASES(int16_t);
    ELTWISE_INT_TEST_CASES(uint16_t);
    ELTWISE_INT_TEST_CASES(int32_t);
    ELTWISE_INT_TEST_CASES(uint32_t);
    ELTWISE_INT_TEST_CASES(int64_t);

#undef ELTWISE_INT_TEST_CASES
}

void run_eltwise_int_bitwise_generic_test(cldnn::eltwise_mode mode) {
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    const int dim_size = 227;

#define ELTWISE_INT_TEST_CASES(type)                                                           \
    generic_eltwise_int_test<type, type>(test_inputs_fmt,                                      \
                                         1,                                                    \
                                         1,                                                    \
                                         dim_size,                                             \
                                         dim_size,                                             \
                                         mode,                                                 \
                                         0,                                                    \
                                         0,                                                    \
                                         0,                                                    \
                                         0,                                                    \
                                         0,                                                    \
                                         static_cast<int>(std::numeric_limits<type>::max())/16,\
                                         0,                                                    \
                                         static_cast<int>(std::numeric_limits<type>::max())/16);

    ELTWISE_INT_TEST_CASES(int8_t);
    ELTWISE_INT_TEST_CASES(uint8_t);
    ELTWISE_INT_TEST_CASES(int16_t);
    ELTWISE_INT_TEST_CASES(uint16_t);
    ELTWISE_INT_TEST_CASES(int32_t);
    ELTWISE_INT_TEST_CASES(uint32_t);
    ELTWISE_INT_TEST_CASES(int64_t);

#undef ELTWISE_INT_TEST_CASES
}
}  // namespace

TEST(eltwise_gpu_f32, equal_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  0, 1, 0, 1,
    //  0, 0, 1, 0,
    //  0, 0, 0, 0,
    //  0, 1, 0, 0

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::eq));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 0, 1, 0, 1,
                                    0, 0, 1, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, not_equal_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  1, 0, 1, 0,
    //  1, 1, 0, 1,
    //  1, 1, 1, 1,
    //  1, 0, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::ne));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 0, 1, 0,
                                    1, 1, 0, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, less_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  0, 0, 0, 0,
    //  1, 1, 0, 0,
    //  1, 1, 1, 0,
    //  0, 0, 0, 0

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::lt));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    1, 1, 0, 0,
                                    1, 1, 1, 0,
                                    0, 0, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, less_equal_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  0, 1, 0, 1,
    //  1, 1, 1, 0,
    //  1, 1, 1, 0,
    //  0, 1, 0, 0

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::le));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 0, 1, 0, 1,
                                    1, 1, 1, 0,
                                    1, 1, 1, 0,
                                    0, 1, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, greater_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  1, 0, 1, 0,
    //  0, 0, 0, 1,
    //  0, 0, 0, 1,
    //  1, 0, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::gt));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 0, 1, 0,
                                    0, 0, 0, 1,
                                    0, 0, 0, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, greater_equal_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  1, 1, 1, 1,
    //  0, 0, 1, 1,
    //  0, 0, 0, 1,
    //  1, 1, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::ge));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    0, 0, 1, 1,
                                    0, 0, 0, 1,
                                    1, 1, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, logicalAND_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  1, 1, 1, 1,
    //  1, 0, 1, 1,
    //  1, 1, 1, 1,
    //  1, 0, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::logic_and));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 0, 1, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, logicalAND_in3_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input3 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Input3
    //  0.f, 0.f, 0.f, 0.f,
    //  0.f, 0.f, 0.f, 0.f,
    //  1.f, 1.f, 1.f, 1.f,
    //  1.f, 1.f, 1.f, 1.f
    //
    //  Output:
    //  0, 0, 0, 0,
    //  0, 0, 0, 0,
    //  1, 1, 1, 1,
    //  1, 0, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    set_values(input3, {
            0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f,
            1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::logic_and));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, logicalOR_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  1, 1, 1, 1,
    //  1, 1, 1, 1,
    //  1, 1, 1, 1,
    //  1, 0, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::logic_or));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, logicalOR_in3_float_out1_int) {
    //  Input3 : 2x2x2x2
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Input3
    //  0.f, 1.f, 1.f, 1.f,
    //  0.f, 1.f, 1.f, 0.f,
    //  1.f, 1.f, 1.f, 1.f,
    //  1.f, 1.f, 1.f, 1.f
    //
    //  Output:
    //  1, 1, 1, 1,
    //  1, 1, 1, 1,
    //  1, 1, 1, 1,
    //  1, 1, 1, 1

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    set_values(input3, {
            0.f, 1.f, 1.f, 1.f,
            0.f, 1.f, 1.f, 0.f,
            1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::logic_or));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, logicalXOR_in2_float_out1_int) {
    //  Input2 : 2x2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  1.f,   2.5f,    5.f,    1.5f,
    //  2.f,   0.f,     6.f,    5.2f,
    //  3.f,   0.5f,    7.f,    12.f,
    //  4.f,   0.f,     8.f,    8.f
    //
    //  Input2
    //  0.5f,  2.5f,    0.5f,   1.5f,
    //  5.f,   7.f,     6.f,    4.f,
    //  15.f,  17.f,    8.f,    10.f,
    //  -2.f,  0.f,     -0.5f,  -2.5f
    //
    //  Output:
    //  0, 0, 0, 0,
    //  0, 1, 0, 0,
    //  0, 0, 0, 0,
    //  0, 0, 0, 0

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    set_values(input1, {
            1.f,   2.5f, 5.f, 1.5f,
            2.f,   0.f, 6.f, 5.2f,
            3.f,  0.5f, 7.f, 12.f,
            4.f, 0.f, 8.f,  8.f
    });

    set_values(input2, {
            0.5f,   2.5f,  0.5f,  1.5f,
            5.f,   7.f,    6.f,   4.f,
            15.f,  17.f,    8.f,  10.f,
            -2.f,  0.f,  -0.5f, -2.5f
    });

    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::logic_xor));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, isfinite_in1_float_out1_int) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 2, 2, 2}});
    set_values(input,
               {
                   1.f,
                   2.5f,
                   5.f,
                   std::numeric_limits<float>::infinity(),
                   2.f,
                   0.f,
                   6.f,
                   -std::numeric_limits<float>::infinity(),
                   3.f,
                   0.5f,
                   7.f,
                   std::numeric_limits<float>::quiet_NaN(),
                   4.f,
                   0.f,
                   8.f,
                   -std::numeric_limits<float>::quiet_NaN(),
               });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(eltwise("eltwise", {input_info("input")}, eltwise_mode::is_finite));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    const auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    const auto output = outputs.at("eltwise").get_memory();
    const cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    const std::vector<int8_t> answers = {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, isinf_in1_float_out1_int) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 2, 2, 2}});
    set_values(input,
               {
                   1.f,
                   2.5f,
                   5.f,
                   std::numeric_limits<float>::infinity(),
                   2.f,
                   0.f,
                   6.f,
                   -std::numeric_limits<float>::infinity(),
                   3.f,
                   0.5f,
                   7.f,
                   std::numeric_limits<float>::quiet_NaN(),
                   4.f,
                   0.f,
                   8.f,
                   -std::numeric_limits<float>::quiet_NaN(),
               });

    const std::vector<std::pair<std::vector<float>, std::vector<int8_t>>> params = {
        {{0.f, 0.f}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0.f, 1.f}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{1.f, 0.f}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{1.f, 1.f}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
    };

    for (const auto& param : params) {
        const auto& coefficients = param.first;
        const auto& answers = param.second;

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(eltwise("eltwise", {input_info("input")}, eltwise_mode::is_inf, coefficients, data_types::i8));

        network network(engine, topology, get_test_default_config(engine));
        network.set_input_data("input", input);

        const auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "eltwise");

        const auto output = outputs.at("eltwise").get_memory();
        const cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i) {
            ASSERT_EQ(answers[i], output_ptr[i]);
        }
    }
}

TEST(eltwise_gpu_f32, isnan_in1_float_out1_int) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 2, 2, 2}});
    set_values(input,
               {
                   1.f,
                   2.5f,
                   5.f,
                   std::numeric_limits<float>::infinity(),
                   2.f,
                   0.f,
                   6.f,
                   -std::numeric_limits<float>::infinity(),
                   3.f,
                   0.5f,
                   7.f,
                   std::numeric_limits<float>::quiet_NaN(),
                   4.f,
                   0.f,
                   8.f,
                   -std::numeric_limits<float>::quiet_NaN(),
               });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(eltwise("eltwise", {input_info("input")}, eltwise_mode::is_nan));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    const auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    const auto output = outputs.at("eltwise").get_memory();
    const cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    const std::vector<int8_t> answers = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32, dynamic_kernel_no_broadcast) {
    auto& engine = get_test_engine();

    ov::Shape in_shape = {2, 2, 2, 2};
    auto in_layout = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f32, format::bfyx};
    auto in_mem_layout = layout{ov::PartialShape(in_shape), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in_mem_layout);
    auto input2 = engine.allocate_memory(in_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in_layout));
    topology.add(input_layout("input2", in_layout));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    set_values(input1, {
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

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("eltwise");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,   5.5f,    4.f,
                          7.f,   7.f,    8.f,   9.2f,
                          18.f,17.5f,   15.f,   22.f,
                          2.f,   6.f,   7.5f,  5.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, dynamic_kernel_broadcast) {
    auto& engine = get_test_engine();

    ov::Shape in1_shape = {2, 2, 2, 2};
    ov::Shape in2_shape = {1, 2, 1, 1};
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto in1_mem_layout = layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx};
    auto in2_mem_layout = layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in1_mem_layout);
    auto input2 = engine.allocate_memory(in2_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    set_values(input1, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, { 0.5f, -0.5f });

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("eltwise");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f,  0.5f, 5.5f, 2.f,
                          1.5f, -0.5f, 5.5f, 4.7f,
                          3.5f,  1.0f, 7.5f, 12.5f,
                          3.5f, -1.0f, 7.5f, 7.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, dynamic_kernel_broadcast_mixed_ranks_3d_2d) {
    auto& engine = get_test_engine();

    ov::PartialShape in1_shape = {3, 1, 5};
    ov::PartialShape in2_shape = {1, 5};
    auto in1_layout = layout{{-1, -1, 5}, data_types::f32, format::bfyx};
    auto in2_layout = layout{{-1, 5}, data_types::f32, format::bfyx};
    auto in1_mem_layout = layout{in1_shape, data_types::f32, format::bfyx};
    auto in2_mem_layout = layout{in2_shape, data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in1_mem_layout);
    auto input2 = engine.allocate_memory(in2_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    set_values(input1, {
        1.f,  0.f , 5.f,  1.5f, 2.f,
        0.f,  6.f, 5.2f,   3.f, 0.5f,
        7.f, 12.f,  4.f, -0.5f, 8.f
    });

    set_values(input2, { 0.5f, -0.5f, 1.0f, -1.0f, 2.f });

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("eltwise");

    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    ov::PartialShape expected_shape{3, 1, 5};

    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    float answers[15] = { 1.5f, -0.5f, 6.f,  0.5f, 4.f,
                          0.5f,  5.5f, 6.2f,  2.f, 2.5f,
                          7.5f, 11.5f, 5.f,  -1.5f, 10.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 15; i++) {
        ASSERT_EQ(answers[i], output_ptr[i]) << "i = " << i;
    }
}

TEST(eltwise_gpu_f32, dynamic_kernel_broadcast_mixed_ranks_5d_2d) {
    auto& engine = get_test_engine();

    ov::PartialShape in1_shape = {1, 1, 3, 1, 5};
    ov::PartialShape in2_shape = {1, 5};
    auto in1_layout = layout{{1, 1, -1, -1, 5}, data_types::f32, format::bfzyx};
    auto in2_layout = layout{{-1, 5}, data_types::f32, format::bfyx};
    auto in1_mem_layout = layout{in1_shape, data_types::f32, format::bfzyx};
    auto in2_mem_layout = layout{in2_shape, data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in1_mem_layout);
    auto input2 = engine.allocate_memory(in2_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    set_values(input1, {
        1.f,  0.f , 5.f,  1.5f, 2.f,
        0.f,  6.f, 5.2f,   3.f, 0.5f,
        7.f, 12.f,  4.f, -0.5f, 8.f
    });

    set_values(input2, { 0.5f, -0.5f, 1.0f, -1.0f, 2.f });

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("eltwise");

    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    ov::PartialShape expected_shape{1, 1, 3, 1, 5};

    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    float answers[15] = { 1.5f, -0.5f, 6.f,  0.5f, 4.f,
                          0.5f,  5.5f, 6.2f,  2.f, 2.5f,
                          7.5f, 11.5f, 5.f,  -1.5f, 10.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 15; i++) {
        ASSERT_EQ(answers[i], output_ptr[i]) << "i = " << i;
    }
}

TEST(eltwise_cpu_impl_f32, dynamic_kernel_broadcast_mixed_ranks_5d_2d) {
    auto& engine = get_test_engine();

    ov::PartialShape in1_shape = {1, 1, 3, 1, 5};
    ov::PartialShape in2_shape = {1, 5};
    auto in1_layout = layout{{1, 1, -1, -1, 5}, data_types::f32, format::bfzyx};
    auto in2_layout = layout{{-1, 5}, data_types::f32, format::bfyx};
    auto in1_mem_layout = layout{in1_shape, data_types::f32, format::bfzyx};
    auto in2_mem_layout = layout{in2_shape, data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in1_mem_layout);
    auto input2 = engine.allocate_memory(in2_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    set_values(input1, {
        1.f,  0.f , 5.f,  1.5f, 2.f,
        0.f,  6.f, 5.2f,   3.f, 0.5f,
        7.f, 12.f,  4.f, -0.5f, 8.f
    });

    set_values(input2, { 0.5f, -0.5f, 1.0f, -1.0f, 2.f });

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"eltwise", {format::bfzyx, "", impl_types::cpu}}}));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("eltwise");

    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    ov::PartialShape expected_shape{1, 1, 3, 1, 5};

    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    float answers[15] = { 1.5f, -0.5f, 6.f,  0.5f, 4.f,
                          0.5f,  5.5f, 6.2f,  2.f, 2.5f,
                          7.5f, 11.5f, 5.f,  -1.5f, 10.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 15; i++) {
        ASSERT_EQ(answers[i], output_ptr[i]) << "i = " << i;
    }
}


// This pattern occurs wrong output index due to miss the dyn_pad offset calculation.
//
// input0: [dynamic_shape]
// input1: [static_shape]
// fused_input0: [dynamic_shape + dyn_pad]
// output: [dynamic_shape]
//
TEST(eltwise_gpu_f32, dynamic_padding) {
    auto& engine = get_test_engine();

    ov::Shape in_shape = {2, 2, 2, 2};
    auto in_layout_1 = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f32, format::bfyx};
    auto in_const_layout = layout{{1, 1, 1, 2}, data_types::f32, format::bfyx};
    const padding::DynamicDimsMask mask("1000");
    auto in_layout_2 = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f32, format::bfyx, padding{{0, 0, 0, 1}, {0, 0, 0, 1}, mask}};
    auto in_mem_layout = layout{ov::PartialShape(in_shape), data_types::f32, format::bfyx};
    auto in_const_mem_layout = layout{{1, 1, 1, 2}, data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(in_mem_layout);
    auto input2 = engine.allocate_memory(in_mem_layout);
    auto input_const = engine.allocate_memory(in_const_mem_layout);

    topology topology;
    topology.add(input_layout("input1", in_layout_1));
    topology.add(data("input_const", input_const));
    topology.add(input_layout("input2", in_layout_2));
    topology.add(eltwise("eltwise1", { input_info("input1"), input_info("input_const") }, eltwise_mode::prod));
    topology.add(eltwise("eltwise2", { input_info("eltwise1"), input_info("input2") }, eltwise_mode::prod));
    topology.add(reorder("out", input_info("eltwise2"), format::bfyx, data_types::f32));

    set_values(input1, {
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

    set_values(input_const, {
        0.5f,   2.5f });

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    // config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("out");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output = outputs.at("out").get_memory();

    float answers[16] = {  0.25f,  0.f,     1.25f,  9.375f,
                           5.f,    0.f,     6.f,   52.f,
                          22.5f,  21.25f,  28.f,  300.f,
                          -4.f,   -8.125f, -2.f,  -50.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_8d) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({{1, 3, 2, 2, 2, 3, 2, 3}, data_types::f32, format::bfvuwzyx });
    auto input2 = engine.allocate_memory({{1, 3, 2, 2, 2, 3, 2, 3}, data_types::f32, format::bfvuwzyx });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    {
        mem_lock<float> lock1(input1, get_test_stream());
        mem_lock<float> lock2(input2, get_test_stream());

        std::iota(lock1.begin(), lock1.end(), 0);
        std::iota(lock2.begin(), lock2.end(), 0);
    }

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    ov::PartialShape expected_shape = {1, 3, 2, 2, 2, 3, 2, 3};
    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output->count(); i++) {
        ASSERT_EQ(2.f*i, output_ptr[i]) << " i = " << i;
    }
}

void eltwise_cpu_impl_f32(bool disable_usm = false);
void eltwise_cpu_impl_f32(bool disable_usm) {
    auto engine = create_test_engine(engine_types::ocl, runtime_types::ocl, !disable_usm);

    auto input1 = engine->allocate_memory({{1, 3, 2, 2, 2, 3, 2, 3}, data_types::f32, format::bfvuwzyx });
    auto input2 = engine->allocate_memory({{1, 3, 2, 2, 2, 3, 2, 3}, data_types::f32, format::bfvuwzyx });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum));

    {
        mem_lock<float> lock1(input1, get_test_stream());
        mem_lock<float> lock2(input2, get_test_stream());

        std::iota(lock1.begin(), lock1.end(), 0);
        std::iota(lock2.begin(), lock2.end(), 0);
    }

    auto config = get_test_default_config(*engine);
    auto forcing_map = ov::intel_gpu::ImplForcingMap{ {"eltwise", {format::bfvuwzyx, "", impl_types::cpu}} };
    config.set_property(ov::intel_gpu::force_implementations(forcing_map));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(*engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    ov::PartialShape expected_shape = {1, 3, 2, 2, 2, 3, 2, 3};
    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output->count(); i++) {
        ASSERT_EQ(2.f*i, output_ptr[i]) << " i = " << i;
    }
}

TEST(eltwise_cpu_impl_f32, add_basic_8d) {
    eltwise_cpu_impl_f32();
}

TEST(eltwise_cpu_impl_f32, add_basic_8d_disable_usm) {
    eltwise_cpu_impl_f32(true);
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

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

    float answers[16] = { 1.5f, 2.5f,   5.5f,    4.f,
                          7.f,   7.f,    8.f,   9.2f,
                          18.f,17.5f,   15.f,   22.f,
                          2.f,   6.f,   7.5f,  5.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_channel) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    set_values(input, {
        1.f,   0.f,
        2.f,   0.f,

        3.f,  0.5f,
        4.f, -0.5f,
    });

    set_values(input2, {
         0.5f,  2.5f,
         0.5f,  2.5f,

          5.f,   7.f,
          2.f,   4.f,

         15.f,  17.f,
          8.f,  10.f,

         -2.f,  6.5f,
        -0.5f, -2.5f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          2.5f, 2.5f,

                          6.f,   7.f,
                          4.f,   4.f,

                          18.f, 17.5f,
                          12.f,  9.5f,

                          1.f,   7.f,
                          3.5f, -3.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_x) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
        1.f,
        0.f,

        2.f,
        0.f,

        3.f,
        0.5f,

        4.f,
        -0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 3.5f,
                          0.5f, 2.5f,

                          7.f,   9.f,
                          2.f,   4.f,

                          18.f, 20.f,
                          8.5f,  10.5f,

                          2.f,   10.5f,
                          -1.f, -3.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_y) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
        1.f, 0.f,
        2.f, 0.f,

        3.f, 0.5f,
        4.f, -0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          2.5f, 2.5f,

                          8.f,   7.5f,
                          6.f,   3.5f,

                          16.f,  17.f,
                          10.f,  10.f,

                          1.f,   7.f,
                          3.5f, -3.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_batch) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2")}, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
        1.f, 0.f,

        2.f, 0.f,

        3.f, 0.5f,

        4.f, -0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          1.5f, 2.5f,

                          7.f,   7.f,
                          4.f,   4.f,

                          18.f,  17.5f,
                          11.f,  10.5f,

                          2.f,   6.f,
                          3.5f, -3.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_multiple_dims) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2")}, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
            1.f,
            2.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 3.5f,
                          1.5f, 3.5f,

                          7.f,   9.f,
                          4.f,   6.f,

                          16.f, 18.f,
                          9.f,  11.f,

                          0.f,   8.5f,
                          1.5f, -0.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, pow_in2x2x2x2_broadcast_all) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::pow));

    set_values(input, {
            1.f,  2.f,
            3.f,  4.f,

            5.f,  6.f,
            7.f,  8.f,

            9.f,  10.f,
           11.f,  12.f,

            13.f, 14.f,
            15.f, 16.f });

    set_values(input2, { 2.0f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.f, 4.f,
                          9.f, 16.f,

                          25.f, 36.f,
                          49.f, 64.f,

                          81.f,  100.f,
                          121.f, 144.f,

                          169.f, 196.f,
                          225.f, 256.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_in2x2x2x2_broadcast_2_inputs_same_dim) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
        1.f, 0.f,

        2.f, 0.f,

        3.f, 0.5f,

        4.f, -0.5f,
    });

    set_values(input3, {
        3.f, 2.f,

        1.f, 2.f,

        -2.f, 1.5f,

        -4.f, 0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.5f, 4.5f,
                          4.5f, 4.5f,

                          8.f,   9.f,
                          5.f,   6.f,

                          16.f,  19.f,
                          9.f,  12.f,

                          -2.f,  6.5f,
                          -0.5f, -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_in2x2x2x2_broadcast_2_inputs_diff_dim) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum));

    set_values(input, {
            0.5f,  2.5f,
            0.5f,  2.5f,

            5.f,   7.f,
            2.f,   4.f,

            15.f,  17.f,
            8.f,  10.f,

            -2.f,  6.5f,
            -0.5f, -2.5f });

    set_values(input2, {
        1.f, 0.f,

        2.f, 0.f,

        3.f, 0.5f,

        4.f, -0.5f,
    });

    set_values(input3, {
        3.f, 2.f,
        1.f, 2.f,

        -2.f, 1.5f,
        -4.f, 0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.5f, 4.5f,
                          2.5f, 4.5f,

                          10.f,  9.f,
                          5.f,   6.f,

                          16.f,  19.f,
                          7.f,  11.f,

                           0.f,   7.5f,
                          -0.5f, -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::max));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
         1.f,   2.5f,  5.f,   2.5f,
         5.f,   7.f,   6.f,   5.2f,
        15.f,  17.f,   8.f,  12.f,
         6.f,   8.f,   8.f,   8.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2")}, eltwise_mode::sub));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
         0.5f,  -2.5f,   4.5f,   2.5f,
        -3.f,   -7.f,    4.f,    3.2f,
       -12.f,  -16.5f,  -1.f,    3.5f,
        -2.f,   -8.5f,   8.5f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_int, basic_in4x4x4x4) {
    //  Same params as in eltwise_gpu_f32, sub_basic_in4x4x4x4 but using int types instead

    std::vector<data_types> data_types_to_test = { data_types::i8, data_types::i32, data_types::i64 };
    std::vector<eltwise_mode> eltwise_ops_to_test = {
        eltwise_mode::sum,
        eltwise_mode::sub,
        eltwise_mode::div,
        eltwise_mode::prod,
        eltwise_mode::min,
        eltwise_mode::max,
        eltwise_mode::mod,
        eltwise_mode::floor_mod
    };

    for (auto& data_type : data_types_to_test)
    {
        for (auto& mode : eltwise_ops_to_test)
        {
            auto& engine = get_test_engine();
            auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
            auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

            topology topology;
            topology.add(input_layout("input", input->get_layout()));
            topology.add(input_layout("input2", input2->get_layout()));
            topology.add(reorder("input_reorder", input_info("input"), { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(reorder("input2_reorder", input_info("input2"), { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(eltwise("eltwise", { input_info("input_reorder"), input_info("input2_reorder") }, mode));
            topology.add(reorder("eltwise_reorder", input_info("eltwise"), { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } }));

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

            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input", input);
            network.set_input_data("input2", input2);
            auto outputs = network.execute();

            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "eltwise_reorder");

            auto output = outputs.at("eltwise_reorder").get_memory();

            cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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
                else if (mode == eltwise_mode::mod)
                    expected = std::fmod(input_1_vec[i], input_2_vec[i]);
                else if (mode == eltwise_mode::floor_mod) {
                    const double divisor = static_cast<double>(input_2_vec[i]);
                    expected =  input_1_vec[i] - input_2_vec[i] * std::floor(input_1_vec[i] / divisor);
                }

                ASSERT_TRUE(are_equal(std::floor(expected), output_ptr[i]));
            }
        }
    }
}

TEST(eltwise_gpu_int, div_gather_fusing) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 3, 2, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 2, 2, 2 } });       // 2nd input of eltwise

    set_values(input1, {
        5, 6, 7, 8,
        -5, -6, -7, -8,
        9, 10, 11, 12
    });

    set_values(input2, {
        0, 1,
        2, 1
    });

    std::vector<int32_t> input_3_vec = {
        2, 2, 2, 2,
        2, 2, 2, 2,
        -2, -2, -2, -2,
        -2, -2, -2, -2
    };
    set_values(input3, input_3_vec);

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("Input3", input3->get_layout()));
    topology.add(gather("gather", input_info("InputDictionary"), input_info("InputText"), 0, 4, ov::Shape{2, 2, 2, 2}));
    topology.add(reorder("gather_reorder", input_info("gather"), { data_types::i32, format::bfyx, { 2, 2, 2, 2 } }));
    topology.add(eltwise("eltwise",
                 { input_info("gather_reorder"), input_info("Input3") },
                 eltwise_mode::div,
                 std::vector<float>(0),
                 data_types::i32,
                 DEFAULT_BROADCAST_SPEC,
                 true));
    topology.add(reorder("eltwise_reorder", input_info("eltwise"), { data_types::i32, format::bfyx, { 2, 2, 2, 2 } }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("Input3", input3);

    auto outputs = network.execute();

    auto output = outputs.at("eltwise_reorder").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> gather_expected_results = {
        5, 6, 7, 8,
        -5, -6, -7, -8,
        9, 10, 11, 12,
        -5, -6, -7, -8
    };

    for (size_t i = 0; i < 16; ++i) {
        auto expected = gather_expected_results[i] / input_3_vec[i];
        auto rem = gather_expected_results[i] % input_3_vec[i];
        if (rem != 0 && (gather_expected_results[i] < 0) != (input_3_vec[i] < 0))
            expected -= 1;
        ASSERT_EQ(expected, output_ptr[i]);
    }
}

TEST(eltwise_gpu_f32_int, basic_in4x4x4x4) {
    // Same params as in eltwise_gpu_f32, sub_basic_in4x4x4x4 but using int types for first input.
    //
    // Eltwise supports mixed inputs, but only first input can be set as intX.

    std::vector<data_types> data_types_to_test = { data_types::i8, data_types::i32, data_types::i64 };
    std::vector<eltwise_mode> eltwise_ops_to_test = { eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::div, eltwise_mode::prod, eltwise_mode::min, eltwise_mode::max, eltwise_mode::mod };

    for (auto& data_type : data_types_to_test)
    {
        for (auto& mode : eltwise_ops_to_test)
        {
            auto& engine = get_test_engine();
            auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
            auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

            topology topology;
            topology.add(input_layout("input", input->get_layout()));
            topology.add(input_layout("input2", input2->get_layout()));
            topology.add(reorder("input_reorder", input_info("input"), { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(eltwise("eltwise", { input_info("input_reorder"), input_info("input2") }, mode));
            topology.add(reorder("eltwise_reorder", input_info("eltwise"), { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } }));

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

            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input", input);
            network.set_input_data("input2", input2);
            auto outputs = network.execute();

            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "eltwise_reorder");

            auto output = outputs.at("eltwise_reorder").get_memory();

            cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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
                else if (mode == eltwise_mode::mod)
                    expected = std::fmod(input_1_vec[i], input_2_vec[i]);

                ASSERT_TRUE(are_equal(std::floor(expected), output_ptr[i]));
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

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::prod));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        0.5f,   0.0f,    75.f,    6.0f,
        1.0f,   0.0f,    48.f,   -2.6f,
        7.5f,   3.5f,   119.f,   96.0f,
       10.0f,  -2.0f,    80.f, -18.75f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(reorder("reorder2", input_info("input2"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(eltwise("eltwise", { input_info("reorder"), input_info("reorder2") }, eltwise_mode::max));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.f,   2.5f,  5.f,   2.5f,
        5.f,   7.f,   6.f,   5.2f,
        15.f,  17.f,   8.f,  12.f,
        6.f,   8.f,   8.f,   8.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum, {0.5f, 0.5f}, data_types::f32));

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

    float answers[16] = { 0.75f, 1.25f, 2.75f,  2.f,
                          3.5f,  3.5f,   4.f, 4.6f,
                          9.f, 8.75f,  7.5f, 11.f,
                          1.f,   3.f, 3.75f, 2.75f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, coefficients_count_check) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));

    std::vector<float> coeffs0 = {};
    std::vector<float> coeffs1 = {0.5f};
    std::vector<float> coeffs2 = {0.5f, 0.5f};
    std::vector<float> coeffs3 = {0.5f, 0.5f, 0.5f};

    EXPECT_THROW(topology.add(eltwise("eltwise1", { input_info("input"), input_info("input2") }, eltwise_mode::sum, coeffs1, data_types::f32)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise2", { input_info("input"), input_info("input2") }, eltwise_mode::sum, coeffs3, data_types::f32)), std::invalid_argument);

    EXPECT_THROW(topology.add(eltwise("eltwise3", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum, coeffs1, data_types::f32)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise4", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum, coeffs2, data_types::f32)), std::invalid_argument);

    EXPECT_NO_THROW(topology.add(eltwise("eltwise5", { input_info("input"), input_info("input2") }, eltwise_mode::sum, coeffs0, data_types::f32)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise6", { input_info("input"), input_info("input2") }, eltwise_mode::sum, coeffs2, data_types::f32)));

    EXPECT_NO_THROW(topology.add(eltwise("eltwise7", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum, coeffs0, data_types::f32)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise8", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum, coeffs3, data_types::f32)));
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2"), input_info("input3") }, eltwise_mode::sum, {0.5f, 0.5f, 0.5f}, data_types::f32));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.75f, 1.25f, 4.75f,  2.f,
                          7.0f,  4.0f,  5.5f, 5.1f,
                          12.f, 8.75f,  8.5f, 11.f,
                          3.5f,  3.5f, 4.25f, 3.25f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(reorder("reorder2", input_info("input2"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(reorder("reorder3", input_info("input3"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })));
    topology.add(eltwise("eltwise", { input_info("reorder"), input_info("reorder2"), input_info("reorder3") }, eltwise_mode::max));

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

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.1f,   4.f,  5.f,   2.5f,
        5.f,   7.f,   6.f,   7.f,
        15.f,  17.f,   9.f,  12.f,
        6.f,   8.f,   8.f,   8.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, stride_test_2x2) {
    //  Input  : 2x2x2x2
    //  Input2 : 2x2x4x4
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 1  2  3  4   b1: 17 18 19 20
    //  f0: b0: 5  6  7  8   b1: 21 22 23 24
    //  f0: b0: 9  10 11 12  b1: 25 26 27 28
    //  f0: b0: 13 14 15 16  b1: 29 30 31 32

    //  f1: b0: 33 34 35 36  b1: 49 50 51 52
    //  f1: b0: 37 38 39 40  b1: 53 54 55 56
    //  f1: b0: 41 42 43 44  b1: 57 58 59 60
    //  f1: b0: 45 46 47 48  b1: 61 62 63 64

    //
    //  Output:
    //  f0: b0: 1  3    b1:  17  19
    //  f0: b0: 9  11   b1:  25  27
    //  f1: b0: 33 35   b1:  49  51
    //  f1: b0: 41 43   b1:  57  59
    //
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 4, 4 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", input_info("input"), input_info("input2"), { {0,0,1,1}, {0,0,2,2} }, eltwise_mode::max));

    set_values(input, {
        1.f,   0.f,  5.f,  1.5f,
        2.f,   0.f,  6.f,  5.2f,
        3.f,   0.5f, 7.f, 12.f,
        4.f,  -0.5f, 8.f,  8.f
        });

    set_values<float>(input2, {
        1,  17, 33, 49,
        2,  18, 33, 50,
        3,  19, 35, 51,
        4,  20, 36, 52,
        5,  21, 37, 53,
        6,  22, 38, 54,
        7,  23, 39, 55,
        8,  24, 40, 56,
        9,  25, 41, 57,
        10, 26, 42, 58,
        11, 27, 43, 59,
        12, 28, 44, 60,
        13, 29, 45, 61,
        14, 30, 46, 62,
        15, 31, 47, 63,
        16, 32, 48, 64 });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1,  17,  33,  49,
        3,  19,  35,  51,
        9,  25,  41,  57,
        11, 27,  43,  59 };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, broadcast_test_in4x4x2x2) {
    //  Input2   : 2x1x1
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 0.5  b1: 2.5
    //  f1: b0: 0.5  b1: 2.5
    //
    //  Output:
    //  f0: b0:   1.5  7    b1:  2.5   7
    //  f0: b0:   18   2    b1:  17.5  6
    //  f1: b0:   5.5  8    b1:   4    9.2
    //  f1: b0:   15  16.5  b1:  22    16.5
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.5f,   2.5f,   5.5f,   4.f,
        2.5f,   2.5f,   6.5f,   7.7f,
        3.5f,    3.f,   7.5f,  14.5f,
        4.5f,    2.f,   8.5f,  10.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, broadcast_test_dim3_dim4) {
    auto& engine = get_test_engine();

    ov::Shape in2_shape = {1, 1, 4, 1};
    auto input2 = engine.allocate_memory({ ov::PartialShape(in2_shape), data_types::f32, format::bfyx });

    std::vector<float> const_input = {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    };

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f
    });

    float answers[16] = {
        1.5, 0.5, 7.5, 4,
        2.5, 0.5, 8.5, 7.7,
        3.5, 1, 9.5, 14.5,
        4.5, 0, 10.5, 10.5
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    // in1:dim3, int2:dim4
    {
        ov::Shape in1_shape = {2, 4, 2};

        auto input = engine.allocate_memory({ ov::PartialShape(in1_shape), data_types::f32, format::bfyx });
        set_values(input, const_input);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", input2->get_layout()));
        topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

        network network(engine, topology, config);

        network.set_input_data("input", input);
        network.set_input_data("input2", input2);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "eltwise");

        auto output = outputs.at("eltwise").get_memory();

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (int i = 0; i < 16; i++)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    // in1:extended_dim4_from_dim3, int2:dim4
    // in1_shape = {2, 4, 2} is extended to {1, 2, 4, 2} internally in case allow_new_shape_infer true.
    // So explicit 4d input shpae {1, 2, 4, 2} should have same result from input{2, 4, 2}
    {
        ov::Shape in1_shape = {1, 2, 4, 2};

        auto input = engine.allocate_memory({ ov::PartialShape(in1_shape), data_types::f32, format::bfyx });
        set_values(input, const_input);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", input2->get_layout()));
        topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

        network network(engine, topology, config);

        network.set_input_data("input", input);
        network.set_input_data("input2", input2);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "eltwise");

        auto output = outputs.at("eltwise").get_memory();

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (int i = 0; i < 16; i++)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }
}

TEST(eltwise_gpu_f16, fs_b_yx_fsv32_basic)
{
    // Inputs are 2x2x2x2
    // Inputs:
    //       B1          B2
    //
    //    1111 1112  2111 2112
    //F1
    //    1121 1122  2121 2122
    //
    //
    //    1211 1212  2211 2212
    //F2
    //    1221 1222  2221 2222

    tensor input_tensor(2, 2, 2, 2);
    auto fp16_bfyx_2x2x2x2_input =
    {
        ov::float16(1111),ov::float16(1112),ov::float16(1121),ov::float16(1122),
        ov::float16(1211),ov::float16(1212),ov::float16(1221),ov::float16(1222),
        ov::float16(2111),ov::float16(2112),ov::float16(2121),ov::float16(2122),
        ov::float16(2211),ov::float16(2212),ov::float16(2221),ov::float16(2222)
    };

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_2x2x2x2_input);
    set_values(input2, fp16_bfyx_2x2x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1->get_layout()));
    golden_topology.add(input_layout("input2", input2->get_layout()));
    golden_topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::sum));

    network golden_network(engine, golden_topology, get_test_default_config(engine));
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16> golden_ptr(golden_output, get_test_stream());
    // GOLDEN BFYX ELTWISE - END
    // FS_B_YX_FSV32 ELTWISE
    topology FSV32_topology;
    FSV32_topology.add(input_layout("input1", input1->get_layout()));
    FSV32_topology.add(input_layout("input2", input2->get_layout()));
    FSV32_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FSV32_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FSV32_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum));
    FSV32_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor)));

    network FSV32_network(engine, FSV32_topology, get_test_default_config(engine));
    FSV32_network.set_input_data("input1", input1);
    FSV32_network.set_input_data("input2", input2);

    auto FSV32_outputs = FSV32_network.execute();
    auto FSV32_output = FSV32_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16> FSV32_ptr(FSV32_output, get_test_stream());
    // FS_B_YX_FSV32 ELTWISE - END

    ASSERT_EQ(golden_ptr.size(), FSV32_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++) {
        ASSERT_EQ(float(golden_ptr[i]), float(FSV32_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, fs_b_yx_fsv32_broadcast)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    tensor::value_type input_b = 2;
    tensor::value_type input_f = 72;
    tensor::value_type input1_y = 10, input1_x = 10;
    tensor::value_type input2_y = 1, input2_x = 1;

    tensor input1_tensor(input_b, input_f, input1_x, input1_y);
    tensor input2_tensor(input_b, input_f, input2_x, input2_y);

    VVVVF<ov::float16> input1_rnd = rg.generate_random_4d<ov::float16>(input_b, input_f, input1_y, input1_x, 1, 3);
    VVVVF<ov::float16> input2_rnd = rg.generate_random_4d<ov::float16>(input_b, input_f, input2_y, input2_x, 1, 3);

    VF<ov::float16> input1_flatten = flatten_4d<ov::float16>(format::bfyx, input1_rnd);
    VF<ov::float16> input2_flatten = flatten_4d<ov::float16>(format::bfyx, input2_rnd);

    auto input1 = engine.allocate_memory({ data_types::f16,format::bfyx, input1_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16,format::bfyx, input2_tensor });

    set_values(input1, input1_flatten);
    set_values(input2, input2_flatten);

    topology ref_topology;
    ref_topology.add(input_layout("input1", input1->get_layout()));
    ref_topology.add(input_layout("input2", input2->get_layout()));
    ref_topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::prod));

    network ref_network(engine, ref_topology, get_test_default_config(engine));
    ref_network.set_input_data("input1", input1);
    ref_network.set_input_data("input2", input2);

    auto ref_outputs = ref_network.execute();
    auto ref_output = ref_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16> ref_ptr(ref_output, get_test_stream());

    topology fsv32_topology;
    fsv32_topology.add(input_layout("input1", input1->get_layout()));
    fsv32_topology.add(input_layout("input2", input2->get_layout()));
    fsv32_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input1_tensor)));
    fsv32_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::fs_b_yx_fsv32, input2_tensor)));
    fsv32_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::prod));
    fsv32_topology.add(reorder("reorder_bfyx", input_info("eltwise"), layout(data_types::f16, format::bfyx, input1_tensor)));

    network fsv32_network(engine, fsv32_topology, get_test_default_config(engine));
    fsv32_network.set_input_data("input1", input1);
    fsv32_network.set_input_data("input2", input2);

    auto fsv32_outputs = fsv32_network.execute();
    auto fsv32_output = fsv32_outputs.at("reorder_bfyx").get_memory();
    cldnn::mem_lock<ov::float16> fsv32_ptr(fsv32_output, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), fsv32_ptr.size());

    for (size_t i = 0; i < ref_ptr.size(); i++) {
        ASSERT_EQ(float(ref_ptr[i]), float(fsv32_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, fs_b_yx_fsv32_broadcast_bfyx)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    tensor::value_type input_b = 2;
    tensor::value_type input_f = 72;
    tensor::value_type input1_y = 10, input1_x = 10;

    tensor input1_tensor(input_b, input_f, input1_x, input1_y);
    tensor input2_tensor(1, input_f, 1, 1);

    VVVVF<ov::float16> input1_rnd = rg.generate_random_4d<ov::float16>(input_b, input_f, input1_y, input1_x, 1, 3);
    VVVVF<ov::float16> input2_rnd = rg.generate_random_4d<ov::float16>(1, input_f, 1, 1, 1, 3);

    VF<ov::float16> input1_flatten = flatten_4d<ov::float16>(format::bfyx, input1_rnd);
    VF<ov::float16> input2_flatten = flatten_4d<ov::float16>(format::bfyx, input2_rnd);

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, input1_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, input2_tensor });

    set_values(input1, input1_flatten);
    set_values(input2, input2_flatten);

    topology ref_topology;
    ref_topology.add(input_layout("input1", input1->get_layout()));
    ref_topology.add(input_layout("input2", input2->get_layout()));
    ref_topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::prod));

    network ref_network(engine, ref_topology, get_test_default_config(engine));
    ref_network.set_input_data("input1", input1);
    ref_network.set_input_data("input2", input2);

    auto ref_outputs = ref_network.execute();
    auto ref_output = ref_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16> ref_ptr(ref_output, get_test_stream());

    topology fsv32_topology;
    fsv32_topology.add(input_layout("input1", input1->get_layout()));
    fsv32_topology.add(input_layout("input2", input2->get_layout()));
    fsv32_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input1_tensor)));
    fsv32_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("input2"), eltwise_mode::prod));
    fsv32_topology.add(reorder("reorder_bfyx", input_info("eltwise"), layout(data_types::f16, format::bfyx, input1_tensor)));

    network fsv32_network(engine, fsv32_topology, get_test_default_config(engine));
    fsv32_network.set_input_data("input1", input1);
    fsv32_network.set_input_data("input2", input2);

    auto fsv32_outputs = fsv32_network.execute();
    auto fsv32_output = fsv32_outputs.at("reorder_bfyx").get_memory();
    cldnn::mem_lock<ov::float16> fsv32_ptr(fsv32_output, get_test_stream());

    ASSERT_EQ(ref_ptr.size(), fsv32_ptr.size());

    for (size_t i = 0; i < ref_ptr.size(); i++) {
        ASSERT_EQ(float(ref_ptr[i]), float(fsv32_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, broadcast_test_in4x4x2x2x2) {
    //  Input  : 2x2x2x2x1
    //  Input2 : 2x2x1x1x2
    //  Output : 2x2x2x2x2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 2, 2, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 1, 1, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise("eltwise", { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, { 0.5f, 2.5f, 0.5f, 2.5f, 1.f, 2.f, 3.f, 4.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[32] = {
        1.5f,   0.5f,   5.5f,   2.f,
        3.5f,   2.5f,   7.5f,   4.f,
        2.5f,   0.5f,   6.5f,   5.7f,
        4.5f,   2.5f,   8.5f,   7.7f,
        4.f,    1.5f,   8.f,    13.f,
        5.f,    2.5f,   9.f,    14.f,
        7.f,    2.5f,   11.f,   11.f,
        8.f,    3.5f,   12.f,   12.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 32; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_basic)
{
    // Inputs are 32x96x2x2
    tests::random_generator rg(GET_SUITE_NAME);
    tensor input_tensor(32, 96, 20, 20);
    VVVVF<ov::float16> input_rnd = rg.generate_random_4d<ov::float16>(32, 96, 20, 20, 1, 3);
    VF<ov::float16> fp16_bfyx_32x96x2x2_input = flatten_4d<ov::float16>(format::bfyx, input_rnd);

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1->get_layout()));
    golden_topology.add(input_layout("input2", input2->get_layout()));
    golden_topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::sum));

    network golden_network(engine, golden_topology, get_test_default_config(engine));
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16> golden_ptr(golden_output, get_test_stream());
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::byxf, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor)));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology, get_test_default_config(engine));
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16> FS_B_YX_FSV32_OUTPUT_ptr(FS_B_YX_FSV32_OUTPUT_output, get_test_stream());
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::byxf, input_tensor)));
    BYXF_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    BYXF_OUTPUT_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum));
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor)));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology, get_test_default_config(engine));
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16> BYXF_OUTPUT_ptr(BYXF_OUTPUT_output, get_test_stream());
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++) {
        ASSERT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++) {
        ASSERT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_output_padding) {
    // Inputs are 32x96x2x2
    tests::random_generator rg(GET_SUITE_NAME);
    tensor input_tensor(32, 96, 20, 20);
    VVVVF<ov::float16> input_rnd = rg.generate_random_4d<ov::float16>(32, 96, 20, 20, 1, 3);
    VF<ov::float16> fp16_bfyx_32x96x2x2_input = flatten_4d<ov::float16>(format::bfyx, input_rnd);

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1->get_layout()));
    golden_topology.add(input_layout("input2", input2->get_layout()));
    auto golden_eltwise_prim = eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::sum, DEFAULT_BROADCAST_SPEC);
    golden_eltwise_prim.output_paddings = {  padding{ {0,0,10,5} , 0 } };
    golden_topology.add(golden_eltwise_prim);

    network golden_network(engine, golden_topology, get_test_default_config(engine));
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> golden_ptr(golden_output, get_test_stream());
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::byxf, input_tensor)));
    auto FS_B_YX_FSV32_OUTPUT_eltwise_prim = eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum, DEFAULT_BROADCAST_SPEC);
    FS_B_YX_FSV32_OUTPUT_eltwise_prim.output_paddings = {  padding{ {0,0,10,5} , 0 } };
    FS_B_YX_FSV32_OUTPUT_topology.add(FS_B_YX_FSV32_OUTPUT_eltwise_prim);
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor,
                                              padding{ {0,0,10,5} , 0 })));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology, get_test_default_config(engine));
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> FS_B_YX_FSV32_OUTPUT_ptr(FS_B_YX_FSV32_OUTPUT_output, get_test_stream());
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::byxf, input_tensor)));
    BYXF_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    auto BYXF_OUTPUT_eltwise_prim = eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum, DEFAULT_BROADCAST_SPEC);
    BYXF_OUTPUT_eltwise_prim.output_paddings = {  padding{ {0,0,10,5} , 0 } };
    BYXF_OUTPUT_topology.add(BYXF_OUTPUT_eltwise_prim);
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor,
                                     padding{ {0,0,10,5} , 0 })));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology, get_test_default_config(engine));
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> BYXF_OUTPUT_ptr(BYXF_OUTPUT_output, get_test_stream());
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++) {
        ASSERT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++) {
        ASSERT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_input_padding)
{
    // Inputs are 32x96x20x20
    tests::random_generator rg(GET_SUITE_NAME);
    tensor input_tensor(32, 96, 20, 20);
    VVVVF<ov::float16> input_rnd = rg.generate_random_4d<ov::float16>(32, 96, 20, 20, 1, 3);
    VF<ov::float16> fp16_bfyx_32x96x2x2_input = flatten_4d<ov::float16>(format::bfyx, input_rnd);

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });
    auto input2 = engine.allocate_memory({ data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1->get_layout()));
    golden_topology.add(input_layout("input2", input2->get_layout()));
    golden_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::bfyx, input_tensor, padding{ {0,0,15,10}})));
    golden_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::bfyx, input_tensor, padding{ {0,0,7,5}})));
    golden_topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::sum));

    network golden_network(engine, golden_topology, get_test_default_config(engine));
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    cldnn::mem_lock<ov::float16> golden_ptr(golden_output, get_test_stream());
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ {0,0,15,10} })));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::byxf, input_tensor, padding{ {0,0,7,5} })));
    FS_B_YX_FSV32_OUTPUT_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor)));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology, get_test_default_config(engine));
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16> FS_B_YX_FSV32_OUTPUT_ptr(FS_B_YX_FSV32_OUTPUT_output, get_test_stream());
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1->get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2->get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", input_info("input1"), layout(data_types::f16, format::byxf, input_tensor, padding{ {0,0,15,10} })));
    BYXF_OUTPUT_topology.add(reorder("reorder2", input_info("input2"), layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ {0,0,7,5} })));
    BYXF_OUTPUT_topology.add(eltwise("eltwise", input_info("reorder1"), input_info("reorder2"), eltwise_mode::sum));
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", input_info("eltwise"), layout(data_types::f16, format::bfyx, input_tensor)));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology, get_test_default_config(engine));
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    cldnn::mem_lock<ov::float16> BYXF_OUTPUT_ptr(BYXF_OUTPUT_output, get_test_stream());
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        ASSERT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        ASSERT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

TEST(eltwise_gpu_bool, eltwise_eq) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::eq);
}

TEST(eltwise_gpu_bool, eltwise_ne) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::ne);
}

TEST(eltwise_gpu_bool, eltwise_lt) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::lt);
}

TEST(eltwise_gpu_bool, eltwise_le) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::le);
}

TEST(eltwise_gpu_bool, eltwise_gt) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::gt);
}

TEST(eltwise_gpu_bool, eltwise_ge) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::ge);
}

TEST(eltwise_gpu_bool, eltwise_and) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::logic_and);
}

TEST(eltwise_gpu_bool, eltwise_or) {
    run_eltwise_bool_generic_test(cldnn::eltwise_mode::logic_or);
}

TEST(eltwise_gpu, eltwise_right_shift) {
    run_eltwise_int_shift_generic_test(cldnn::eltwise_mode::right_shift);
}

TEST(eltwise_gpu, eltwise_left_shift) {
    run_eltwise_int_shift_generic_test(cldnn::eltwise_mode::left_shift);
}

TEST(eltwise_gpu, eltwise_bitwise_and) {
    run_eltwise_int_bitwise_generic_test(cldnn::eltwise_mode::bitwise_and);
}

TEST(eltwise_gpu, eltwise_bitwise_or) {
    run_eltwise_int_bitwise_generic_test(cldnn::eltwise_mode::bitwise_or);
}

TEST(eltwise_gpu, eltwise_bitwise_xor) {
    run_eltwise_int_bitwise_generic_test(cldnn::eltwise_mode::bitwise_xor);
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


TEST(eltwise_gpu, b_fs_yx_fsv4_wo_callib) {
    //
    // Input data
    const int BATCH = 1;
    const int in_B = BATCH;

    auto& engine = get_test_engine();

    int in_F = 256;

    int in_X = 56,
        in_Y = in_X;

    // Input data init
    std::vector<char> Data(in_B * in_F * in_X * in_Y);
    for (size_t i = 0; i < Data.size(); i++)
        Data[i] = static_cast<char>(i);
    std::vector<char> DataGold(Data);

    // Mem initialization
    // This is user data, no kernels here
    auto input1 = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
    std::vector<char> data_i1(DataGold);
    for (size_t i = 0; i < data_i1.size(); i++) data_i1[i] = data_i1[i] + 1;
    set_values(input1, std::move(data_i1));

    auto input2 = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
    std::vector<char> data_i2(DataGold);
    for (size_t i = 0; i < data_i2.size(); i++) data_i2[i] = data_i2[i] + 2;
    set_values(input2, std::move(data_i2));

    auto input3 = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
    std::vector<char> data_i3(DataGold);
    for (size_t i = 0; i < data_i3.size(); i++) data_i3[i] = data_i3[i] + 3;
    set_values(input3, std::move(data_i3));

    cldnn::eltwise_mode mode[] = { cldnn::eltwise_mode::min,
                                   cldnn::eltwise_mode::max,
                                   cldnn::eltwise_mode::sum };

    for (int i = 0; i < 3; i++) {
        // Expected "gold" output and IMAD output.
        std::vector<char>  vGoldOutput;
        std::vector<char>  vTestOutput;

        // "Golden" Eltwise
        {
            topology topology;

            auto eltw = eltwise("eltw_GOLD_no_relu",
                                { input_info("input1"), input_info("input2"), input_info("input3") },
                                mode[i]);
            auto actv = activation("eltw_GOLD", input_info(eltw), activation_func::relu);

            // Create a topology
            topology.add(input_layout("input1", input1->get_layout()),
                         input_layout("input2", input2->get_layout()),
                         input_layout("input3", input3->get_layout()),
                         eltw, actv);

            // Network processing
            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input1", input1);
            network.set_input_data("input2", input2);
            network.set_input_data("input3", input3);
            auto outputs = network.execute();

            // Validation
            auto searchC = outputs.find("eltw_GOLD");
            EXPECT_NE(searchC, outputs.end());
            auto output = outputs.begin()->second.get_memory();
            cldnn::mem_lock<char> output_ptr(output, get_test_stream());
            vGoldOutput.reserve(output_ptr.size());
            for (size_t i = 0; i < output_ptr.size(); i++)
                vGoldOutput.push_back(output_ptr[i]);
        }

        // "IMAD" Eltwise
        {
            topology topology;

            // Reorder (a-ka swizzelling) input to MMAD/IMAD Pooling format
            topology.add(reorder("reorder1_Swizzelled",
                                 input_info("input1"),
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })),
                         reorder("reorder2_Swizzelled",
                                 input_info("input2"),
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })),
                         reorder("reorder3_Swizzelled",
                                 input_info("input3"),
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })));

            auto eltw = eltwise("eltw_IMAD_no_relu",
                                { input_info("reorder1_Swizzelled"),
                                  input_info("reorder2_Swizzelled"),
                                  input_info("reorder3_Swizzelled") },
                                mode[i]);
            auto actv = activation("eltw_IMAD", input_info(eltw), activation_func::relu);
            topology.add(input_layout("input1", input1->get_layout()),
                         input_layout("input2", input2->get_layout()),
                         input_layout("input3", input3->get_layout()),
                         eltw, actv);

            // Back reordering (a-ka unswizzelling) output from MMAD/IMAD pooling
            topology.add(reorder("reorder_UnSwizzelled",
                                 input_info("eltw_IMAD"),
                                 layout(data_types::i8,
                                     format::bfyx,
                                     { in_B, in_F, in_X, in_Y })));

            // Network processing
            network network(engine, topology, get_test_default_config(engine));
            network.set_input_data("input1", input1);
            network.set_input_data("input2", input2);
            network.set_input_data("input3", input3);
            auto outputs = network.execute();

            // Validation
            auto searchC = outputs.find("reorder_UnSwizzelled");
            EXPECT_NE(searchC, outputs.end());
            auto output = outputs.begin()->second.get_memory();
            cldnn::mem_lock<char> output_ptr(output, get_test_stream());
            vTestOutput.reserve(output_ptr.size());
            for (size_t i = 0; i < output_ptr.size(); i++)
                vTestOutput.push_back(output_ptr[i]);
        }

        // Result validation
        ASSERT_TRUE(vGoldOutput.size() == vTestOutput.size());
        for (size_t i = 0; i < vGoldOutput.size(); i++)
            ASSERT_TRUE(vTestOutput[i] == vGoldOutput[i]);
    }
}

TEST(DISABLED_eltwise_gpu, generic_random) {
    VF<cldnn::format> test_inputs_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<cldnn::eltwise_mode> modes = { cldnn::eltwise_mode::sum, cldnn::eltwise_mode::sub, cldnn::eltwise_mode::max, cldnn::eltwise_mode::prod };
    VF<bool> relu = { true, false };
    VF<float> slopes = { 0.0f, -0.0f, -17.19f, 1028.8f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
    std::vector<std::pair<int, int>> input_sizes = { { 100, 100 },{ 227, 227 },{ 400, 600 } };

    auto& engine = get_test_engine();
    bool f16_supported = engine.get_device_info().supports_fp16;
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
                                                generic_eltwise_test<ov::float16>(test_input_fmt, input_b, input_f, input_yx.first, input_yx.second, mode, relu_activated, (ov::float16)slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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

//
struct eltwise_same_input_test_params {
    data_types  input_type;
    tensor      input_size;

    format::type in_format;
    format::type out_format;
};

struct eltwise_same_input_test : testing::TestWithParam<eltwise_same_input_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr input_ref) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = input_ref->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        mem_lock<T> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T> input_ptr{input_ref, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto input_out_val = input_ptr[opt_out_offset];

                        ASSERT_EQ((input_out_val+input_out_val), ref_out_val);
                        // ASSERT_NEAR(static_cast<float>(opt_out_val), static_cast<float>(ref_out_val), 1.e-1f);
                    }
                }
            }
        }
    }

    void execute_same_input(const eltwise_same_input_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);
        auto input = engine.allocate_memory(in_layout);
        fill_random(input);

        cldnn::topology topo;
        topo.add(input_layout("input1", input->get_layout()));
        topo.add(input_layout("input2", input->get_layout()));
        auto prim = eltwise("eltwise", { input_info("input1"), input_info("input2") }, eltwise_mode::sum);
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"eltwise"}));

        cldnn::network net(engine, topo, config);
        net.set_input_data("input1", input);
        net.set_input_data("input2", input);

        auto result = net.execute();
        auto output = result.at("eltwise").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare_outputs<float>(output, input);
            } else if (params.input_type == data_types::f16) {
                compare_outputs<ov::float16>(output, input);
            } else if (params.input_type == data_types::i8) {
                compare_outputs<int8_t>(output, input);
            } else if (params.input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, input);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

struct eltwise_same_input_param_generator : std::vector<eltwise_same_input_test_params> {
    eltwise_same_input_param_generator& add(eltwise_same_input_test_params params) {
        push_back(params);
        return *this;
    }

    eltwise_same_input_param_generator& simple_params(data_types type, format::type input_format, format::type output_format) {
        push_back(eltwise_same_input_test_params{ type, {1, 40, 4, 4}, input_format, output_format});
        push_back(eltwise_same_input_test_params{ type, {1, 5, 4, 4}, input_format, output_format});
        return *this;
    }
};

TEST_P(eltwise_same_input_test, random) {
    auto param = GetParam();
    execute_same_input(param, true);
}

INSTANTIATE_TEST_SUITE_P(eltwise_same_input,
                        eltwise_same_input_test,
                        testing::ValuesIn(
                            eltwise_same_input_param_generator()
                            .simple_params(data_types::f32, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .simple_params(data_types::f32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::f32, format::fs_b_yx_fsv32, format::fs_b_yx_fsv32)
                        ));

// mode, input type, input sizes
using eltwise_test_params = std::tuple<eltwise_mode, data_types, std::vector<std::vector<int32_t>>>;

template<typename T>
class BaseEltwiseTest : public ::testing::TestWithParam<T> {
public:
    template<typename T1, typename T2>
    VF<float> eltwise_ref(VVVVVVF<T1> input0, VVVVVVF<T2> input1, tensor input0_size, tensor input1_size, eltwise_mode mode) {
        auto out_size = tensor::max(input0_size, input1_size);

        int output_b = out_size.batch[0];
        int output_f = out_size.feature[0];
        int output_w = out_size.spatial[3];
        int output_z = out_size.spatial[2];
        int output_y = out_size.spatial[1];
        int output_x = out_size.spatial[0];

        VVVVVVF<float> output(output_b, VVVVVF<float>(output_f,
                                         VVVVF<float>(output_w,
                                          VVVF<float>(output_z,
                                           VVF<float>(output_y,
                                            VF<float>(output_x))))));

        for (int b = 0; b < output_b; ++b) {
            for (int f = 0; f < output_f; ++f) {
                for (int w = 0; w <output_w; ++w) {
                    for (int z = 0; z <output_z; ++z) {
                        for (int y = 0; y <output_y; ++y) {
                            for (int x = 0; x < output_x; ++x) {
                                int in0_b = b % input0_size.batch[0];
                                int in0_f = f % input0_size.feature[0];
                                int in0_w = w % input0_size.spatial[3];
                                int in0_z = z % input0_size.spatial[2];
                                int in0_y = y % input0_size.spatial[1];
                                int in0_x = x % input0_size.spatial[0];

                                int in1_b = b % input1_size.batch[0];
                                int in1_f = f % input1_size.feature[0];
                                int in1_w = w % input1_size.spatial[3];
                                int in1_z = z % input1_size.spatial[2];
                                int in1_y = y % input1_size.spatial[1];
                                int in1_x = x % input1_size.spatial[0];

                                auto in0 = static_cast<float>(input0[in0_b][in0_f][in0_w][in0_z][in0_y][in0_x]);
                                auto in1 = static_cast<float>(input1[in1_b][in1_f][in1_w][in1_z][in1_y][in1_x]);
                                output[b][f][w][z][y][x] = eltwise_execute<float>(mode, in0, in1);
                            }
                        }
                    }
                }
            }
        }

        return flatten_6d<float>(format::bfwzyx, output);
    }
};

class eltwise_test : public BaseEltwiseTest<eltwise_test_params> {
};

TEST_P(eltwise_test, fsv16) {
    auto p = GetParam();
    tests::random_generator rg(GET_SUITE_NAME);

    ASSERT_EQ(std::get<2>(p).size(), 2);

    auto mode = std::get<0>(p);
    auto dt = std::get<1>(p);
    auto input0_size = std::get<2>(p)[0];
    auto input1_size = std::get<2>(p)[1];

    int b0 = input0_size[0];
    int f0 = input0_size[1];
    int z0 = input0_size.size() == 4 ? 1 : input0_size[2];
    int y0 = input0_size[input0_size.size() == 4 ? 2 : 3];
    int x0 = input0_size[input0_size.size() == 4 ? 3 : 4];

    int b1 = input1_size[0];
    int f1 = input1_size[1];
    int z1 = input1_size.size() == 4 ? 1 : input1_size[2];
    int y1 = input1_size[input1_size.size() == 4 ? 2 : 3];
    int x1 = input1_size[input1_size.size() == 4 ? 3 : 4];

    int min_random = -2, max_random = 2;
    VVVVVVF<float> input1_rnd = rg.generate_random_6d<float>(b0, f0, 1, z0, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = rg.generate_random_6d<float>(b1, f1, 1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    auto& engine = get_test_engine();
    auto fmt_pln = input0_size.size() == 4 ? format::bfyx : format::bfzyx;
    auto fmt_fsv16 = input0_size.size() == 4 ? format::b_fs_yx_fsv16 : format::b_fs_zyx_fsv16;

    auto in0_size = tensor(fmt_pln, input0_size);
    auto in1_size = tensor(fmt_pln, input1_size);

    auto input1 = engine.allocate_memory({ data_types::f32, fmt_pln, in0_size });
    auto input2 = engine.allocate_memory({ data_types::f32, fmt_pln, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), fmt_fsv16, dt));
    topology.add(reorder("reorder2", input_info("input2"), fmt_fsv16, dt));
    topology.add(eltwise("eltwise", { input_info("reorder1"), input_info("reorder2") }, mode));
    topology.add(reorder("out", input_info("eltwise"), fmt_pln, data_types::f32));
    primitive_id out_id = "out";

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}


static std::vector<eltwise_mode> modes = {eltwise_mode::sum, eltwise_mode::prod};
static std::vector<data_types> types = {data_types::f32, data_types::f16};
static std::vector<std::vector<std::vector<int32_t>>> inputs = {
        {{1, 2, 3, 4}, {1, 2, 3, 4}},
        {{1, 16, 8, 2}, {1, 16, 8, 2}},
        {{1, 128, 16, 8}, {1, 1, 16, 8}},
        {{1, 32, 2, 2}, {1, 32, 2, 2}},
        {{8, 32, 4, 5}, {8, 32, 4, 5}},
        {{1, 2, 3, 4}, {1, 2, 1, 1}},
        {{1, 16, 8, 2}, {1, 16, 1, 1}},
        {{1, 32, 2, 2}, {1, 32, 1, 1}},
        {{8, 32, 4, 5}, {1, 32, 1, 1}},
        {{1, 2, 3, 4}, {1, 1, 1, 1}},
        {{1, 16, 8, 2}, {1, 1, 1, 1}},
        {{1, 32, 2, 2}, {1, 1, 1, 1}},
        {{8, 32, 4, 5}, {1, 1, 1, 1}},
        {{1, 2, 1, 1}, {1, 2, 3, 4}},
        {{1, 16, 1, 1}, {1, 16, 8, 2}},
        {{1, 32, 1, 1}, {1, 32, 2, 2}},
        {{1, 32, 1, 1}, {8, 32, 4, 5}},
        {{1, 2, 1, 1}, {1, 1, 3, 1}},
        {{1, 2, 1, 1}, {4, 1, 3, 5}},

        {{1, 16, 8, 2, 4}, {1, 16, 8, 2, 4}},
        {{8, 32, 4, 5, 6}, {1, 32, 1, 1, 1}},
        {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}},
        {{1, 32, 1, 1, 1}, {8, 32, 3, 4, 5}},
};

INSTANTIATE_TEST_SUITE_P(eltwise, eltwise_test,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(types),
                                ::testing::ValuesIn(inputs)
                                ));

class eltwise_test_6d : public eltwise_test {};
TEST_P(eltwise_test_6d, bfwzyx) {
    auto p = GetParam();
    tests::random_generator rg(GET_SUITE_NAME);

    ASSERT_EQ(std::get<2>(p).size(), 2);

    auto mode = std::get<0>(p);
    auto dt = std::get<1>(p);
    auto input0_size = std::get<2>(p)[0];
    auto input1_size = std::get<2>(p)[1];

    int b0 = input0_size[0];
    int f0 = input0_size[1];
    int w0 = input0_size[2];
    int z0 = input0_size[3];
    int y0 = input0_size[4];
    int x0 = input0_size[5];

    int b1 = input1_size[0];
    int f1 = input1_size[1];
    int w1 = input1_size[2];
    int z1 = input1_size[3];
    int y1 = input1_size[4];
    int x1 = input1_size[5];

    int min_random = -2, max_random = 2;
    VVVVVVF<float> input1_rnd = rg.generate_random_6d<float>(b0, f0, w0, z0, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = rg.generate_random_6d<float>(b1, f1, w1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    auto in0_size = tensor(format::bfwzyx, input0_size);
    auto in1_size = tensor(format::bfwzyx, input1_size);

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfwzyx, in0_size });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfwzyx, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), format::bfwzyx, dt));
    topology.add(reorder("reorder2", input_info("input2"), format::bfwzyx, dt));
    topology.add(eltwise("eltwise", { input_info("reorder1"), input_info("reorder2") }, mode));
    topology.add(reorder("out", input_info("eltwise"), format::bfwzyx, data_types::f32));
    primitive_id out_id = "out";

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

static std::vector<std::vector<std::vector<int32_t>>> inputs_6d = {
        {{1, 2, 3, 4, 5, 6},  {1, 2, 3, 4, 5, 6}},
        {{1, 32, 1, 1, 1, 1}, {8, 32, 4, 5, 6, 7}},
        {{1, 32, 1, 1, 1, 7}, {8, 32, 4, 5, 6, 7}},
};

INSTANTIATE_TEST_SUITE_P(eltwise, eltwise_test_6d,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(types),
                                ::testing::ValuesIn(inputs_6d)
                                ));

class eltwise_test_mixed_precision : public eltwise_test {};
TEST_P(eltwise_test_mixed_precision, fsv16) {
    auto p = GetParam();
    tests::random_generator rg(GET_SUITE_NAME);

    ASSERT_EQ(std::get<2>(p).size(), 2);

    auto mode = std::get<0>(p);
    auto input0_dt = data_types::f16;
    auto input1_dt = std::get<1>(p);
    auto input0_size = std::get<2>(p)[0];
    auto input1_size = std::get<2>(p)[1];

    int b0 = input0_size[0];
    int f0 = input0_size[1];
    int z0 = input0_size.size() == 4 ? 1 : input0_size[2];
    int y0 = input0_size[input0_size.size() == 4 ? 2 : 3];
    int x0 = input0_size[input0_size.size() == 4 ? 3 : 4];

    int b1 = input1_size[0];
    int f1 = input1_size[1];
    int z1 = input1_size.size() == 4 ? 1 : input1_size[2];
    int y1 = input1_size[input1_size.size() == 4 ? 2 : 3];
    int x1 = input1_size[input1_size.size() == 4 ? 3 : 4];

    int min_random = input1_dt == data_types::u8 ? 0 : -2;
    int max_random = input1_dt == data_types::u8 ? 4 : 2;
    VVVVVVF<float> input1_rnd = rg.generate_random_6d<float>(b0, f0, 1, z0, y0, x0, min_random, max_random);
    VVVVVVF<int> input2_rnd = rg.generate_random_6d<int>(b1, f1, 1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<int> input2_rnd_vec = flatten_6d<int>(format::bfwzyx, input2_rnd);

    auto& engine = get_test_engine();
    auto fmt_pln = input0_size.size() == 4 ? format::bfyx : format::bfzyx;
    auto fmt_fsv16 = input0_size.size() == 4 ? format::b_fs_yx_fsv16 : format::b_fs_zyx_fsv16;

    auto in0_size = tensor(fmt_pln, input0_size);
    auto in1_size = tensor(fmt_pln, input1_size);

    auto input1 = engine.allocate_memory({ data_types::f32, fmt_pln, in0_size });
    auto input2 = engine.allocate_memory({ data_types::i32, fmt_pln, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), fmt_fsv16, input0_dt));
    topology.add(reorder("reorder2", input_info("input2"), fmt_fsv16, input1_dt));
    topology.add(eltwise("eltwise", { input_info("reorder1"), input_info("reorder2") }, mode));
    topology.add(reorder("out", input_info("eltwise"), fmt_pln, data_types::f32));
    primitive_id out_id = "out";

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

static std::vector<data_types> mixed_types = {data_types::i8, data_types::u8};

INSTANTIATE_TEST_SUITE_P(eltwise, eltwise_test_mixed_precision,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(mixed_types),
                                ::testing::ValuesIn(inputs)
                                ));


struct eltwise_layout_test_params {
    eltwise_mode mode;
    std::vector<int32_t> input0_size;
    std::vector<int32_t> input1_size;
    format input0_format;
    format input1_format;
    std::string selected_kernel_name;
};

#define CASE_ELTWISE_TEST1  eltwise_mode::sum, {1, 2, 1, 1}, {4, 2, 4, 4}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST2  eltwise_mode::sum, {4, 1, 4, 4}, {1, 5, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST3  eltwise_mode::sum, {4, 5, 4, 1}, {4, 1, 4, 1}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST4  eltwise_mode::sum, {4, 2, 4, 4}, {1, 1, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "eltwise_blocked_opt"
#define CASE_ELTWISE_TEST5  eltwise_mode::sum, {1, 2, 1, 1}, {4, 2, 4, 4}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST6  eltwise_mode::sum, {4, 1, 4, 4}, {1, 5, 1, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST7  eltwise_mode::sum, {4, 5, 4, 1}, {4, 1, 4, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST8  eltwise_mode::sum, {4, 2, 4, 4}, {1, 1, 1, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST9  eltwise_mode::eq,  {4, 2, 4, 4}, {1, 1, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST10 eltwise_mode::sum, {4, 8, 1, 1}, {1, 8, 1, 1}, format::b_fs_yx_fsv32, format::bfyx, "eltwise_blocked_opt"
#define CASE_ELTWISE_TEST11 eltwise_mode::sum, {4, 8, 1, 1}, {1, 8, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "eltwise_blocked_opt"
#define CASE_ELTWISE_TEST12 eltwise_mode::sum, {4, 16, 4, 4}, {1, 16, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "eltwise_blocked_opt"
#define CASE_ELTWISE_TEST13 eltwise_mode::sum, {4, 7, 4, 4}, {1, 7, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST14 eltwise_mode::sum, {1, 8, 1, 1}, {4, 8, 1, 1}, format::bfyx, format::b_fs_yx_fsv32, "generic_eltwise_ref"

class eltwise_layout_test : public BaseEltwiseTest<eltwise_layout_test_params> {
public:
    std::string PrintToString(const eltwise_layout_test_params& params) {
        std::string res;
        res += " format1 (" + format::traits(params.input0_format).str + ")";
        res += " format2 (" + format::traits(params.input1_format).str + ")";

        return res;
    }
};

class eltwise_test_mixed_layout : public eltwise_layout_test {};
TEST_P(eltwise_test_mixed_layout, mixed_layout) {
    auto p = GetParam();
    tests::random_generator rg(GET_SUITE_NAME);

    auto mode = p.mode;
    auto input0_size = p.input0_size;
    auto input1_size = p.input1_size;
    auto format0 = p.input0_format;
    auto format1 = p.input1_format;
    auto selected_kernel = p.selected_kernel_name;

    int b0 = input0_size[0];
    int f0 = input0_size[1];
    int y0 = input0_size[2];
    int x0 = input0_size[3];

    int b1 = input1_size[0];
    int f1 = input1_size[1];
    int y1 = input1_size[2];
    int x1 = input1_size[3];

    int min_random = -2, max_random = 2;
    VVVVVVF<float> input1_rnd = rg.generate_random_6d<float>(b0, f0, 1, 1, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = rg.generate_random_6d<float>(b1, f1, 1, 1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    auto& engine = get_test_engine();
    auto in0_size = tensor(format::bfyx, input0_size);
    auto in1_size = tensor(format::bfyx, input1_size);

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, in0_size });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("reorder1", input_info("input1"), format0, data_types::f32));
    topology.add(reorder("reorder2", input_info("input2"), format1, data_types::f32));
    topology.add(eltwise("eltwise", { input_info("reorder1"), input_info("reorder2") }, mode));
    topology.add(reorder("out", input_info("eltwise"), format::bfyx, data_types::f32));
    primitive_id out_id = "out";

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    ASSERT_TRUE(network.get_primitive_info("eltwise").find(selected_kernel) != std::string::npos);

    auto output_memory = outputs.at(out_id).get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(eltwise, eltwise_test_mixed_layout,
                        ::testing::ValuesIn(std::vector<eltwise_layout_test_params>{
                            eltwise_layout_test_params{CASE_ELTWISE_TEST1},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST2},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST3},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST4},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST5},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST6},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST7},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST8},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST9},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST10},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST11},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST12},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST13},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST14},
                        }));

//
struct eltwise_random_test_params {
    data_types  input_type;
    tensor      first_input_size;
    tensor      second_input_size;

    format::type in_format;
    format::type in_format_second;  // For testing 1x1x1x1 bfyx
    format::type out_format;
    eltwise_mode mode;
    impl_types   impl_type;
    bool         is_caching_test;
};

struct eltwise_random_test : testing::TestWithParam<eltwise_random_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static std::string PrintToString(const eltwise_random_test_params& params) {
        std::string res = " data (" + ov::element::Type(params.input_type).get_type_name() + "), ";
        res += " format (" + format::traits(params.in_format).str + ") input1 : ";
        res += params.first_input_size.to_string() + " / input2 : ";
        res += params.second_input_size.to_string() + "\n";

        return res;
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        mem_lock<T> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T> opt_ptr{out_opt, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto opt_out_val = opt_ptr[opt_out_offset];

                        ASSERT_EQ(opt_out_val, ref_out_val);
                        // ASSERT_NEAR(static_cast<float>(opt_out_val), static_cast<float>(ref_out_val), 1.e-1f);
                    }
                }
            }
        }
    }

    void execute_compare(const eltwise_random_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        auto in_layout1 = layout(params.input_type, params.in_format, params.first_input_size);
        auto in_layout2 = layout(params.input_type, params.in_format_second, params.second_input_size);
        bool is_caching_test = params.is_caching_test;
        auto input1 = engine.allocate_memory(in_layout1);
        auto input2 = engine.allocate_memory(in_layout2);
        fill_random(input1);
        fill_random(input2);

        cldnn::topology topo;
        topo.add(input_layout("input1", input1->get_layout()));
        topo.add(input_layout("input2", input2->get_layout()));
        auto prim = eltwise("eltwise", { input_info("input1"), input_info("input2") }, params.mode);
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"eltwise"}));
        if (params.impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"eltwise", {params.in_format, "", params.impl_type}} }));
        else
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"eltwise", {params.in_format, "generic_eltwise_ref"}} }));

        cldnn::network net(engine, topo, config);
        net.set_input_data("input1", input1);
        net.set_input_data("input2", input2);

        auto result = net.execute();
        auto output = result.at("eltwise").get_memory();

        cldnn::topology topo_opt;
        topo_opt.add(input_layout("input1", input1->get_layout()));
        topo_opt.add(input_layout("input2", input2->get_layout()));
        auto prim_opt = eltwise("eltwise_opt", { input_info("input1"), input_info("input2") }, params.mode);
        topo_opt.add(prim_opt);

        ExecutionConfig config_opt = get_test_default_config(engine);
        config_opt.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"eltwise_opt"}));

        cldnn::network::ptr net_opt = get_network(engine, topo_opt, config_opt, get_test_stream_ptr(), is_caching_test);

        net_opt->set_input_data("input1", input1);
        net_opt->set_input_data("input2", input2);

        auto result_opt = net_opt->execute();
        auto output_opt = result_opt.at("eltwise_opt").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.input_type == data_types::f16) {
                compare_outputs<ov::float16>(output, output_opt);
            } else if (params.input_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (params.input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

struct eltwise_random_test_param_generator : std::vector<eltwise_random_test_params> {
    eltwise_random_test_param_generator& add(eltwise_random_test_params params) {
        push_back(params);
        return *this;
    }

    eltwise_random_test_param_generator& broadcast_params(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {1, 1, 48, 64},  {1, 10, 48, 64}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 16, 8, 8},   {1, 1, 8, 8},    input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 36, 8, 16},  {1, 36, 1, 1},   input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 36, 4, 4},   {1, 1, 4, 4},    input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 8, 4, 4},    {1, 1, 1, 1},    input_format, format::bfyx, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 1, 1, 1},    {1, 8, 4, 4},    input_format, format::bfyx, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& simple_params(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {1, 10, 10, 10}, {1, 10, 10, 10}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 20, 10, 10}, {1, 20, 10, 10}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 5, 4, 4},    {1, 5, 4, 4},    input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 32, 16, 16}, {1, 32, 16, 16}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& simple_params_bsv(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {16, 32, 4, 4}, {16, 32, 4, 4}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {20, 16, 4, 1}, {20, 16, 4, 1}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& simple_params_zyx(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {1, 32, 4, 4, 4}, {1, 32, 4, 4, 4}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& broadcast_params_zyx(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {1, 1,  4, 4, 8},  {1, 10, 4, 4, 8}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 36, 8, 8, 16}, {1, 36, 1, 1, 1}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1, 1,  1, 1, 1},  {1, 16, 4, 4, 2}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& simple_params_bsv_zyx(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {48, 32, 4, 4, 4}, {48, 32, 4, 4, 4}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }

    eltwise_random_test_param_generator& broadcast_params_bsv_zyx(data_types type, format::type input_format, format::type output_format, impl_types impl_type = impl_types::any) {
        push_back(eltwise_random_test_params{ type, {32, 1,  4, 4, 8}, {32, 10, 4, 4, 8}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {32, 36, 4, 4, 4}, {32, 36, 1, 1, 1}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        push_back(eltwise_random_test_params{ type, {1,  1,  1, 1, 1}, {32, 16, 4, 4, 4}, input_format, input_format, output_format, eltwise_mode::sum, impl_type, false });
        return *this;
    }
};

TEST_P(eltwise_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true);
}

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_fsv4,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .broadcast_params(data_types::f32, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .broadcast_params(data_types::f16, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .broadcast_params(data_types::i8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .broadcast_params(data_types::u8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .simple_params(data_types::f32, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .simple_params(data_types::f16, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .simple_params(data_types::i8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .simple_params(data_types::u8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_export_import,
                         eltwise_random_test,
                         testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .add(eltwise_random_test_params{ data_types::f16, {1, 1, 48, 64}, {1, 10, 48, 64}, format::b_fs_yx_fsv4,
                                                             format::b_fs_yx_fsv4, format::b_fs_yx_fsv4, eltwise_mode::sum, impl_types::any, true })
                         ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_fsv16,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .broadcast_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .broadcast_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .broadcast_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .broadcast_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_fsv32,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .broadcast_params(data_types::f32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .broadcast_params(data_types::f16, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .broadcast_params(data_types::i8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .broadcast_params(data_types::u8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params(data_types::f32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params(data_types::f16, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params(data_types::i8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params(data_types::u8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_bsv_fsv,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .broadcast_params(data_types::f32, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16)
                            .broadcast_params(data_types::f16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16)
                            .broadcast_params(data_types::i8, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32)
                            .broadcast_params(data_types::u8, format::bs_fs_yx_bsv16_fsv32, format::bs_fs_yx_bsv16_fsv32)
                            .simple_params(data_types::f32, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16)
                            .simple_params(data_types::f16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16)
                            .simple_params(data_types::i8, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32)
                            .simple_params(data_types::u8, format::bs_fs_yx_bsv16_fsv32, format::bs_fs_yx_bsv16_fsv32)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_batch_case_neg,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .simple_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params_bsv(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .simple_params(data_types::u8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                            .simple_params_bsv(data_types::u8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_batch_case_pos,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .simple_params(data_types::f16, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32)
                            .simple_params_bsv(data_types::f16, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32)
                            .simple_params_bsv(data_types::u8, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32)
                            .simple_params_bsv(data_types::f16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_smoke_zyx,
                        eltwise_random_test,
                        testing::ValuesIn(
                            eltwise_random_test_param_generator()
                            .simple_params_zyx(data_types::f16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
                            .simple_params_zyx(data_types::u8, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32)
                            .broadcast_params_zyx(data_types::f16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
                            .broadcast_params_zyx(data_types::u8, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32)
                            .simple_params_bsv_zyx(data_types::f16, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32)
                            .simple_params_bsv_zyx(data_types::u8, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32)
                            .simple_params_bsv_zyx(data_types::f16, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16)
                            .broadcast_params_bsv_zyx(data_types::f16, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32)
                            .broadcast_params_bsv_zyx(data_types::u8, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32)
                            .broadcast_params_bsv_zyx(data_types::f16, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16)
                        ));

INSTANTIATE_TEST_SUITE_P(eltwise_cpu_impl,
                         eltwise_random_test,
                         testing::ValuesIn(
                             eltwise_random_test_param_generator()
                             .broadcast_params(data_types::f32, format::bfyx, format::bfyx, impl_types::cpu)
                             .broadcast_params(data_types::f16, format::bfyx, format::bfyx, impl_types::cpu)
                             .simple_params(data_types::f32, format::bfyx, format::bfyx, impl_types::cpu)
                             .simple_params(data_types::f16, format::bfyx, format::bfyx, impl_types::cpu)
                         ));
