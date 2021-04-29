// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/eltwise.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/reorder.hpp>
#include <api/data.hpp>
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

    const auto& engine = get_test_engine();
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input1 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto input2 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", input1.get_layout().with_padding(padding{{ 0, 0, input_padding_x, input_padding_y }, 0 })));
    topology.add(eltwise("eltwise", {"reorder1", "input2"}, mode, padding{ { 0, 0, output_padding_x, output_padding_y }, 0 }));
    primitive_id out_id = "eltwise";
    if (relu)
    {
        topology.add(activation("out", out_id, activation_func::relu, { slope, 0.0f }));
        out_id = "out";
    }
    network network(engine, topology);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::eq));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 0, 1, 0, 1,
                                    0, 0, 1, 0,
                                    0, 0, 0, 0,
                                    0, 1, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::ne));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 0, 1, 0,
                                    1, 1, 0, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::lt));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    1, 1, 0, 0,
                                    1, 1, 1, 0,
                                    0, 0, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::le));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 0, 1, 0, 1,
                                    1, 1, 1, 0,
                                    1, 1, 1, 0,
                                    0, 1, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::gt));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 0, 1, 0,
                                    0, 0, 0, 1,
                                    0, 0, 0, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::ge));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    0, 0, 1, 1,
                                    0, 0, 0, 1,
                                    1, 1, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::logic_and));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 0, 1, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::logic_and));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::logic_or));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 0, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::logic_or));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
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

    const auto& engine = get_test_engine();

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

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
    topology.add(input_layout("input", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::logic_xor));

    network network(engine, topology);

    network.set_input_data("input", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    std::vector<int8_t> answers = { 0, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0 };

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }
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

    const auto& engine = get_test_engine();

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

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_channel) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          2.5f, 2.5f,

                          6.f,   7.f,
                          4.f,   4.f,

                          18.f, 17.5f,
                          12.f,  9.5f,

                          1.f,   7.f,
                          3.5f, -3.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_x) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 3.5f,
                          0.5f, 2.5f,

                          7.f,   9.f,
                          2.f,   4.f,

                          18.f, 20.f,
                          8.5f,  10.5f,

                          2.f,   10.5f,
                          -1.f, -3.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_y) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          2.5f, 2.5f,

                          8.f,   7.5f,
                          6.f,   3.5f,

                          16.f,  17.f,
                          10.f,  10.f,

                          1.f,   7.f,
                          3.5f, -3.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_batch) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 2.5f,
                          1.5f, 2.5f,

                          7.f,   7.f,
                          4.f,   4.f,

                          18.f,  17.5f,
                          11.f,  10.5f,

                          2.f,   6.f,
                          3.5f, -3.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_in2x2x2x2_broadcast_multiple_dims) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.5f, 3.5f,
                          1.5f, 3.5f,

                          7.f,   9.f,
                          4.f,   6.f,

                          16.f, 18.f,
                          9.f,  11.f,

                          0.f,   8.5f,
                          1.5f, -0.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, pow_in2x2x2x2_broadcast_all) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::pow));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 1.f, 4.f,
                          9.f, 16.f,

                          25.f, 36.f,
                          49.f, 64.f,

                          81.f,  100.f,
                          121.f, 144.f,

                          169.f, 196.f,
                          225.f, 256.f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_in2x2x2x2_broadcast_2_inputs_same_dim) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.5f, 4.5f,
                          4.5f, 4.5f,

                          8.f,   9.f,
                          5.f,   6.f,

                          16.f,  19.f,
                          9.f,  12.f,

                          -2.f,  6.5f,
                          -0.5f, -2.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, add_basic_in2x2x2x2_broadcast_2_inputs_diff_dim) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::sum));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = { 4.5f, 4.5f,
                          2.5f, 4.5f,

                          10.f,  9.f,
                          5.f,   6.f,

                          16.f,  19.f,
                          7.f,  11.f,

                           0.f,   7.5f,
                          -0.5f, -2.5f };

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
    const auto& engine = get_test_engine();

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

    const auto& engine = get_test_engine();
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
    std::vector<eltwise_mode> eltwise_ops_to_test = { eltwise_mode::sum, eltwise_mode::sub, eltwise_mode::div, eltwise_mode::prod, eltwise_mode::min, eltwise_mode::max, eltwise_mode::mod };

    for (auto& data_type : data_types_to_test)
    {
        for (auto& mode : eltwise_ops_to_test)
        {
            const auto& engine = get_test_engine();
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
                else if (mode == eltwise_mode::mod) {
                    expected = std::fmod(input_1_vec[i], input_2_vec[i]);
                }

                EXPECT_TRUE(are_equal(std::floor(expected), output_ptr[i]));
            }
        }
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
            const auto& engine = get_test_engine();
            auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
            auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

            topology topology;
            topology.add(input_layout("input", input.get_layout()));
            topology.add(input_layout("input2", input2.get_layout()));
            topology.add(reorder("input_reorder", "input", { data_type, format::yxfb,{ 2, 2, 2, 2 } }));
            topology.add(eltwise("eltwise", { "input_reorder", "input2" }, mode));
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
                else if (mode == eltwise_mode::mod)
                    expected = std::fmod(input_1_vec[i], input_2_vec[i]);

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

    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder2", "input2", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2"}, eltwise_mode::sum, {0.5f, 0.5f}, data_types::f32));

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
    const auto& engine = get_test_engine();

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

    EXPECT_THROW(topology.add(eltwise("eltwise1", {"input", "input2"}, eltwise_mode::sum, coeffs1, data_types::f32)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise2", {"input", "input2"}, eltwise_mode::sum, coeffs3, data_types::f32)), std::invalid_argument);

    EXPECT_THROW(topology.add(eltwise("eltwise3", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs1, data_types::f32)), std::invalid_argument);
    EXPECT_THROW(topology.add(eltwise("eltwise4", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs2, data_types::f32)), std::invalid_argument);

    EXPECT_NO_THROW(topology.add(eltwise("eltwise5", {"input", "input2"}, eltwise_mode::sum, coeffs0, data_types::f32)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise6", {"input", "input2"}, eltwise_mode::sum, coeffs2, data_types::f32)));

    EXPECT_NO_THROW(topology.add(eltwise("eltwise7", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs0, data_types::f32)));
    EXPECT_NO_THROW(topology.add(eltwise("eltwise8", {"input", "input2", "input3"}, eltwise_mode::sum, coeffs3, data_types::f32)));
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(eltwise("eltwise", {"input", "input2", "input3"}, eltwise_mode::sum, {0.5f, 0.5f, 0.5f}, data_types::f32));

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
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder2", "input2", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
    topology.add(reorder("reorder3", "input3", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })));
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
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 4, 4 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", "input", "input2", { {0,0,1,1}, {0,0,2,2} }, eltwise_mode::max));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1,  17,  33,  49,
        3,  19,  35,  51,
        9,  25,  41,  57,
        11, 27,  43,  59 };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", { "input", "input2" }, eltwise_mode::sum));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output = outputs.at("eltwise").get_memory();

    float answers[16] = {
        1.5f,   2.5f,   5.5f,   4.f,
        2.5f,   2.5f,   6.5f,   7.7f,
        3.5f,    3.f,   7.5f,  14.5f,
        4.5f,    2.f,   8.5f,  10.5f };

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
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
        FLOAT16(1111),FLOAT16(1112),FLOAT16(1121),FLOAT16(1122),
        FLOAT16(1211),FLOAT16(1212),FLOAT16(1221),FLOAT16(1222),
        FLOAT16(2111),FLOAT16(2112),FLOAT16(2121),FLOAT16(2122),
        FLOAT16(2211),FLOAT16(2212),FLOAT16(2221),FLOAT16(2222)
    };

    const auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });
    auto input2 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_2x2x2x2_input);
    set_values(input2, fp16_bfyx_2x2x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1.get_layout()));
    golden_topology.add(input_layout("input2", input2.get_layout()));
    golden_topology.add(eltwise("eltwise", "input1", "input2", eltwise_mode::sum));

    network golden_network(engine, golden_topology);
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    auto golden_ptr = golden_output.pointer<FLOAT16>();
    // GOLDEN BFYX ELTWISE - END
    // FS_B_YX_FSV32 ELTWISE
    topology FSV32_topology;
    FSV32_topology.add(input_layout("input1", input1.get_layout()));
    FSV32_topology.add(input_layout("input2", input2.get_layout()));
    FSV32_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FSV32_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FSV32_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum));
    FSV32_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor)));

    network FSV32_network(engine, FSV32_topology);
    FSV32_network.set_input_data("input1", input1);
    FSV32_network.set_input_data("input2", input2);

    auto FSV32_outputs = FSV32_network.execute();
    auto FSV32_output = FSV32_outputs.at("reorderOutput").get_memory();
    auto FSV32_ptr = FSV32_output.pointer<FLOAT16>();
    // FS_B_YX_FSV32 ELTWISE - END

    ASSERT_EQ(golden_ptr.size(), FSV32_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(FSV32_ptr[i]));
    }
}

TEST(eltwise_gpu_f32, broadcast_test_in4x4x2x2x2) {
    //  Input  : 2x2x2x2x1
    //  Input2 : 2x2x1x1x2
    //  Output : 2x2x2x2x2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 2, 2, 2, 2, 1 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 2, 2, 1, 1, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(eltwise("eltwise", { "input", "input2" }, eltwise_mode::sum));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, { 0.5f, 2.5f, 0.5f, 2.5f, 1.f, 2.f, 3.f, 4.f });

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

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

    auto output_ptr = output.pointer<float>();

    for (int i = 0; i < 32; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_basic)
{
    // Inputs are 32x96x2x2

    tensor input_tensor(32, 96, 20, 20);
    VVVVF<FLOAT16> input_rnd = generate_random_4d<FLOAT16>(32, 96, 20, 20, 1, 3);
    VF<FLOAT16> fp16_bfyx_32x96x2x2_input = flatten_4d<FLOAT16>(format::bfyx, input_rnd);

    const auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });
    auto input2 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1.get_layout()));
    golden_topology.add(input_layout("input2", input2.get_layout()));
    golden_topology.add(eltwise("eltwise", "input1", "input2", eltwise_mode::sum));

    network golden_network(engine, golden_topology);
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    auto golden_ptr = golden_output.pointer<FLOAT16>();
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::byxf, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor)));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto FS_B_YX_FSV32_OUTPUT_ptr = FS_B_YX_FSV32_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::byxf, input_tensor)));
    BYXF_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    BYXF_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum));
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor)));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology);
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto BYXF_OUTPUT_ptr = BYXF_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_output_padding)
{
    // Inputs are 32x96x2x2

    tensor input_tensor(32, 96, 20, 20);
    VVVVF<FLOAT16> input_rnd = generate_random_4d<FLOAT16>(32, 96, 20, 20, 1, 3);
    VF<FLOAT16> fp16_bfyx_32x96x2x2_input = flatten_4d<FLOAT16>(format::bfyx, input_rnd);

    const auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });
    auto input2 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1.get_layout()));
    golden_topology.add(input_layout("input2", input2.get_layout()));
    golden_topology.add(eltwise("eltwise", "input1", "input2", eltwise_mode::sum, padding{ {0,0,5,10} , 0 }));

    network golden_network(engine, golden_topology);
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    auto golden_ptr = golden_output.pointer<FLOAT16>();
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::byxf, input_tensor)));
    FS_B_YX_FSV32_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum, padding{ {0,0,5,10} , 0 }));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor,
                                              padding{ {0,0,5,10} , 0 })));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto FS_B_YX_FSV32_OUTPUT_ptr = FS_B_YX_FSV32_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::byxf, input_tensor)));
    BYXF_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor)));
    BYXF_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum, padding{ {0,0,5,10} , 0 }));
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor,
                                     padding{ {0,0,5,10} , 0 })));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology);
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto BYXF_OUTPUT_ptr = BYXF_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

TEST(eltwise_gpu_f16, bfyx_and_fs_b_yx_fsv32_input_padding)
{
    // Inputs are 32x96x20x20

    tensor input_tensor(32, 96, 20, 20);
    VVVVF<FLOAT16> input_rnd = generate_random_4d<FLOAT16>(32, 96, 20, 20, 1, 3);
    VF<FLOAT16> fp16_bfyx_32x96x2x2_input = flatten_4d<FLOAT16>(format::bfyx, input_rnd);

    const auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
        return;
    }

    auto input1 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });
    auto input2 = memory::allocate(engine, { data_types::f16,format::bfyx, input_tensor });

    set_values(input1, fp16_bfyx_32x96x2x2_input);
    set_values(input2, fp16_bfyx_32x96x2x2_input);

    // GOLDEN BFYX ELTWISE
    topology golden_topology;
    golden_topology.add(input_layout("input1", input1.get_layout()));
    golden_topology.add(input_layout("input2", input2.get_layout()));
    golden_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::bfyx, input_tensor, padding{ {0,0,10,15},0.0f })));
    golden_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::bfyx, input_tensor, padding{ {0,0,5,7},0.0f })));
    golden_topology.add(eltwise("eltwise", "input1", "input2", eltwise_mode::sum));

    network golden_network(engine, golden_topology);
    golden_network.set_input_data("input1", input1);
    golden_network.set_input_data("input2", input2);

    auto golden_outputs = golden_network.execute();
    auto golden_output = golden_outputs.at("eltwise").get_memory();
    auto golden_ptr = golden_output.pointer<FLOAT16>();
    // GOLDEN BFYX ELTWISE - END
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT
    topology FS_B_YX_FSV32_OUTPUT_topology;
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ {0,0,10,15},0.0f })));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::byxf, input_tensor, padding{ {0,0,5,7},0.0f })));
    FS_B_YX_FSV32_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum));
    FS_B_YX_FSV32_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor)));

    network FS_B_YX_FSV32_OUTPUT_network(engine, FS_B_YX_FSV32_OUTPUT_topology);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input1", input1);
    FS_B_YX_FSV32_OUTPUT_network.set_input_data("input2", input2);

    auto FS_B_YX_FSV32_OUTPUT_outputs = FS_B_YX_FSV32_OUTPUT_network.execute();
    auto FS_B_YX_FSV32_OUTPUT_output = FS_B_YX_FSV32_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto FS_B_YX_FSV32_OUTPUT_ptr = FS_B_YX_FSV32_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, FS_B_YX_FSV32 OUTPUT - END
    // MIXED INPUT, BYXF OUTPUT
    topology BYXF_OUTPUT_topology;
    BYXF_OUTPUT_topology.add(input_layout("input1", input1.get_layout()));
    BYXF_OUTPUT_topology.add(input_layout("input2", input2.get_layout()));
    BYXF_OUTPUT_topology.add(reorder("reorder1", "input1", layout(data_types::f16, format::byxf, input_tensor, padding{ {0,0,10,15},0.0f })));
    BYXF_OUTPUT_topology.add(reorder("reorder2", "input2", layout(data_types::f16, format::fs_b_yx_fsv32, input_tensor, padding{ {0,0,5,7},0.0f })));
    BYXF_OUTPUT_topology.add(eltwise("eltwise", "reorder1", "reorder2", eltwise_mode::sum));
    BYXF_OUTPUT_topology.add(reorder("reorderOutput", "eltwise", layout(data_types::f16, format::bfyx, input_tensor)));

    network BYXF_OUTPUT_network(engine, BYXF_OUTPUT_topology);
    BYXF_OUTPUT_network.set_input_data("input1", input1);
    BYXF_OUTPUT_network.set_input_data("input2", input2);

    auto BYXF_OUTPUT_outputs = BYXF_OUTPUT_network.execute();
    auto BYXF_OUTPUT_output = BYXF_OUTPUT_outputs.at("reorderOutput").get_memory();
    auto BYXF_OUTPUT_ptr = BYXF_OUTPUT_output.pointer<FLOAT16>();
    // MIXED INPUT, BYXF OUTPUT - END

    ASSERT_EQ(golden_ptr.size(), FS_B_YX_FSV32_OUTPUT_ptr.size());
    ASSERT_EQ(golden_ptr.size(), BYXF_OUTPUT_ptr.size());

    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(FS_B_YX_FSV32_OUTPUT_ptr[i]));
    }
    for (size_t i = 0; i < golden_ptr.size(); i++)
    {
        EXPECT_EQ(float(golden_ptr[i]), float(BYXF_OUTPUT_ptr[i]));
    }
}

template <typename T>
int8_t eltwise_bool_execute(cldnn::eltwise_mode mode, T x, T y) {
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
    default:
        return (int8_t)0;
    }
}

template <typename T>
VVVVF<int8_t> eltwise_bool_reference(VVVVF<T> &input1, VVVVF<T> &input2,
    cldnn::eltwise_mode mode, int input_padding_y = 0,
    int input_padding_x = 0, int output_padding_y = 0,
    int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t output_b = input1.size();
    size_t output_f = input1[0].size();
    size_t output_y = input1[0][0].size() + 2 * padding_y;
    size_t output_x = input1[0][0][0].size() + 2 * padding_x;
    VVVVF<int8_t> output(output_b, VVVF<int8_t>(output_f, VVF<int8_t>(output_y, VF<int8_t>(output_x))));

    T res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t f = 0; f < output_f; ++f) {
            for (size_t y = 0; y < input1[0][0].size(); ++y) {
                for (size_t x = 0; x < input1[0][0][0].size(); ++x) {
                    res = eltwise_bool_execute<T>(mode, input1[b][f][y][x], input2[b][f][y][x]);
                    output[b][f][y + padding_y][x + padding_x] = res;
                }
            }
        }
    }
    return output;
}

template <typename T>
void generic_eltwise_bool_test(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, cldnn::eltwise_mode mode,
    int input_padding_y, int input_padding_x, int output_padding_y, int output_padding_x) {

    int min_random = -2, max_random = 2;
    VVVVF<T> input1_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VVVVF<T> input2_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input1_rnd_vec = flatten_4d<T>(test_input_fmt, input1_rnd);
    VF<T> input2_rnd_vec = flatten_4d<T>(test_input_fmt, input2_rnd);

    const auto& engine = get_test_engine();
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input1 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto input2 = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", input1.get_layout().with_padding(padding{{ 0, 0, input_padding_x, input_padding_y }, 0 })));
    topology.add(eltwise("eltwise", {"reorder1", "input2"}, mode, padding{ { 0, 0, output_padding_x, output_padding_y }, 0 }));

    network network(engine, topology);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "eltwise");

    auto output_memory = outputs.at("eltwise").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<int8_t>();

    VVVVF<int8_t> output_cpu = eltwise_bool_reference<T>(input1_rnd, input2_rnd, mode, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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
    VF<int8_t> output_cpu_vec = flatten_4d<int8_t>(test_input_fmt, output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
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

    generic_eltwise_bool_test<int32_t>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, 0, 0, 0, 0);
    generic_eltwise_bool_test<int8_t>(test_inputs_fmt, 1, 1, input_size.first, input_size.second, mode, 0, 0, 0, 0);
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

void run_eltwise_generic_test(cldnn::eltwise_mode mode)
{
    cldnn::format test_inputs_fmt = cldnn::format::bfyx;
    std::pair<int, int> input_size = { 227, 227 };

    const auto& engine = get_test_engine();
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


TEST(eltwise_gpu, b_fs_yx_fsv4_wo_callib) {
    //
    // Input data
    const int BATCH = 1;
    const int in_B = BATCH;

    const auto& engine = get_test_engine();

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
    auto input1 = memory::allocate(engine,
                                   { data_types::i8,
                                       format::bfyx,
                                       { in_B, in_F, in_X, in_Y } });
    std::vector<char> data_i1(DataGold);
    for (size_t i = 0; i < data_i1.size(); i++) data_i1[i] = data_i1[i] + 1;
    set_values(input1, std::move(data_i1));

    auto input2 = memory::allocate(engine,
                                  { data_types::i8,
                                      format::bfyx,
                                      { in_B, in_F, in_X, in_Y } });
    std::vector<char> data_i2(DataGold);
    for (size_t i = 0; i < data_i2.size(); i++) data_i2[i] = data_i2[i] + 2;
    set_values(input2, std::move(data_i2));

    auto input3 = memory::allocate(engine,
                                  { data_types::i8,
                                      format::bfyx,
                                      { in_B, in_F, in_X, in_Y } });
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
                                { "input1", "input2", "input3" },
                                mode[i]);
            auto actv = activation("eltw_GOLD", eltw, activation_func::relu);

            // Create a topology
            topology.add(input_layout("input1", input1.get_layout()),
                         input_layout("input2", input2.get_layout()),
                         input_layout("input3", input3.get_layout()),
                         eltw, actv);

            // Network processing
            network network(engine, topology);
            network.set_input_data("input1", input1);
            network.set_input_data("input2", input2);
            network.set_input_data("input3", input3);
            auto outputs = network.execute();

            // Validation
            auto searchC = outputs.find("eltw_GOLD");
            EXPECT_NE(searchC, outputs.end());
            auto output = outputs.begin()->second.get_memory();
            auto output_ptr = output.pointer<char>();
            vGoldOutput.reserve(output_ptr.size());
            for (size_t i = 0; i < output_ptr.size(); i++)
                vGoldOutput.push_back(output_ptr[i]);
        }

        // "IMAD" Eltwise
        {
            topology topology;

            // Reorder (a-ka swizzelling) input to MMAD/IMAD Pooling format
            topology.add(reorder("reorder1_Swizzelled",
                                 "input1",
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })),
                         reorder("reorder2_Swizzelled",
                                 "input2",
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })),
                         reorder("reorder3_Swizzelled",
                                 "input3",
                                 layout(data_types::i8,
                                        format::b_fs_yx_fsv4,
                                        { in_B, in_F, in_X, in_Y })));

            auto eltw = eltwise("eltw_IMAD_no_relu",
                                { "reorder1_Swizzelled",
                                  "reorder2_Swizzelled",
                                  "reorder3_Swizzelled" },
                                mode[i]);
            auto actv = activation("eltw_IMAD", eltw, activation_func::relu);
            topology.add(input_layout("input1", input1.get_layout()),
                         input_layout("input2", input2.get_layout()),
                         input_layout("input3", input3.get_layout()),
                         eltw, actv);

            // Back reordering (a-ka unswizzelling) output from MMAD/IMAD pooling
            topology.add(reorder("reorder_UnSwizzelled",
                                 "eltw_IMAD",
                                 layout(data_types::i8,
                                     format::bfyx,
                                     { in_B, in_F, in_X, in_Y })));

            // Network processing
            network network(engine, topology);
            network.set_input_data("input1", input1);
            network.set_input_data("input2", input2);
            network.set_input_data("input3", input3);
            auto outputs = network.execute();

            // Validation
            auto searchC = outputs.find("reorder_UnSwizzelled");
            EXPECT_NE(searchC, outputs.end());
            auto output = outputs.begin()->second.get_memory();
            auto output_ptr = output.pointer<char>();
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

    const auto& engine = get_test_engine();
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
    VVVVVVF<float> input1_rnd = generate_random_6d<float>(b0, f0, 1, z0, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = generate_random_6d<float>(b1, f1, 1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    const auto& engine = get_test_engine();
    auto fmt_pln = input0_size.size() == 4 ? format::bfyx : format::bfzyx;
    auto fmt_fsv16 = input0_size.size() == 4 ? format::b_fs_yx_fsv16 : format::b_fs_zyx_fsv16;

    auto in0_size = tensor(fmt_pln, input0_size);
    auto in1_size = tensor(fmt_pln, input1_size);

    auto input1 = memory::allocate(engine, { data_types::f32, fmt_pln, in0_size });
    auto input2 = memory::allocate(engine, { data_types::f32, fmt_pln, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", fmt_fsv16, dt));
    topology.add(reorder("reorder2", "input2", fmt_fsv16, dt));
    topology.add(eltwise("eltwise", {"reorder1", "reorder2"}, mode));
    topology.add(reorder("out", "eltwise", fmt_pln, data_types::f32));
    primitive_id out_id = "out";

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_ptr = output_memory.pointer<float>();

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        EXPECT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
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

INSTANTIATE_TEST_CASE_P(eltwise, eltwise_test,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(types),
                                ::testing::ValuesIn(inputs)
                                ), );

class eltwise_test_6d : public eltwise_test {};
TEST_P(eltwise_test_6d, bfwzyx) {
    auto p = GetParam();

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
    VVVVVVF<float> input1_rnd = generate_random_6d<float>(b0, f0, w0, z0, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = generate_random_6d<float>(b1, f1, w1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    auto in0_size = tensor(format::bfwzyx, input0_size);
    auto in1_size = tensor(format::bfwzyx, input1_size);

    const auto& engine = get_test_engine();
    auto input1 = memory::allocate(engine, { data_types::f32, format::bfwzyx, in0_size });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfwzyx, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", format::bfwzyx, dt));
    topology.add(reorder("reorder2", "input2", format::bfwzyx, dt));
    topology.add(eltwise("eltwise", {"reorder1", "reorder2"}, mode));
    topology.add(reorder("out", "eltwise", format::bfwzyx, data_types::f32));
    primitive_id out_id = "out";

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_ptr = output_memory.pointer<float>();

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        EXPECT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

static std::vector<std::vector<std::vector<int32_t>>> inputs_6d = {
        {{1, 2, 3, 4, 5, 6},  {1, 2, 3, 4, 5, 6}},
        {{1, 32, 1, 1, 1, 1}, {8, 32, 4, 5, 6, 7}},
        {{1, 32, 1, 1, 1, 7}, {8, 32, 4, 5, 6, 7}},
};

INSTANTIATE_TEST_CASE_P(eltwise, eltwise_test_6d,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(types),
                                ::testing::ValuesIn(inputs_6d)
                                ), );

class eltwise_test_mixed_precision : public eltwise_test {};
TEST_P(eltwise_test_mixed_precision, fsv16) {
    auto p = GetParam();

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
    VVVVVVF<float> input1_rnd = generate_random_6d<float>(b0, f0, 1, z0, y0, x0, min_random, max_random);
    VVVVVVF<int> input2_rnd = generate_random_6d<int>(b1, f1, 1, z1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<int> input2_rnd_vec = flatten_6d<int>(format::bfwzyx, input2_rnd);

    const auto& engine = get_test_engine();
    auto fmt_pln = input0_size.size() == 4 ? format::bfyx : format::bfzyx;
    auto fmt_fsv16 = input0_size.size() == 4 ? format::b_fs_yx_fsv16 : format::b_fs_zyx_fsv16;

    auto in0_size = tensor(fmt_pln, input0_size);
    auto in1_size = tensor(fmt_pln, input1_size);

    auto input1 = memory::allocate(engine, { data_types::f32, fmt_pln, in0_size });
    auto input2 = memory::allocate(engine, { data_types::i32, fmt_pln, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", fmt_fsv16, input0_dt));
    topology.add(reorder("reorder2", "input2", fmt_fsv16, input1_dt));
    topology.add(eltwise("eltwise", {"reorder1", "reorder2"}, mode));
    topology.add(reorder("out", "eltwise", fmt_pln, data_types::f32));
    primitive_id out_id = "out";

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_ptr = output_memory.pointer<float>();

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        EXPECT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

static std::vector<data_types> mixed_types = {data_types::i8, data_types::u8};

INSTANTIATE_TEST_CASE_P(eltwise, eltwise_test_mixed_precision,
                        ::testing::Combine(
                                ::testing::ValuesIn(modes),
                                ::testing::ValuesIn(mixed_types),
                                ::testing::ValuesIn(inputs)
                                ), );


struct eltwise_layout_test_params {
    eltwise_mode mode;
    std::vector<int32_t> input0_size;
    std::vector<int32_t> input1_size;
    format input0_format;
    format input1_format;
    std::string selected_kernel_name;
};

#define CASE_ELTWISE_TEST1  eltwise_mode::sum, {1, 2, 1, 1}, {4, 2, 4, 4}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST2  eltwise_mode::sum, {4, 1, 4, 4}, {1, 5, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "eltwise_b_fs_yx_fsv16"
#define CASE_ELTWISE_TEST3  eltwise_mode::sum, {4, 5, 4, 1}, {4, 1, 4, 1}, format::b_fs_yx_fsv16, format::bfyx, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST4  eltwise_mode::sum, {4, 2, 4, 4}, {1, 1, 1, 1}, format::b_fs_yx_fsv16, format::bfyx, "eltwise_b_fs_yx_fsv16"
#define CASE_ELTWISE_TEST5  eltwise_mode::sum, {1, 2, 1, 1}, {4, 2, 4, 4}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST6  eltwise_mode::sum, {4, 1, 4, 4}, {1, 5, 1, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST7  eltwise_mode::sum, {4, 5, 4, 1}, {4, 1, 4, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"
#define CASE_ELTWISE_TEST8  eltwise_mode::sum, {4, 2, 4, 4}, {1, 1, 1, 1}, format::bfyx, format::b_fs_yx_fsv16, "generic_eltwise_ref"

class eltwise_layout_test : public BaseEltwiseTest<eltwise_layout_test_params> {
};

class eltwise_test_mixed_layout : public eltwise_layout_test {};
TEST_P(eltwise_test_mixed_layout, mixed_layout) {
    auto p = GetParam();

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
    VVVVVVF<float> input1_rnd = generate_random_6d<float>(b0, f0, 1, 1, y0, x0, min_random, max_random);
    VVVVVVF<float> input2_rnd = generate_random_6d<float>(b1, f1, 1, 1, y1, x1, min_random, max_random);
    VF<float> input1_rnd_vec = flatten_6d<float>(format::bfwzyx, input1_rnd);
    VF<float> input2_rnd_vec = flatten_6d<float>(format::bfwzyx, input2_rnd);

    const auto& engine = get_test_engine();
    auto in0_size = tensor(format::bfyx, input0_size);
    auto in1_size = tensor(format::bfyx, input1_size);

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, in0_size });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, in1_size });
    set_values(input1, input1_rnd_vec);
    set_values(input2, input2_rnd_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("reorder1", "input1", format0, data_types::f32));
    topology.add(reorder("reorder2", "input2", format1, data_types::f32));
    topology.add(eltwise("eltwise", {"reorder1", "reorder2"}, mode));
    topology.add(reorder("out", "eltwise", format::bfyx, data_types::f32));
    primitive_id out_id = "out";

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    EXPECT_TRUE(network.get_primitive_info("eltwise").find(selected_kernel) != std::string::npos);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_ptr = output_memory.pointer<float>();

    VF<float> output_cpu_vec = eltwise_ref(input1_rnd, input2_rnd, in0_size, in1_size, mode);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        EXPECT_TRUE(!(std::isnan((float)output_cpu_vec[i]) && std::isnan((float)output_ptr[i])));
        ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

INSTANTIATE_TEST_CASE_P(eltwise, eltwise_test_mixed_layout,
                        ::testing::ValuesIn(std::vector<eltwise_layout_test_params>{
                            eltwise_layout_test_params{CASE_ELTWISE_TEST1},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST2},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST3},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST4},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST5},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST6},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST7},
                            eltwise_layout_test_params{CASE_ELTWISE_TEST8},
                        }), );
