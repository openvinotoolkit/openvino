// Copyright (c) 2019 Intel Corporation
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

#include <api/engine.hpp>
#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/contract.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include "test_utils/test_utils.h"
#include "test_utils/uniform_quantized_real_distribution.hpp"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
T reduce_execute(cldnn::contract_mode mode, T x, T y) {
    switch (mode) {
    case contract_mode::sum:
        return x + y;
    case contract_mode::prod:
        return x * y;
    case contract_mode::all:
        return x && y;
    case contract_mode::any:
        return x || y;
    case contract_mode::max:
        return x > y ? x : y;
    default:
        return (T)0;
    }
}

template <typename T>
VVVVF<T> reduce_dim(VVVVF<T> &input,
    cldnn::contract_mode mode, uint16_t axis,
    int input_padding_y = 0, int input_padding_x = 0,
    int output_padding_y = 0, int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t out_sizes[4];
    out_sizes[0] = input.size();
    out_sizes[1] = input[0].size();
    out_sizes[2] = input[0][0].size() + 2 * padding_y;
    out_sizes[3] = input[0][0][0].size() + 2 * padding_x;
    if (axis == 0)
        out_sizes[0] = 1;
    else
        for (uint16_t i = axis; i > 0; --i)
        {
            out_sizes[i] = out_sizes[i - 1];
            out_sizes[i - 1] = 1;
        }
    VVVVF<T> output(out_sizes[0], VVVF<T>(out_sizes[1], VVF<T>(out_sizes[2], VF<T>(out_sizes[3]))));

    switch (axis) {
    case 0:
        for (size_t f = 0; f < out_sizes[1]; ++f)
            for (size_t y = 0; y < out_sizes[2]; ++y)
                for (size_t x = 0; x < out_sizes[3]; ++x)
                {
                    T res = input[0][f][y][x];
                    size_t orig_b = input.size();
                    for (size_t b = 1; b < orig_b; ++b)
                        res = reduce_execute<T>(mode, res, input[b][f][y][x]);
                    output[0][f][y][x] = res;
                }
        break;
    case 1:
        for (size_t b = 0; b < out_sizes[1]; ++b)
            for (size_t y = 0; y < out_sizes[2]; ++y)
                for (size_t x = 0; x < out_sizes[3]; ++x)
                {
                    T res = input[b][0][y][x];
                    size_t orig_f = input[0].size();
                    for (size_t f = 1; f < orig_f; ++f)
                        res = reduce_execute<T>(mode, res, input[b][f][y][x]);
                    output[0][b][y][x] = res;
                }
        break;
    case 2:
        for (size_t b = 0; b < out_sizes[1]; ++b)
            for (size_t f = 0; f < out_sizes[2]; ++f)
                for (size_t x = 0; x < out_sizes[3]; ++x)
                {
                    T res = input[b][f][0][x];
                    size_t orig_y = input[0][0].size();
                    for (size_t y = 1; y < orig_y; ++y)
                        res = reduce_execute<T>(mode, res, input[b][f][y][x]);
                    output[0][b][f][x] = res;
                }
        break;
    case 3:
        for (size_t b = 0; b < out_sizes[1]; ++b)
            for (size_t f = 0; f < out_sizes[2]; ++f)
                for (size_t y = 0; y < out_sizes[3]; ++y)
                {
                    T res = input[b][f][y][0];
                    size_t orig_x = input[0][0][0].size();
                    for (size_t x = 1; x < orig_x; ++x)
                        res = reduce_execute<T>(mode, res, input[b][f][y][x]);
                    output[0][b][f][y] = res;
                }
        break;
    default: break;
    }
    return output;
}

template <typename T>
VVVVF<T> reduce_input(VVVVF<T> &input,
    cldnn::contract_mode mode, std::vector<uint16_t> reduction_axes,
    int input_padding_y = 0, int input_padding_x = 0,
    int output_padding_y = 0, int output_padding_x = 0) {
    VVVVF<T> output(input);
    for (size_t i = 0; i < reduction_axes.size(); ++i)
        output = reduce_dim<T>(output, mode, reduction_axes[i], input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    return output;
}

std::string print_axes(std::vector<uint16_t> reduction_axes)
{
    std::stringstream res;
    res << "[";
    for (size_t i = 0; i < reduction_axes.size(); ++i)
    {
        if (i != 0)
            res << ", ";
        res << reduction_axes[i];
    }
    res << "]";
    return res.str();
}

template <typename T>
void generic_contract_test_float(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, cldnn::contract_mode mode,
    std::vector<uint16_t> reduction_axes, int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0, int output_padding_x = 0) {

    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);

    const auto& engine = get_test_engine();
    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", mode, reduction_axes));

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    VVVVF<T> output_cpu = reduce_input<T>(input_rnd, mode, reduction_axes, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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
        << "contract_mode = " << (int)mode << std::endl
        << "axes = " << print_axes(reduction_axes) << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;
}

template <typename T>
void generic_contract_test_int(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, cldnn::contract_mode mode,
    std::vector<uint16_t> reduction_axes, int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0, int output_padding_x = 0) {

    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);

    const auto& engine = get_test_engine();
    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", mode, reduction_axes));

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    VVVVF<T> output_cpu = reduce_input<T>(input_rnd, mode, reduction_axes, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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
        << "contract_mode = " << (int)mode << std::endl
        << "axes = " << print_axes(reduction_axes) << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;
}

TEST(contract_gpu_f32, generic_y_sum) {
    generic_contract_test_float<float>(format::bfyx, 5, 5, 5, 5, contract_mode::sum, { 2 });
}

TEST(contract_gpu_f32, generic_fx_prod) {
    generic_contract_test_float<float>(format::bfyx, 5, 5, 5, 5, contract_mode::sum, { 1, 3 });
}

TEST(contract_gpu_i32, generic_f_all) {
    generic_contract_test_int<int32_t>(format::bfyx, 5, 5, 5, 5, contract_mode::all, { 1 });
}

TEST(contract_gpu_i32, generic_bfyx_any) {
    generic_contract_test_int<int32_t>(format::bfyx, 5, 5, 5, 5, contract_mode::any, { 0, 1, 2, 3 });
}

TEST(contract_gpu_f32, generic_f_max) {
    generic_contract_test_float<float>(format::bfyx, 5, 5, 5, 5, contract_mode::max, { 1 });
}

TEST(contract_gpu_i32, generic_f_max) {
    generic_contract_test_int<int32_t>(format::bfyx, 5, 5, 5, 5, contract_mode::max, { 1 });
}

TEST(contract_gpu_i64, generic_f_max) {
    generic_contract_test_int<int64_t>(format::bfyx, 5, 5, 5, 5, contract_mode::max, { 1 });
}

TEST(contract_gpu_i64, generic_x_sum) {
    generic_contract_test_int<int64_t>(format::bfyx, 5, 5, 5, 5, contract_mode::sum, { 3 });
}

TEST(contract_gpu_i64, generic_fy_any) {
    generic_contract_test_int<int64_t>(format::bfyx, 5, 5, 5, 5, contract_mode::any, { 1, 2 });
}

TEST(contract_error, basic_error_empty_r_axes) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", contract_mode::sum, { }));

    std::string msg_to_find = "Incorrect parameters configuration: reduction_axes should not be empty.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(contract_error, basic_error_wrong_r_axes_size) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", contract_mode::sum, { 0, 1, 2, 3, 4 }));

    std::string msg_to_find = "Incorrect parameters configuration: reduction_axes size should be less or equal 4.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(contract_error, basic_error_wrong_r_axis_value) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", contract_mode::sum, { 0, 4 }));

    std::string msg_to_find = "Incorrect parameters configuration: reduction_axes index should be within reduction_axes range.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(contract_error, basic_error_duplicate_r_axis_values) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(contract("output", "input", contract_mode::sum, { 0, 1, 1 }));

    std::string msg_to_find = "Incorrect parameters configuration: Duplicate axes numbers was found in reduction_axes.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}
