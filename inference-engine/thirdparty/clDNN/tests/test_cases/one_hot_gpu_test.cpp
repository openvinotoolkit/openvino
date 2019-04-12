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

#include <api/CPP/engine.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/one_hot.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include "test_utils/test_utils.h"
#include "test_utils/uniform_quantized_real_distribution.hpp"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
VVVVF<T> one_hot_cpu(VVVVF<T> &input, uint16_t axis,
    int32_t one_hot_limit, int input_padding_y = 0,
    int input_padding_x = 0, int output_padding_y = 0,
    int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t out_sizes[4];
    out_sizes[0] = input.size();
    out_sizes[1] = input[0].size();
    out_sizes[2] = input[0][0].size() + 2 * padding_y;
    out_sizes[3] = input[0][0][0].size() + 2 * padding_x;
    for (uint16_t i = 0; i < axis; ++i)
        out_sizes[i] = out_sizes[i + 1];
    out_sizes[axis] = one_hot_limit;
    VVVVF<T> output(out_sizes[0], VVVF<T>(out_sizes[1], VVF<T>(out_sizes[2], VF<T>(out_sizes[3]))));

    switch (axis) {
    case 0:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[0][f][y][x] == (T)b ? 1 : 0;
        break;
    case 1:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[0][b][y][x] == (T)f ? 1 : 0;
        break;
    case 2:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[0][b][f][x] == (T)y ? 1 : 0;
        break;
    case 3:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[0][b][f][y] == (T)x ? 1 : 0;
        break;
    default: break;
    }
    return output;
}

template <typename T>
void generic_one_hot_test_int(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, tensor shape,
    uint16_t one_hot_axis, int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0, int output_padding_x = 0) {
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
        shape.spatial[1], shape.spatial[0] };
    int32_t one_hot_limit = output_dims[one_hot_axis];

    int min_random = -2, max_random = one_hot_limit + 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);

    const auto& engine = get_test_engine();
    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(one_hot("output", "input", shape, one_hot_axis));

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    VVVVF<T> output_cpu = one_hot_cpu<T>(input_rnd, one_hot_axis, one_hot_limit, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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
        << "one_hot_limit = " << one_hot_limit << std::endl
        << "one_hot_axis = " << one_hot_axis << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;
}

TEST(one_hot_gpu_i32, generic_y_in10_oh5) {
    generic_one_hot_test_int<int32_t>(format::bfyx, 1, 10, 10, 10, tensor(10, 10, 10, 5), 2);
}


TEST(one_hot_error, basic_error_wrong_batch_size) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::i32, format::bfyx, { 10, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(one_hot("output", "input", tensor(10, 1, 1, 50), 2));

    std::string msg_to_find = "Incorrect parameters configuration: input batch size should be equal to 1.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(one_hot_error, basic_error_wrong_axis) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::i32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(one_hot("output", "input", tensor(1, 1, 1, 50), 4));

    std::string msg_to_find = "Incorrect parameters configuration: one_hot_axis should be less or equal to 3.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(one_hot_error, basic_error_bad_shape) {

    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::i32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(one_hot("output", "input", tensor(1, 5, 1, 50), 2));

    std::string msg_to_find = "Incorrect parameters configuration: shape does not fit input size.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}
