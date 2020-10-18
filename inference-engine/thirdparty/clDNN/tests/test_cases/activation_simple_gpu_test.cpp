/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include <cmath>
#include <gtest/gtest.h>
#include <algorithm>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/activation.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/data.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include "api/reorder.hpp"

using namespace cldnn;
using namespace tests;

TEST(activation_f32_fw_gpu, not_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0
    //
    //  Output:
    //  0, 1, 0, 0, 0,
    //  1, 0, 0, 0, 0,
    //  0, 0, 0, 1, 0,
    //  0, 0, 0, 0, 1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
      0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 0.0f });
    VF<float> output_vec = {
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("not", "input", activation_func::negation));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, erf_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::erf));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        EXPECT_FLOAT_EQ(std::erf(input_ptr[i]), output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, hard_sigmoid_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::hard_sigmoid, params));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = std::fmax(0.0f, std::fmin(1.0f, params.a * input_ptr[i] + params.b));
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, reciprocal_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::reciprocal));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = 1 / input_ptr[i];
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, selu_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::selu, params));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] <= 0 ? params.b * (params.a * std::exp(input_ptr[i]) - params.a) :
                                        params.b * input_ptr[i];
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, softplus_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::softplus));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = std::log(std::exp(input_ptr[i]) + 1);
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, softsign_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::softsign));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] / (1 + std::abs(input_ptr[i]));
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, sign_basic_yxfb) {
    //  Input:
    //  1 0 -3  4  5
    //  0  2  3  4 -6
    //  3 -3  3  0  1
    //  1  1  1 -1  0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_func::sign));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "not");

    auto output_memory = outputs.at("not").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < b_size * f_size * y_size * x_size; ++i) {
        float res = input_ptr[i] > 0 ? 1.0f : input_ptr[i] < 0 ? -1.0f : 0.0f;
        EXPECT_FLOAT_EQ(res, output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, pow_basic_yxfb) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    set_values(input,
    { 1.0f, 2.0f, 3.0f, 4.0f });
    VF<float> output_vec = { 1.0f, 4.0f, 9.0f, 16.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("pow", "input", activation_func::pow, { 2.0f, 0.0f }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pow");

    auto output_memory = outputs.at("pow").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f16_fw_gpu, pow_basic_yxfb) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 1, 2, 2 } });
    set_values(input,
        { FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.5f) });
    VF<FLOAT16> output_vec = { FLOAT16(1.0f), FLOAT16(8.0f), FLOAT16(27.0f), FLOAT16(91.125f) };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("pow", "input", activation_func::pow, { FLOAT16(3.0f), FLOAT16(0.0f) }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pow");

    auto output_memory = outputs.at("pow").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<FLOAT16>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f16_fw_gpu, linear_basic_yxfb) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, {data_types::f16, format::yxfb, {1, 1, 2, 2}});
    set_values(input,
               {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.5f)});
    VF<FLOAT16> output_vec = {FLOAT16(5.0f), FLOAT16(8.0f), FLOAT16(11.0f), FLOAT16(15.5f)};

    topology topology(
        input_layout("input", input.get_layout()),
        activation("linear", "input", activation_func::linear, {FLOAT16(3.0f), FLOAT16(2.0f)}));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "linear");

    auto output_memory = outputs.at("linear").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<FLOAT16>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_yxfb) {
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
      2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("relu", "input", activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_bfzyx) {
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  2 -1 -4  5  2
    //  2  1 -3  2 -2
    //  4 -3  2  4 -1
    //  1  2  1 -2  2
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1
    //
    //  2   -0.5 -2    5    2
    //  2    1   -1.5  2   -1
    //  4   -1.5  2    4   -0.5
    //  1    2    1   -1    2
    //

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 5, 4, 2 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
        2.0f, -1.0f, -4.0f, 5.0f, 2.0f,
        2.0f, 1.0f, -3.0f, 2.0f, -2.0f,
        4.0f, -3.0f, 2.0f, 4.0f, -1.0f,
        1.0f, 2.0f, 1.0f, -2.0f, 2.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f,
        2.0f, -0.5f, -2.0f, 5.0f, 2.0f,
        2.0f, 1.0f, -1.5f, 2.0f, -1.0f,
        4.0f, -1.5f, 2.0f, 4.0f, -0.5f,
        1.0f, 2.0f, 1.0f, -1.0f, 2.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("relu", "input", activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f16_fw_gpu, basic_yxfb_mish) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { FLOAT16(0.0f), FLOAT16(-2.0f), FLOAT16(-3.0f), FLOAT16(4.0f), FLOAT16(5.0f),
      FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(-6.0f),
      FLOAT16(3.0f), FLOAT16(-3.0f), FLOAT16(3.0f), FLOAT16(5.0f), FLOAT16(1.0f),
      FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(-1.0f), FLOAT16(1.0f) });

    topology topology(
        input_layout("input", input.get_layout()),
        activation("mish", "input", activation_func::mish));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mish");

    auto output_memory = outputs.at("mish").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<FLOAT16>();
    auto input_ptr = input.pointer<FLOAT16>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
        EXPECT_NEAR((FLOAT16)((float)input_ptr[i] * std::tanh(std::log(1.f + std::exp((float)input_ptr[i])))),
                    output_ptr[i], 1e-2f);
    }
}

TEST(activation_f16_fw_gpu, basic_yxfb_hswish) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 2, 5, 2 } });
    set_values(input,
    { FLOAT16(0.0f), FLOAT16(-2.0f), FLOAT16(-3.0f), FLOAT16(4.0f), FLOAT16(5.0f),
      FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(-6.0f),
      FLOAT16(3.0f), FLOAT16(-3.0f), FLOAT16(3.0f), FLOAT16(5.0f), FLOAT16(1.0f),
      FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(-1.0f), FLOAT16(1.0f) });

    topology topology(
        input_layout("input", input.get_layout()),
        activation("hswish", "input", activation_func::hswish));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "hswish");

    auto output_memory = outputs.at("hswish").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<FLOAT16>();
    auto input_ptr = input.pointer<FLOAT16>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
        EXPECT_NEAR((FLOAT16)((float)input_ptr[i] * std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f),
                    output_ptr[i], 1e-3f);
    }
}

TEST(activation_f16_fw_gpu, basic_yxfb_hsigmoid) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 2, 5, 2 } });
    set_values(input,
    { FLOAT16(0.0f), FLOAT16(-2.0f), FLOAT16(-3.0f), FLOAT16(4.0f), FLOAT16(5.0f),
      FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(-6.0f),
      FLOAT16(3.0f), FLOAT16(-3.0f), FLOAT16(3.0f), FLOAT16(5.0f), FLOAT16(1.0f),
      FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(-1.0f), FLOAT16(1.0f) });

    topology topology(
        input_layout("input", input.get_layout()),
        activation("hsigmoid", "input", activation_func::hsigmoid));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "hsigmoid");

    auto output_memory = outputs.at("hsigmoid").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<FLOAT16>();
    auto input_ptr = input.pointer<FLOAT16>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_layout.get_linear_size(); ++i) {
        EXPECT_NEAR((FLOAT16)(std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f),
                    output_ptr[i], 1e-3f);
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_all_functions)
{
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  a: 0.5, b: 2.5
    //

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 5, 4 } });
    auto input_params = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    set_values(input,
    { 0.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });

    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::logistic,
        activation_func::hyperbolic_tan,
        activation_func::relu,
        activation_func::relu_negative_slope,
        activation_func::clamp,
        activation_func::softrelu,
        activation_func::abs,
        activation_func::linear,
        activation_func::square,
        activation_func::sqrt,
        activation_func::elu,
        activation_func::sin,
        activation_func::sinh,
        activation_func::cos,
        activation_func::cosh,
        activation_func::exp,
        activation_func::negation,
        activation_func::log2,
        activation_func::tan,
        activation_func::negative,
        activation_func::abs,
        activation_func::swish,
        activation_func::hswish,
        activation_func::mish,
        activation_func::gelu,
        activation_func::hsigmoid
    };

    activation_additional_params params = { 0.5f, 2.5f };
    set_values(input_params, { params.a, params.b });

    for (uint8_t i = 0 ; i < 2 ; i++)
    {
        for (auto func : funcs)
        {
            topology topology(input_layout("input", input.get_layout()));

            if (i == 0)
            {
                topology.add(activation("activation", "input", func, params));
            }
            else
            {
                topology.add(data("input_params", input_params));
                topology.add(activation("activation", "input", "input_params", func));
            }

            network network(engine, topology);
            network.set_input_data("input", input);
            auto outputs = network.execute();
            EXPECT_EQ(outputs.size(), size_t(1));
            EXPECT_EQ(outputs.begin()->first, "activation");

            auto output_memory = outputs.at("activation").get_memory();
            auto output_layout = output_memory.get_layout();
            auto output_ptr = output_memory.pointer<float>();
            auto input_ptr = input.pointer<float>();

            int y_size = output_layout.size.spatial[1];
            int x_size = output_layout.size.spatial[0];
            int f_size = output_layout.size.feature[0];
            int b_size = output_layout.size.batch[0];
            EXPECT_EQ(output_layout.format, format::yxfb);
            EXPECT_EQ(y_size, 4);
            EXPECT_EQ(x_size, 5);
            EXPECT_EQ(f_size, 1);
            EXPECT_EQ(b_size, 1);

            for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
            {
                switch (func)
                {
                case activation_func::none:
                    EXPECT_FLOAT_EQ(input_ptr[i], output_ptr[i]);
                    break;
                case activation_func::logistic:
                    EXPECT_FLOAT_EQ(1.f / (1.f + std::exp((float)-input_ptr[i])), output_ptr[i]);
                    break;
                case activation_func::hyperbolic_tan:
                    EXPECT_FLOAT_EQ(std::tanh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::relu:
                    EXPECT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.f), output_ptr[i]);
                    break;
                case activation_func::clamp:
                    EXPECT_FLOAT_EQ(std::fmin((float)std::fmax((float)input_ptr[i], params.a), params.b), output_ptr[i]);
                    break;
                case activation_func::softrelu:
                    EXPECT_FLOAT_EQ(std::log(1.f + std::exp((float)input_ptr[i])), output_ptr[i]);
                    break;
                case activation_func::abs:
                    EXPECT_FLOAT_EQ(std::fabs(input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::linear:
                    EXPECT_FLOAT_EQ((params.a*input_ptr[i] + params.b), output_ptr[i]);
                    break;
                case activation_func::square:
                    EXPECT_FLOAT_EQ((input_ptr[i] * input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::sqrt:
                    if (input_ptr[i] >= 0)
                    {
                        EXPECT_FLOAT_EQ(std::sqrt((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_func::elu:
                    EXPECT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.0f) +
                                    params.a*(std::exp(std::fmin((float)input_ptr[i], 0.0f)) - 1), output_ptr[i]);
                    break;
                case activation_func::sin:
                    EXPECT_FLOAT_EQ(std::sin((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::sinh:
                    EXPECT_FLOAT_EQ(std::sinh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::cos:
                    EXPECT_FLOAT_EQ(std::cos((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::cosh:
                    EXPECT_FLOAT_EQ(std::cosh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::exp:
                    EXPECT_FLOAT_EQ(std::exp((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::negation:
                    EXPECT_FLOAT_EQ((float)(!input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::log2:
                    if (input_ptr[i] > 0) //logarithm exist only for positive real values
                    {
                        EXPECT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_func::tan:
                    EXPECT_FLOAT_EQ(std::tan((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::negative:
                    EXPECT_FLOAT_EQ(-((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_func::swish:
                    EXPECT_FLOAT_EQ((float)input_ptr[i] / (1.f + std::exp((float)(-input_ptr[i]))), output_ptr[i]);
                    break;
                case activation_func::hswish:
                    EXPECT_FLOAT_EQ((float)input_ptr[i] * std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f, output_ptr[i]);
                    break;
                case activation_func::mish:
                    EXPECT_NEAR((float)input_ptr[i] * std::tanh(std::log(1.f + std::exp((float)input_ptr[i]))),
                                output_ptr[i], 1e-5f);
                    break;
                case activation_func::gelu:
                    EXPECT_NEAR(0.5f * (float)input_ptr[i] * (1.f + std::erf((float)(input_ptr[i]) / std::sqrt(2.0f))),
                                output_ptr[i], 1e-5f);
                    break;
                case activation_func::hsigmoid:
                    EXPECT_FLOAT_EQ(std::fmin(std::fmax(0.f, (float)input_ptr[i] + 3.f), 6.f) / 6.f, output_ptr[i]);
                    break;
                default:
                    break;
                }
            }
        }
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_asin_acos_log_atan)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 4 } });
    set_values(input, { 0.12f, 0.56f, 0.45f, 0.789f, 0.546f, 0.999f, 0.7899f, 0.6677f});

    std::vector<activation_func> funcs = {
        activation_func::asin,
        activation_func::acos,
        activation_func::log,
        activation_func::log2,
        activation_func::atan,
        activation_func::asin,
        activation_func::asinh,
        activation_func::atanh
    };

    for (auto func : funcs)
    {
        topology topology(input_layout("input", input.get_layout()));
        topology.add(activation("activation", "input", func));

        network network(engine, topology);
        network.set_input_data("input", input);
        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<float>();
        auto input_ptr = input.pointer<float>();

        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(y_size, 4);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, 1);
        EXPECT_EQ(b_size, 1);

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::asin:
                EXPECT_FLOAT_EQ(std::asin((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::acos:
                EXPECT_FLOAT_EQ(std::acos((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::log:
                EXPECT_FLOAT_EQ(std::log((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::log2:
                EXPECT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::atan:
                EXPECT_FLOAT_EQ(std::atan((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::asinh:
                EXPECT_FLOAT_EQ(std::asinh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::acosh:
                EXPECT_FLOAT_EQ(std::acosh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::atanh:
                EXPECT_FLOAT_EQ(std::atanh((float)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_f32_fw_gpu, relu_basic_acosh_yxfb) {
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //  The expected size is the same as input - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5

    const auto &engine = get_test_engine();

    auto input = memory::allocate(engine, {data_types::f32, format::yxfb, {1, 1, 5, 4}});

    set_values(input,
               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                2.0f, 2.0f, 3.0f, 4.0f, 6.0f,
                3.0f, 3.0f, 3.0f, 5.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    topology topology(
            input_layout("input", input.get_layout()),
            reorder("reorder", "input", input.get_layout().with_padding(padding{{0, 0, 2, 1}, 0})),
            activation("relu", "reorder", activation_func::acosh, {0.5f, 0.f}, padding{{0, 0, 0, 0}, 0}));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();
    auto input_ptr = input.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int i = 0; i < x_size * y_size * f_size * b_size; ++i) {
        EXPECT_FLOAT_EQ(std::acosh(input_ptr[i]), output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_input_padding_yxfb) {
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //  The expected size is the same as in put - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });

    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
         1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
         2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
         3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
         1.0f, 1.0f, 1.0f, -0.5f, 1.0f};

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })),
        activation("relu", "reorder", activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_input_padding_bfzyx) {
    //  Input Padding: 0x2x1 (zyx format) out of the reorder layer
    //  The expected size is the same as input - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 5, 4, 2 } });
    set_values(input,
    {
        1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f,

        1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f,
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 2, 1, 0 }, 0 })),
        activation("relu", "reorder", activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();

    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 2);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, relu_basic_output_padding_yxfb) {
    //  Output Padding: 3x3 (yx format)
    //
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    1   -1   -1.5  4    5    0    0    0
    //  0    0    0    2    2    3    4   -3    0    0    0
    //  0    0    0    3   -1.5  3    5    1    0    0    0
    //  0    0    0    1    1    1   -0.5  1    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.5f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 3.0f, 4.0f, -3.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 3.0f, -1.5f, 3.0f, 5.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("relu", "input", activation_func::relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 3, 3 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_size = output_layout.get_buffer_size();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_size.spatial[1];
    int x_size = output_size.spatial[0];
    int f_size = output_size.feature[0];
    int b_size = output_size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 10);
    EXPECT_EQ(x_size, 11);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(activation_f32_fw_gpu, basic_yxfb_floor_ceil)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 4 } });
    set_values(input, { 0.01f, 0.99f, -0.01f, -0.99f, 1.1f, 1.0f, 0.0f, -1.1f });

    std::vector<activation_func> funcs = {
        activation_func::floor,
        activation_func::ceil
    };

    for (auto func : funcs)
    {
        topology topology(input_layout("input", input.get_layout()));
        topology.add(activation("activation", "input", func));

        network network(engine, topology);
        network.set_input_data("input", input);
        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<float>();
        auto input_ptr = input.pointer<float>();

        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(y_size, 4);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, 1);
        EXPECT_EQ(b_size, 1);

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::floor:
                EXPECT_FLOAT_EQ(std::floor((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::ceil:
                EXPECT_FLOAT_EQ(std::ceil((float)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_i8_fw_gpu, basic_yxfb_all_funcs)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });

    std::vector<int8_t> input_vec = {
        1,   0,  5,   1,
        2,   0,  6,  -5,
        3,   0, -7,  12,
        4,   0, -8,   8
    };
    set_values(input, input_vec);

    // functions valid for int8 type input
    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation
    };

    for (auto func : funcs)
    {
        topology topology;
        topology.add(input_layout("input", input.get_layout()));
        topology.add(activation("activation", "input", func));

        network network(engine, topology);
        network.set_input_data("input", input);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<int8_t>();
        auto input_ptr = input.pointer<int8_t>();

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::none:
                EXPECT_EQ((int8_t)input_ptr[i], output_ptr[i]);
                break;
            case activation_func::negative:
                EXPECT_EQ(-((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::negation:
                EXPECT_EQ(!((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_i32_fw_gpu, basic_yxfb_i32_funcs)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::i32, format::yxfb,{ 2, 2, 2, 2 } });

    std::vector<int32_t> input_vec = {
        1,   0,  5,   1,
        2,   0,  6,  -5,
        3,   0, -7,  12,
        4,   0, -8,   8
    };
    set_values(input, input_vec);

    // functions valid for int8 type input
    std::vector<activation_func> funcs = {
        activation_func::none,
        activation_func::negative,
        activation_func::negation,
        activation_func::relu,
        activation_func::clamp
    };

    for (auto func : funcs)
    {
        topology topology;
        activation_additional_params params = {0.0, 1.0};
        topology.add(input_layout("input", input.get_layout()));
        topology.add(activation("activation", "input", func, params));

        network network(engine, topology);
        network.set_input_data("input", input);
        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "activation");

        auto output_memory = outputs.at("activation").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<int32_t>();
        auto input_ptr = input.pointer<int32_t>();

        for (size_t i = 0; i < output_layout.get_linear_size(); ++i)
        {
            switch (func)
            {
            case activation_func::none:
                EXPECT_EQ((int32_t)input_ptr[i], output_ptr[i]);
                break;
            case activation_func::negative:
                EXPECT_EQ(-((int32_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::negation:
                EXPECT_EQ(!((int32_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_func::relu:
                EXPECT_EQ(std::max(static_cast<int32_t>(input_ptr[i]), 0), output_ptr[i]);
                break;
            case activation_func::clamp:
                EXPECT_EQ(std::min(std::max(input_ptr[i], static_cast<int32_t>(params.a)), static_cast<int32_t>(params.b)), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}

TEST(activation_f32_fw_gpu, b_fs_yx_fsv16_prelu) {
    constexpr int b = 1;
    constexpr int f = 17;
    constexpr int x = 2;
    constexpr int y = 2;

    auto eng = get_test_engine();

    auto in_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(b, f, x, y));
    auto params_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, f, 1, 1));

    auto in_mem = cldnn::memory::allocate(eng, in_lay);
    auto params_mem = cldnn::memory::allocate(eng, params_lay);

    auto in_data = generate_random_4d<float>(b, f, y, x, -1, 1);
    auto params_data = generate_random_1d<float>(f, -1, 1);

    set_values(params_mem, params_data);

    auto topo = cldnn::topology(
        cldnn::input_layout("in", in_lay),
        cldnn::reorder("in_fsv16", "in", cldnn::format::b_fs_yx_fsv16, cldnn::data_types::f32),
        cldnn::data("actv_params", params_mem),
        cldnn::activation("actv", "in_fsv16", "actv_params", cldnn::activation_func::relu_negative_slope),
        cldnn::reorder("out", "actv", cldnn::format::bfyx, cldnn::data_types::f32)
    );

    auto net = cldnn::network(eng, topo);
    set_values(in_mem, flatten_4d(format::bfyx, in_data));
    net.set_input_data("in", in_mem);

    auto result = net.execute();
    auto out_mem = result.at("out").get_memory();

    std::vector<float> expected = flatten_4d(format::bfyx, in_data);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] < 0.f) {
            expected[i] = expected[i] * params_data[i / (x * y) % f];
        }
    }

    auto out_ptr = out_mem.pointer<float>();
    ASSERT_EQ(expected.size(), out_ptr.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], out_ptr[i]) << "at i=" << i;
    }
}
