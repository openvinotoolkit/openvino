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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/activation.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include <api/CPP/data.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include "api/CPP/reorder.hpp"

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
        activation("not", "input", activation_not));
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
            activation("not", "input", activation_erf));
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
    cldnn_activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
                 0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_hard_sigmoid, params));
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
            activation("not", "input", activation_reciprocal));
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
    cldnn_activation_additional_params params = { 1.0f, 0.5f };
    set_values(input,
               { 1.0f, 0.3f, -3.0f, 4.0f, 5.0f,
                 21.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                 3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 1.0f, -1.0f, 0.1f });

    topology topology(
            input_layout("input", input.get_layout()),
            activation("not", "input", activation_selu, params));
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
            activation("not", "input", activation_softplus));
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
            activation("not", "input", activation_softsign));
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
            activation("not", "input", activation_sign));
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
        float res = input_ptr[i] > 0 ? 1 : input_ptr[i] < 0 ? -1 : 0;
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
        activation("pow", "input", activation_pow, { 2.0f }));
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
        activation("pow", "input", activation_pow, { FLOAT16(3.0f) }));
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
        activation("linear", "input", activation_linear, {FLOAT16(3.0f), FLOAT16(2.0f)}));
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
        activation("relu", "input", activation_relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
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
        activation("relu", "input", activation_relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
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
    auto input_params = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 1 } });
    set_values(input,
    { 0.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });

    std::vector<cldnn_activation_func> funcs = {
        activation_none,
        activation_logistic,
        activation_hyperbolic_tan,
        activation_relu,
        activation_relu_negative_slope,
        activation_clamp,
        activation_softrelu,
        activation_abs,
        activation_linear,
        activation_square,
        activation_sqrt,
        activation_elu,
        activation_sin,
        activation_sinh,
        activation_cos,
        activation_cosh,
        activation_exp,
        activation_not,
        activation_log2,
        activation_tan,
        activation_negative,
        activation_abs
    };

    cldnn_activation_additional_params params = { 0.5f, 2.5f };
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
                case activation_none:
                    EXPECT_FLOAT_EQ(input_ptr[i], output_ptr[i]);
                    break;
                case activation_logistic:
                    EXPECT_FLOAT_EQ(1.f / (1.f + std::exp((float)-input_ptr[i])), output_ptr[i]);
                    break;
                case activation_hyperbolic_tan:
                    EXPECT_FLOAT_EQ(std::tanh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_relu:
                    EXPECT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.f), output_ptr[i]);
                    break;
                case activation_clamp:
                    EXPECT_FLOAT_EQ(std::fmin((float)std::fmax((float)input_ptr[i], params.a), params.b), output_ptr[i]);
                    break;
                case activation_softrelu:
                    EXPECT_FLOAT_EQ(std::log(1.f + std::exp((float)input_ptr[i])), output_ptr[i]);
                    break;
                case activation_abs:
                    EXPECT_FLOAT_EQ(std::fabs(input_ptr[i]), output_ptr[i]);
                    break;
                case activation_linear:
                    EXPECT_FLOAT_EQ((params.a*input_ptr[i] + params.b), output_ptr[i]);
                    break;
                case activation_square:
                    EXPECT_FLOAT_EQ((input_ptr[i] * input_ptr[i]), output_ptr[i]);
                    break;
                case activation_sqrt:
                    if (input_ptr[i] >= 0)
                    {
                        EXPECT_FLOAT_EQ(std::sqrt((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_elu:
                    EXPECT_FLOAT_EQ(std::fmax((float)input_ptr[i], 0.0f) +
                                    params.a*(std::exp(std::fmin((float)input_ptr[i], 0.0f)) - 1), output_ptr[i]);
                    break;
                case activation_sin:
                    EXPECT_FLOAT_EQ(std::sin((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_sinh:
                    EXPECT_FLOAT_EQ(std::sinh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_cos:
                    EXPECT_FLOAT_EQ(std::cos((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_cosh:
                    EXPECT_FLOAT_EQ(std::cosh((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_exp:
                    EXPECT_FLOAT_EQ(std::exp((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_not:
                    EXPECT_FLOAT_EQ((float)(!input_ptr[i]), output_ptr[i]);
                    break;
                case activation_log2:
                    if (input_ptr[i] > 0) //logarithm exist only for positive real values
                    {
                        EXPECT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                    }
                    break;
                case activation_tan:
                    EXPECT_FLOAT_EQ(std::tan((float)input_ptr[i]), output_ptr[i]);
                    break;
                case activation_negative:
                    EXPECT_FLOAT_EQ(-((float)input_ptr[i]), output_ptr[i]);
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

    std::vector<cldnn_activation_func> funcs = {
        activation_asin,
        activation_acos,
        activation_log,
        activation_log2,
        activation_atan,
        activation_asin,
        activation_asinh,
        activation_atanh
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
            case activation_asin:
                EXPECT_FLOAT_EQ(std::asin((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_acos:
                EXPECT_FLOAT_EQ(std::acos((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_log:
                EXPECT_FLOAT_EQ(std::log((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_log2:
                EXPECT_FLOAT_EQ(std::log2((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_atan:
                EXPECT_FLOAT_EQ(std::atan((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_asinh:
                EXPECT_FLOAT_EQ(std::asinh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_acosh:
                EXPECT_FLOAT_EQ(std::acosh((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_atanh:
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
            activation("relu", "reorder", activation_acosh, {0.5f, 0.f}, padding{{0, 0, 0, 0}, 0}));
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
        activation("relu", "reorder", activation_relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
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
        activation("relu", "reorder", activation_relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 0, 0, 0 }, 0 }));
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
        activation("relu", "input", activation_relu_negative_slope, { 0.5f, 0.f }, padding{ { 0, 0, 3, 3 }, 0 }));
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

    std::vector<cldnn_activation_func> funcs = {
        activation_floor,
        activation_ceil
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
            case activation_floor:
                EXPECT_FLOAT_EQ(std::floor((float)input_ptr[i]), output_ptr[i]);
                break;
            case activation_ceil:
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
    std::vector<cldnn_activation_func> funcs = {
        activation_none,
        activation_negative,
        activation_not
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
            case activation_none:
                EXPECT_EQ((int8_t)input_ptr[i], output_ptr[i]);
                break;
            case activation_negative:
                EXPECT_EQ(-((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            case activation_not:
                EXPECT_EQ(!((int8_t)input_ptr[i]), output_ptr[i]);
                break;
            default:
                break;
            }
        }
    }
}
