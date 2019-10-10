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
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/fully_connected_grad_weights.hpp"
#include "api/fully_connected.hpp"
#include "api/fully_connected_grad_input.hpp"
#include "api/reorder.hpp"
#include <api/mutable_data.hpp>
#include <api/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(fully_connected_grad_weights_gpu, basic_bfyx) {
    //  Filter : 2x2
    //  Input  : 1x1x1x3
    //  Stride : 2x2
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Input_grad:
    //   1.5   0.75  -2.25  3

    const auto& engine = get_test_engine();
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 1 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 4, 1, 3, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });

    set_values(input, { -0.5f, 2.0f, 0.5f });
    set_values(input_grad, { 1.5f, 0.75f, -2.25f, 3.0f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        fully_connected_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" })
    );

    network network(engine, topology);
    network.set_input_data("input_grad", input_grad);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv_grad_weights");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto weights_ptr = weights.pointer<float>();
    auto biases_ptr = biases.pointer<float>();

    std::vector<float> expected_weights_vec = {
        0.75f, -3.f, -0.75,
        0.375f, -1.5f, -0.375f,
        -1.125f, 4.5f, 1.125f,
        1.5f, -6.f, -1.5f
    };

    std::vector<float> expected_bias_vec = {
        -1.5f, -0.75f, 2.25f, -3.0f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i]*lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on weights verification" << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i]*lr), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on biases verification" << random_seed << std::endl;
    }
}

TEST(fully_connected_grad_weights_gpu, basic_bfyx_b8) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  -0.5     2    0.5
    //  1     1    1
    //
    //  Input_grad:
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1.5   0.75  -2.25  3
    //   1   1  1  1

    const auto& engine = get_test_engine();
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 8, 1, 4, 1 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 8, 1, 3, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 4, 1, 3, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });

    set_values(input, { -0.5f, 2.0f, 0.5f,
                        -0.5f, 2.0f, 0.5f, 
                        -0.5f, 2.0f, 0.5f, 
                        -0.5f, 2.0f, 0.5f, 
                        -0.5f, 2.0f, 0.5f, 
                        -0.5f, 2.0f, 0.5f, 
                        -0.5f, 2.0f, 0.5f, 
                        1.f, 1.f, 1.f });
    set_values(input_grad, { 1.5f, 0.75f, -2.25f, 3.0f,
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.5f, 0.75f, -2.25f, 3.0f, 
                            1.f, 1.f, 1.f, 1.f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        fully_connected_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" })
    );

    network network(engine, topology);
    network.set_input_data("input_grad", input_grad);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv_grad_weights");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto weights_ptr = weights.pointer<float>();
    auto biases_ptr = biases.pointer<float>();

    std::vector<float> expected_weights_vec = {
        4.25e-05f, -0.00022f, -6.25e-05f,
        1.625e-05f, -0.000115f, -3.625e-05f,
        -8.875e-05f, 0.000305f, 6.875e-05f,
        9.5e-05f, -0.00043f, -0.000115f
    };

    std::vector<float> expected_bias_vec = {
        -0.000115f, -6.25e-05f, 0.0001475f, -0.00022f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i]), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on weights verification" << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i]), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on biases verification" << random_seed << std::endl;
    }
}

TEST(fully_connected_grad_weights_gpu, basic_bfyx_no_bias) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Input_grad:
    //   1.5   0.75  -2.25  3

    const auto& engine = get_test_engine();
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 4, 1, 3, 1 } });

    set_values(input, { -0.5f, 2.0f, 0.5f });
    set_values(input_grad, { 1.5f, 0.75f, -2.25f, 3.0f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        fully_connected_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input_grad", input_grad);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv_grad_weights");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto weights_ptr = weights.pointer<float>();

    std::vector<float> expected_weights_vec = {
        0.75f, -3.f, -0.75,
        0.375f, -1.5f, -0.375f,
        -1.125f, 4.5f, 1.125f,
        1.5f, -6.f, -1.5f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i]*lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on weights verification" << random_seed << std::endl;
    }
}