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
#include "api/CPP/convolution_grad_weights.hpp"
#include "api/CPP/convolution.hpp"
#include "api/CPP/convolution_grad_input.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/mutable_data.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input grad:
    //  0.5    0.6    0.9   1
    //  0.7    0.8    1.7   1.8
    //
    //  Input:
    //  8  0.5
    //  6  9

    engine engine;
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(input_grad, { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.7f, 1.8f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
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
        7.2f, 4.2f, 6.3f,
        0.3f, 4.0f, 0.25f,
        5.4f, 3.0f, 4.5f,

        16.2f, 10.2f, 15.3f,
        0.5f, 7.2f, 0.45f,
        9.0f, 5.4f, 8.1f
    };

    std::vector<float> expected_bias_vec = {
        2.6f, 5.4f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification" << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i] * lr), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on biases verification" << random_seed << std::endl;
    }
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in8x1x2x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input grad:
    //  0.5    0.6    0.9   1
    //  0.7    0.8    1.7   1.8
    //  0.5    0.6    0.9   1
    //  0.7    0.8    1.7   1.8
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9

    engine engine;
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 4.f, 7.f });
    set_values(input_grad, { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.7f, 1.8f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.f, 1.7f, 1.8f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
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
        10.f, 5.4f, 8.4f,
        0.4f, 4.8f, 0.3f,
        6.8f, 3.4f, 5.2f,

        28.8f, 17.f, 27.2f,
        1.f, 11.2f, 0.7f,
        16.0f, 7.4f, 11.6f
    };

    std::vector<float> expected_bias_vec = {
        3.6f, 10.4f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification" << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i] * lr), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on biases verification" << random_seed << std::endl;
    }
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_no_bias) {
    //  Filter : 2x2
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input grad:
    //  0.5    0.6    0.9   1
    //  0.7    0.8    1.7   1.8
    //
    //  Input:
    //  8  0.5
    //  6  9

    engine engine;
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 3, 3 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(input_grad, { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.7f, 1.8f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
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
        7.2f, 4.2f, 6.3f,
        0.3f, 4.0f, 0.25f,
        5.4f, 3.0f, 4.5f,

        16.2f, 10.2f, 15.3f,
        0.5f, 7.2f, 0.45f,
        9.0f, 5.4f, 8.1f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification" << random_seed << std::endl;
    }
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_fwd_backw) {
    //  Filter : 2x2
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input:
    //  1  2
    //  3  4
    //
    //  Weights:
    //  2    1
    //  1    1
    //
    //  Bias:
    //  0

    engine engine;
    float lr = 0.001f;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f });
    set_values(weights, { 2.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 0.f } );

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("input_reordered", "input", input.get_layout()),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        convolution("conv", "input_reordered", { "weights" }, { "biases" }, { 1, 1, 1, 1 }, { 0, 0, -1, -1 }),
        convolution_grad_input("conv_grad_input", "conv", { "weights" }, { 1, 1, 1, 1 }, { 0, 0, -1, -1 }),
        convolution_grad_weights("conv_grad_weights", "conv", "input_reordered", { "weights" }, { "biases" }, { 1, 1, 1, 1 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_learning_rate(lr);

    auto outputs = network.execute();

    auto output_prim = outputs.at("conv_grad_input").get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto weights_ptr = weights.pointer<float>();
    auto biases_ptr = biases.pointer<float>();

    std::vector<float> expected_output_vec = {
        30.0f, 32.0f, 38.0f, 45.0f
    };

    std::vector<float> expected_weights_vec = {
        2 - 89 * lr, 1 - 75 * lr, 1 - 72 * lr, 1 - 63 * lr
    };
    
    std::vector<float> expected_bias_vec = {
        -50.0f * lr
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        float x = float_round(expected_output_vec[i]), y = float_round(output_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on output verification" << random_seed << std::endl;
        x = float_round(expected_weights_vec[i]);
        y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "on weights verification" << random_seed << std::endl;
    }

    float x = float_round(expected_bias_vec[0]), y = float_round(biases_ptr[0]);
    EXPECT_FLOAT_EQ(x, y) << "on biases verification" << random_seed << std::endl;
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_split2) {
    //  Filter : 2x2
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input grad:
    //  0.5    0.6    0.9   1
    //  0.7    0.8    1.7   1.8
    //
    //  Input:
    //  8  0.5    1  2
    //  6  9      3  4

    engine engine;
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 3 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.0f, 2.0f, 3.0f, 4.0f });
    set_values(input_grad, { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.7f, 1.8f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        mutable_data("weights2", weights2),
        mutable_data("biases2", biases2),
        convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights", "weights2" }, { "biases", "biases2" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
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
    auto weights2_ptr = weights2.pointer<float>();
    auto biases2_ptr = biases2.pointer<float>();

    std::vector<float> expected_weights_vec = {
        7.2f, 4.2f, 6.3f,
        0.3f, 4.0f, 0.25f,
        5.4f, 3.0f, 4.5f,
    };

    std::vector<float> expected_bias_vec = {
        2.6f
    };

    std::vector<float> expected_weights2_vec = {
        7.2f, 5.1f, 6.8f,
        2.0f, 0.9f, 1.8f,
        4.0f, 2.7f, 3.6f,
    };

    std::vector<float> expected_bias2_vec = {
        5.4f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification" << random_seed << std::endl;
        x = float_round(expected_weights2_vec[i] * lr);
        y = float_round(weights2_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification" << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i] * lr), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on biases verification" << random_seed << std::endl;
        x = float_round(expected_bias2_vec[i] * lr);
        y = float_round(biases2_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on biases verification" << random_seed << std::endl;
    }
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz1x1_in1x2x5x5_bfyx_stride2_pad1) {
    //  Filter : 1x1
    //  Input grad  : 1x2x2x2
    //  Input  : 1x1x2x2
    //  Stride : 2x2
    //
    //  Input grad:
    //  0.5    0.6    0.7    0.9   1      1.1    0.7    0.9    0.1    1.9
    //  0.7    0.8    0.8    1.7   1.8    1.2    2.1    0.5    0.2    0.9
    //  0.6    0.5    0.4    0.2   0.1    1.5    0.6    0.7    0.3    0.8
    //  0.7    0.8    0.9    0.2   0.4    1.8    0.4    0.9    0.4    0.7
    //  0.6    0.5    0.4    0.1   0.1    1.7    0.5    0.4    0.5    0.6
    //
    //  Input:
    //  8  0.5 1  2
    //  6  9   3  4
    //  5  6   7  8
    //  9  10 11 11

    engine engine;
    float lr = 0.00001f;
    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 5, 5 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 1, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    set_values(input, { 
        8.f, 0.5f, 1.f, 2.f,
        6.f, 9.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 11.f
    });
    set_values(input_grad, { 
        0.5f, 0.6f, 0.7f, 0.9f, 1.f,
        0.7f, 0.8f, 0.8f, 1.7f, 1.8f,
        0.6f, 0.5f, 0.4f, 0.2f, 0.1f,
        0.7f, 0.8f, 0.9f, 0.2f, 0.4f,
        0.6f, 0.5f, 0.4f, 0.1f, 0.1f,
        1.1f, 0.7f, 0.9f, 0.1f, 1.9f,
        1.2f, 2.1f, 0.5f, 0.2f, 0.9f,
        1.5f, 0.6f, 0.7f, 0.3f, 0.8f,
        1.8f, 0.4f, 0.9f, 0.4f, 0.7f,
        1.7f, 0.5f, 0.4f, 0.5f, 0.6f
    });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        mutable_data("weights", weights),
        mutable_data("biases", biases),
        convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
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
        19.8f, 34.6f
    };

    std::vector<float> expected_bias_vec = {
        16.0f, 21.4f
    };

    for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
    {
        float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on weights verification " << random_seed << std::endl;
    }

    for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
    {
        float x = float_round(expected_bias_vec[i] * lr), y = float_round(biases_ptr[i]);
        EXPECT_FLOAT_EQ(x, -y) << "on biases verification " << random_seed << std::endl;
    }
}

TEST(convolution_grad_weights_f32_fw_gpu, basic_wsiz2x2_in32x1x2x2_yxfb_stride1) {
	//  Filter : 1x1
	//  Input grad  : 32x1x2x2
	//  Input  : 32x1x3x3
	//  Stride : 1x1
	//
	//  Input grad:
	// y0: x0: 0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  1.9  0.6  0.5  0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.5  0.6 
	// y0: x1: 0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  1.9  0.1  1.5  0.6  2.1  0.4  0.3
	// y1: x0: 0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  1.9  0.1  1.7  0.5  0.4  0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  1.2  2.1  0.5  0.2  0.9  0.4  0.1  1.2  0.2  0.1
	// y1: x1: 0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.9  0.1  1.9  0.1  1.7  0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  0.1  1.7  0.5  0.4  1.7  0.5  0.4  0.5  0.6  0.0  0.7
	//
	//  Input:
	// y0: x0: 0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  0.6  1.2  2.1  0.1  0.2
	// y0: x1: 0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.4  0.1  0.1  1.7  0.5  0.4  0.5  1    1.1  0.7  0.9  0.1  1.9  0.1  1.7  0.5  0.4  0.4  0.1  0.1  0.3  0.4
	// y0: x2: 0.1  1.7  0.5  0.4  0.4  0.1  0.1  1.7  0.5  0.4  0.5  1.1  0.7  0.9  0.1  0.1  1.7  0.5  0.1  1.9  0.6  0.5  0.4  0.1  0.1  1.7  0.5  0.4  2.1  0.5  0.5  0.6
	// y1: x0: 1.9  0.1  1.7  0.5  0.6  0.7  0.9  1    1.1  0.7  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.4  0.2  0.9  0.4  0.1  1.2  1.9  0.1  1.5  0.6  2.1  2.3  0.7  0.8
	// y1: x1: 0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  0.6  1.2  2.1  0.9  1.0
	// y1: x2: 0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  0.4  0.1  0.1  1.7  0.5  0.4  0.5  1    1.1  0.7  0.9  0.1  1.9  0.1  1.7  0.5  0.4  0.4  0.1  0.1  1.1  1.2
	// y2: x0: 0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.9  0.1  1.9  0.1  1.7  0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  0.1  1.7  0.5  0.4  1.7  0.5  0.4  0.5  0.6  1.3  1.4
	// y2: x1: 0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  1.9  0.6  0.5  0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  0.7  0.8  0.8  1.7  1.8  1.2  2.1  0.5  0.2  0.9  1.5  1.6
	// y2: x2: 0.5  0.6  0.7  0.9  1    1.1  0.7  0.9  0.1  1.9  0.1  1.7  0.5  0.4  0.4  0.1  0.1  1.7  0.5  0.4  0.5  0.6  1.2  2.1  0.5  0.2  0.9  0.4  0.1  1.2  1.7  1.8

	engine engine;
	float lr = 0.00001f;
	auto input_grad = memory::allocate(engine, { data_types::f32, format::yxfb,{ 32, 1, 2, 2 } });
	auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 32, 1, 3, 3 } });
	auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
	auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

	set_values(input, {
		0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 0.6f, 1.2f, 2.1f, 0.1f, 0.2f,
		0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.4f, 0.4f, 0.1f, 0.1f, 0.3f, 0.4f,
		0.1f, 1.7f, 0.5f, 0.4f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 1.1f, 0.7f, 0.9f, 0.1f, 0.1f, 1.7f, 0.5f, 0.1f, 1.9f, 0.6f, 0.5f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 2.1f, 0.5f, 0.5f, 0.6f,
		1.9f, 0.1f, 1.7f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.2f, 0.9f, 0.4f, 0.1f, 1.2f, 1.9f, 0.1f, 1.5f, 0.6f, 2.1f, 2.3f, 0.7f, 0.8f,
		0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 0.6f, 1.2f, 2.1f, 0.9f, 1.0f,
		0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.4f, 0.4f, 0.1f, 0.1f, 1.1f, 1.2f,
		0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 1.3f, 1.4f,
		0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.6f, 0.5f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 1.5f, 1.6f,
		0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.4f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.1f, 1.2f, 1.7f, 1.8f
		});
	set_values(input_grad, {
		0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.6f, 0.5f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.5f, 0.6f,
		0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.1f, 1.5f, 0.6f, 2.1f, 0.4f, 0.3f,
		0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.4f, 0.4f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.2f, 0.9f, 0.4f, 0.1f, 1.2f, 0.2f, 0.1f,
		0.7f, 0.8f, 0.8f, 1.7f, 1.8f, 1.2f, 2.1f, 0.9f, 0.1f, 1.9f, 0.1f, 1.7f, 0.5f, 0.6f, 0.7f, 0.9f, 1.f,  1.1f, 0.7f, 0.9f, 0.1f, 0.1f, 1.7f, 0.5f, 0.4f, 1.7f, 0.5f, 0.4f, 0.5f, 0.6f, 0.0f, 0.7f
		});

	topology topology(
		input_layout("input_grad", input_grad.get_layout()),
		data("input", input),
		mutable_data("weights", weights),
		mutable_data("biases", biases),
		convolution_grad_weights("conv_grad_weights", "input_grad", "input", { "weights" }, { "biases" }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 })
	);

	network network(engine, topology);
	network.set_input_data("input_grad", input_grad);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "conv_grad_weights");

	auto output_prim = outputs.begin()->second.get_memory();

	auto weights_ptr = weights.pointer<float>();
	auto biases_ptr = biases.pointer<float>();

	std::vector<float> expected_weights_vec = {
		90.58f, 85.92f, 97.22f, 91.86f
	};

	std::vector<float> expected_bias_vec = {
		102.6f
	};

	for (unsigned int i = 0; i < expected_weights_vec.size(); i++)
	{
		float x = float_round(expected_weights_vec[i] * lr), y = float_round(weights_ptr[i]);
		EXPECT_FLOAT_EQ(x, -y) << "on weights verification " << random_seed << std::endl;
	}

	for (unsigned int i = 0; i < expected_bias_vec.size(); i++)
	{
		float x = float_round(expected_bias_vec[i] * lr), y = float_round(biases_ptr[i]);
		EXPECT_FLOAT_EQ(x, -y) << "on biases verification " << random_seed << std::endl;
	}
}