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
#include "api/CPP/deconvolution.hpp"
#include <api/CPP/data.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include "api/CPP/reorder.hpp"


using namespace cldnn;
using namespace tests;

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25 
    //   18    0.75  7.25   
    //   23    42.5  15.5   

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}


TEST(deconvolution_f32_fw_gpu, no_bias_basic_wsiz2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  no bias
    //  
    //
    //  Output:
    //  -14    5     2.25 
    //   18    0.75  7.25   
    //   23    42.5  15.5   

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -16.f, 3.f, 0.25f,
        16.f, -1.25f, 5.25f,
        21.f, 40.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}


TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad_bfyx) {    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25 
    //   18    0.75  7.25   
    //   23    42.5  15.5   

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  0.75  

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 1, 1 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(0.75f, output_ptr[0]);
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride2_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  0.75  

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2) {
    //  Filter : 3x3
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 4x4
    //  Pad    : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5   1
    //   3.5 1.5   2
    //   3   4     5
    //
    //  Bias
    //  0
    //
    //  Output:
    //  40   0    1.5
    //  0    0    0
    //  6    0   -18

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f });
    set_values(biases, { 0.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 4, 4 }, { 0, 0, -2, -2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        40.f, 0.f, 1.5f,
        0.f, 0.f, 0.f,
        6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 1.f, 0.5f, 3.f, 6.f, 2.f, 9.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 0.5f, 4.5f, 22.f,
        13.f, 5.f, -17.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  f0:-2   2
    //  f0: 7  -0.5
    //  f1:-2   2
    //  f1: 7  -0.5
    //
    //  Bias
    //  1  5
    //
    //  Output:
    //  f0: -3   4.5 
    //  f0: 13   -17 
    //  f1: 1    8.5
    //  f1: 17 - 13

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_input_padding) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Out Padding   : 1x1
    //  Input Padding : 2x1 (with reorder)
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "reorder", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1_input_padding) {
    //  Filter : 2x2
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Stride : 2x2
    //  Out Padding   : 1x1
    //  Input Padding : 2x1 (with reorder)
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  f0:-2   2
    //  f0: 7  -0.5
    //  f1:-2   2
    //  f1: 7  -0.5
    //
    //  Bias
    //  1  5
    //
    //  Output:
    //  f0: -3   4.5 
    //  f0: 13   -17 
    //  f1: 1    8.5
    //  f1: 17 - 13

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "reorder", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f), FLOAT16(6.f), FLOAT16(9.f),
        FLOAT16(1.f), FLOAT16(3.f), FLOAT16(2.f), FLOAT16(4.f) });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f});
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<uint16_t>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2) {
    //  Filter : 2x2x2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter1
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Filter2
    //  -4   1
    //  -9  -7
    //
    //  Bias
    //  -1
    //
    //  Output:
    //  -3    4.5    -8   -28
    //   13  -17     1    -17

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });
    set_values(weights2, { -4.f, 1.f, -9.f, -7.f });
    set_values(biases2, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("weights2", weights2),
        data("biases2", biases2),
        deconvolution("deconv", "input", { "weights", "weights2" }, { "biases", "biases2" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group2) {
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, {
        -2.f, 2.f, 7.f, -0.5f,
        -4.f, 1.f, -9.f, -7.f
    });
    set_values(biases, { 1.0f, -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2

    const auto& engine = get_test_engine(); 

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input, 
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f, 
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
        set_values(biases, { 1.0f });
        set_values(weights2, { -4.f, 1.f, -9.f, -7.f });
        set_values(biases2, { -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights),
            data(bias_id, biases),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
            );
    }

    topology.add(deconvolution("deconv", "input", weights_vec, bias_vec, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16) {
    //  Test for depthwise separable optimization, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input,
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;
    
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 16, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 16, 1 } });

    set_values(weights, 
        {
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f
        }
    );
    set_values(biases, { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f });
    topology.add(
        data("weights", weights),
        data("bias", biases)
    );
   
    topology.add(deconvolution("deconv", "input", { "weights" }, { "bias" }, 16, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt_ofm2) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input,
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
        auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

        set_values(weights, { -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f });
        set_values(biases, { 1.0f, 1.0f });
        set_values(weights2, { -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f });
        set_values(biases2, { -1.0f, -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights),
            data(bias_id, biases),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
        );
    }

    topology.add(deconvolution("deconv", "input", weights_vec, bias_vec, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16_ofm2) {
    //  Test for depthwise separable optimization, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt_ofm2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input,
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;
    
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 32, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 32, 1 } });
    
    set_values(weights,
        {
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
        }
    );

    set_values(biases,
        {
            1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
            1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f
        }
    );

    topology.add(
        data("weights", weights),
        data("bias", biases)
    );

    topology.add(deconvolution("deconv", "input", { "weights" }, { "bias" }, 16, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}


TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_split2_ofm3) {
    //  Filter : 1x1
    //  Stride : 1x1
    //  Input  : 1x1x4
    //  Output : 1x1x6
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  2
    //  f3: -1.0
    //
    //  Filter1:
    //  -2   1   ofm=0
    //   1   3   ofm=1
    //   0.5 8   ofm=2
    //  Bias1:
    //   1   5   3
    //
    //  Filter2:
    //   4  -4   ofm=3
    //   2   0.5 ofm=4
    //  -0.5 3   ofm=5
    //
    //  Bias2:
    //  -1   2.5 2
    //
    //  Output:
    //  -1.5  
    //   8
    //   7.75
    //
    //   11
    //   6
    //  -2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 4, 1, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 3, 2, 1, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 3, 2, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights, { -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f });
    set_values(biases, { 1.0f, 5.0f, 3.0f });
    set_values(weights2, { 4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f });
    set_values(biases2, { -1.0f, 2.5f, 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("weights2", weights2),
        data("biases2", biases2),
        deconvolution("deconv", "input", { "weights", "weights2" }, { "biases", "biases2" }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -1.5f, 8.0f, 7.75f, 11.0f, 6.0f, -2.0f
    };
    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_group2_ofm3) {
    //  data is similar as in basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_split2_ofm3

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 4, 1, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 6, 2, 1, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 6, 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights, {
        -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f,
        4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f
    });
    set_values(biases, {
        1.0f, 5.0f, 3.0f,
        -1.0f, 2.5f, 2.0f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 1, 1 }, { 0, 0, 0, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -1.5f, 8.0f, 7.75f, 11.0f, 6.0f, -2.0f
    };
    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}
TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x1_in1x1x2x2x1_nopad) {
    //  Filter : 2x2x1
    //  Input  : 2x2x1
    //  Output : 3x3x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25
    //   18    0.75  7.25
    //   23    42.5  15.5

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 1 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz3x3x3_in1x1x4x4x4_nopad) {
    //  Filter : 3x3x3
    //  Input  : 3x3x3
    //  Output : 6x6x6
    //
    //  Input:
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //
    //  Filter
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //
    //  Output:
    //
    //  1  2  3  3  2  1
    //  2  4  6  6  4  2
    //  3  6  9  9  6  3
    //  3  6  9  9  6  3
    //  2  4  6  6  4  2
    //  1  2  3  3  2  1
    //
    //  2   4   6   6   4  2
    //  4   8  12  12   8  4
    //  6  12  18  18  12  6
    //  6  12  18  18  12  6
    //  4   8  12  12   8  4
    //  2   4   6   6   4  2
    //
    //  3   6   9   9   6  3
    //  6  12  18  18  12  6
    //  9  18  27  27  18  9
    //  9  18  27  27  18  9
    //  6  12  18  18  12  6
    //  3   6   9   9   6  3
    //
    //  3   6   9   9   6  3
    //  6  12  18  18  12  6
    //  9  18  27  27  18  9
    //  9  18  27  27  18  9
    //  6  12  18  18  12  6
    //  3   6   9   9   6  3
    //
    //  2   4   6   6   4  2
    //  4   8  12  12   8  4
    //  6  12  18  18  12  6
    //  6  12  18  18  12  6
    //  4   8  12  12   8  4
    //  2   4   6   6   4  2
    //
    //  1  2  3  3  2  1
    //  2  4  6  6  4  2
    //  3  6  9  9  6  3
    //  3  6  9  9  6  3
    //  2  4  6  6  4  2
    //  1  2  3  3  2  1
    //

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 4, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 3, 3, 3 } });

    set_values(input,
    {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    });
    set_values(weights, {
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,

        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,

        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,

        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,

        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,

        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_nopad) {
    //  Filter : 2x2x2
    //  Input  : 2x2x2
    //  Output : 1x1
    //  Stride : 2x2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //  -2   0.5
    //   3.5 1.5
    //

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });
    //set_values(input, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f });
    //set_values(weights, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();
    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_pad1) {
    //  Filter : 2x2x2
    //  Input  : 2x2x2
    //  Output : 1x1
    //  Stride : 2x2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //  -2   0.5
    //   3.5 1.5
    //
    //  Output:
    //  12 1.75
    //   3 -18
    //  12 1.75
    //   3 -18

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 }, { 0, 0, -1, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();
    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        12.f, 1.75f, 3.f, -18.f,
        12.f, 1.75f, 3.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }

}
