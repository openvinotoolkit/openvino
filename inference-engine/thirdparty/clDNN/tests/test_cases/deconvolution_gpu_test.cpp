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
#include "api/deconvolution.hpp"
#include "api/crop.hpp"
#include <api/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include "api/reorder.hpp"

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 3, 3 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxio, { 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });
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
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::yxio,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

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

    network network(engine, topology, options);
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f),
                        FLOAT16(6.f), FLOAT16(9.f),

                        FLOAT16(1.f), FLOAT16(3.f),
                        FLOAT16(2.f), FLOAT16(4.f) });
    set_values(weights, { -2.f, 2.f,
                          7.f, -0.5f});
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
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f, -4.f, 1.f, -9.f, -7.f });
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

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group2) {
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

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

    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 1, 1 } });

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

    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(16), batch(2), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 32, 1, 1 } });

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

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_group2_ofm3) {
    //  data is similar as in basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_split2_ofm3

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 4, 1, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(3), feature(2), spatial(1, 1)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 6, 1, 1 } });

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oizyx,{ 1, 1, 2, 2, 1 } });
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
    auto weights = memory::allocate(engine, { data_types::f32, format::oizyx,{ 1, 1, 3, 3, 3 } });

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

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
    auto weights = memory::allocate(engine, { data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

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

TEST(deconvolution_f16_gpu, basic_k9x9_s2x2_pad4x4) {
    //  Filter : 1x32x9x9
    //  Input  : 1x32x16x16
    //  Stride : 2x2
    //  Pad    : 4x4

    //const auto& engine = get_test_engine();
    engine engine;

    VVVVF<FLOAT16> input_rnd = generate_random_4d<FLOAT16>(1, 32, 16, 16, -2, 2);
    VF<FLOAT16> input_rnd_vec = flatten_4d<FLOAT16>(format::bfyx, input_rnd);
    VVVVF<FLOAT16> filter_rnd = generate_random_4d<FLOAT16>(1, 32, 9, 9, -1, 1);
    VF<FLOAT16> filter_rnd_vec = flatten_4d<FLOAT16>(format::bfyx, filter_rnd);
    VF<FLOAT16> bias_rnd = generate_random_1d<FLOAT16>(1, -1, 1);
    VF<float> filter_rnd_f32_vec, bias_f32_rnd;

    for (unsigned int i = 0; i < filter_rnd_vec.size(); i++)
        filter_rnd_f32_vec.push_back(float(filter_rnd_vec[i]));

    for (unsigned int i = 0; i < bias_rnd.size(); i++)
        bias_f32_rnd.push_back(float(bias_rnd[i]));

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 32, 16, 16 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::oiyx, { 1, 32, 9, 9 } });
    auto biases = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 1, 1, 1 } });
    auto weights_f32 = memory::allocate(engine, { data_types::f32, format::oiyx, { 1, 32, 9, 9 } });
    auto biases_f32 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);
    set_values(weights_f32, filter_rnd_f32_vec);
    set_values(biases_f32, bias_f32_rnd);

    topology topology_ref(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -4, -4 }, tensor{ 1, 1, 32, 32 })
    );

    network network_ref(engine, topology_ref);
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "deconv");
    auto output_ref_prim = outputs_ref.begin()->second.get_memory();
    auto output_ref_ptr = output_ref_prim.pointer<FLOAT16>();

    std::vector<FLOAT16> output_vec_ref;
    for (unsigned int i = 0; i < output_ref_prim.get_layout().count(); i++)
    {
        output_vec_ref.push_back(output_ref_ptr[i]);
    }

    topology topology_act(
        input_layout("input_act", input.get_layout()),
        data("weights_f32", weights_f32),
        data("biases_f32", biases_f32),
        deconvolution("deconv_act", "input_act", { "weights_f32" }, { "biases_f32" }, { 1, 1, 2, 2 }, { 0, 0, -4, -4 }),
        reorder("out", "deconv_act", format::bfyx, data_types::f16)
    );

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network_act(engine, topology_act, options);
    network_act.set_input_data("input_act", input);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "out");
    auto output_act_prim = outputs_act.begin()->second.get_memory();
    auto output_act_ptr = output_act_prim.pointer<FLOAT16>();

    std::vector<float> output_vec;
    for (unsigned int i = 0; i < output_act_prim.get_layout().count(); i++)
    {
        float x = float_round(output_act_ptr[i]), y = float_round(output_vec_ref[i]);
        EXPECT_NEAR(x, y, 1e-0f);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_b_fs_yx_fsv16_stride2_pad1) {
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
    auto weights = memory::allocate(engine, { data_types::f32, format::yxio, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in2x2x1x2_b_fs_yx_fsv16_stride2_pad1) {
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
    auto weights = memory::allocate(engine, { data_types::f16, format::oiyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f16, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f),
                        FLOAT16(6.f), FLOAT16(9.f),

                        FLOAT16(1.f), FLOAT16(3.f),
                        FLOAT16(2.f), FLOAT16(4.f) });
    set_values(weights, { FLOAT16(-2.f), FLOAT16(2.f),
                          FLOAT16(7.f), FLOAT16(-0.5f)});
    set_values(biases, { FLOAT16(1.0f) });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f16)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<uint16_t>();

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_b_fs_yx_fsv16_stride2_pad1_group2) {
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

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
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_b_fs_yx_fsv16_stride2_pad1_b_fs_yx_fsv16_dw) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, {
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f
    });
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            -4.f, 3.5f, 12.f, -18.f,
            -7.f, -27.f, 2.f, -16.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad_b_fs_yx_fsv16_dw) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,  8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,  -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f, 2.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            reorder("input_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", "input_fsv16", { "weights" }, { "biases" }, 2, { 1,1,1,1 }, { 0, 0, 0, 0 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            -14.f, 5.f, 2.25f,
            18.f, 0.75f, 7.25f,
            23.f, 42.5f, 15.5f,

            -14.f, 5.f, 2.25f,
            18.f, 0.75f, 7.25f,
            23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_pad1_b_fs_yx_fsv16_dw) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,
                          -2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases, { 2.0f, 2.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            reorder("input_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", "input_fsv16", { "weights" }, { "biases" }, 2, { 1, 1, 1, 1 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(0.75f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.75f, output_ptr[1]);
}


TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride2_nopad_b_fs_yx_fsv16_dw) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,
                          -2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases, { 1.0f, 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1,1,2,2 }),
        reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f,


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

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2_b_fs_yx_fsv16_dw) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(3, 3)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f,
                          -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f});
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 4, 4 }, { 0, 0, -2, -2 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2_b_fs_yx_fsv16_dw_batch2) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(3, 3)) });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f,

                        8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f,
                          -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f});
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 4, 4 }, { 0, 0, -2, -2 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,



            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, bs_fs_zyx_bsv16_fsv16_wsiz2x2x2_in1x1x2x2x2_stride2_nopad) {
    //  Batch : 32
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

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 32, 1, 2, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

    std::vector<float> input_single_batch = { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f };
    std::vector<float> input_batched;
    for (size_t i = 0; i < 32; i++) {
        for (size_t j = 0; j < 8; j++) {
            input_batched.push_back(input_single_batch[j]);
        }
    }

    set_values(input, input_batched);
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
            input_layout("input", input.get_layout()),
            data("weights", weights),
            deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 }, { 0, 0, -1, -1, -1 }),
            reorder("out", "deconv", format::bfzyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::bs_fs_zyx_bsv16_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();
    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            12.f, 1.75f, 3.f, -18.f,
            12.f, 1.75f, 3.f, -18.f
    };

    for (size_t b = 0; b < 32; b++) {
        for (size_t i = 0; i < expected_output_vec.size(); i++) {
            ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[b*expected_output_vec.size() + i]) << " b = " << b << " i = " << i;
        }
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f16, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f), FLOAT16(6.f), FLOAT16(9.f),
                        FLOAT16(1.f), FLOAT16(3.f), FLOAT16(2.f), FLOAT16(4.f)
                        });
    set_values(weights, {
            FLOAT16(-2.f), FLOAT16(2.f), FLOAT16(7.f), FLOAT16(-0.5f),
            FLOAT16(-4.f), FLOAT16(1.f), FLOAT16(-9.f), FLOAT16(-7.f)
    });
    set_values(biases, { FLOAT16(1.0f), FLOAT16(-1.0f) });

    topology topology(
            input_layout("input", input.get_layout()),
            reorder("reorder", "input", format::fs_b_yx_fsv32, data_types::f16),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "reorder", { "weights" }, { "biases" }, 1, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
            -15.f, 16.f, 2.f, 45.f, -5.5f, 18.75f, 43.f, 61.f, -3.5f,
            -33.f, 5.f, -0.5f, -97.f, -91.5f, 4.5f, -55.f, -124.f, -64.f,
            -1.f, -3.f, 7.f, 4.f, 17.5f, 7.5f, 15.f, 28.f, -1.f,
            -5.f, -12.f, 2.f, -18.f, -49.f, -18.f, -19.f, -51.f, -29.f,
    };
    ASSERT_EQ(expected_output_vec.size(), output_prim.count());

    for (size_t i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]) << " index=" << i;
    }
}
