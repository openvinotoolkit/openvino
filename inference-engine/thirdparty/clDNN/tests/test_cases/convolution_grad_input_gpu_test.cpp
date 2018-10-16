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
#include "api/CPP/convolution_grad_input.hpp"
#include <api/CPP/data.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(convolution_grad_input_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Output:
    //  -4    3.5    -0.5   21
    //   12  -18      4     -9

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution_grad_input("deconv", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -4.f, 3.5f, 12.f, -18.f,
        -.5f, 21.f, 4.f, -8.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}


TEST(convolution_grad_input_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_output_size) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Output:
    //  -4    3.5    -0.5   21
    //   12  -18      4     -9

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution_grad_input("deconv", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }, { 2, 1, 2, 2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -4.f, 3.5f, 12.f, -18.f,
        -.5f, 21.f, 4.f, -8.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}