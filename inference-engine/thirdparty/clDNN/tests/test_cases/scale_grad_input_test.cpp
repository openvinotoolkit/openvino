/*
// Copyright (c) 2018 Intel Corporation
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
#include "api/CPP/scale_grad_input.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

#include <iostream>

using namespace cldnn;
using namespace tests;

TEST(scale_grad_input_gpu, basic_in2x3x2x2_scale_same_size) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 3, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale_grad_input("scale_grad", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.3f, 1.1f, 1.3f,
        0.2f, 0.4f, 1.2f, 1.4f,
        0.25f, 0.5f, 1.25f, 1.5f,
        0.6f, 0.8f, 1.6f, 1.8f,
        0.7f, 0.9f, 1.7f, 1.9f,
        0.75f, 1.f, 1.75f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale_grad").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i], 1e-05F);
    }
}