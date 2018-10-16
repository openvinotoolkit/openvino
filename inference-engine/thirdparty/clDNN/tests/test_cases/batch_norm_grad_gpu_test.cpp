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
#include "api/CPP/batch_norm_grad.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/reorder.hpp>
#include <api/CPP/data.hpp>

using namespace cldnn;
using namespace tests;

TEST(batch_normalization_backward_gpu, basic_in2x2x2x3) {
    //  Grad input  : 2x2x2x3
    //  Input : 2x2x2x3
    //  Inverted variance : 1x2x1x1
    //  Output : 2x2x2x3

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13     
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //
    //  Grad input
    //  f0: b0:  1    2  3   b1:  -1    -2     -3
    //  f0: b0:  5    6  7   b1:   0.5  -0.5   -4  
    //  f1: b0:  8    9  10  b1:   1.5   5     -5     
    //  f1: b0: 11   12  13  b1:   2    -7.2   -6
    //
    //  Inverted variance
    //  f0: 0.1491862
    //  f1: 0.0966454

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto grad_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto inv_var = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("grad_input", grad_input));
    topology.add(data("inv_var", inv_var));
    topology.add(batch_norm_grad("batch_norm_grad", "grad_input", "input", "inv_var"));

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f, 
        5.f, 6.f, -12.f, 
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f, 
        0.5f, -0.5f, -15.f, 
        1.5f, 5.2f, -13.f, 
        12.f, 9.f, -17.f
    });

    set_values(grad_input, {
        1.f, 2.f, 3.f,
        5.f, 6.f, 7.f,
        8.f, 9.f, 10.f,
        11.f, 12.f, 13.f,
        -1.f, -2.f, -3.f,
        0.5f, -0.5f, -4.f,
        1.5f, 5.f, -5.f,
        2.f, -7.2f, -6.f
    });

    set_values(inv_var, { 0.1491862f, 0.0966454f });

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm_grad").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_out = {
        -0.142969f, -0.111888f, 1.45456f, 
         0.217566f, 0.248648f, 2.52372f,
        -3.41923f, -4.07521f, 5.f,
        -4.63455f, -5.f, 5.f,
        -0.323237f, -0.472423f, 0.677543f, 
        -0.15851f, -0.189591f, 1.00078f,
        -1.41324f, -3.85969f, 5.f,
        -5.f, -5.f, 5.f
    };

    for (int i = 0; i < 2 * 2 * 3 * 2; i++)
    {    
        EXPECT_NEAR(expected_out[i], output_ptr[i], 1e-03F);
    }
}