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
#include "api/fully_connected_grad_input.hpp"
#include <api/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(fully_connected_grad_input_gpu, basic_bfyx) {
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
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Output:
    //  -1.125  5.625   10.125

    const auto& engine = get_test_engine();

    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 4, 1, 3, 1 } });

    set_values(input, { -0.5f, 2.0f, 0.5f });
    set_values(input_grad, { 1.5f, 0.75f, -2.25f, 3.0f });
    set_values(weights, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    topology topology(
        input_layout("input_grad", input_grad.get_layout()),
        data("input", input),
        data("weights", weights),
        fully_connected_grad_input("fully_connected_grad_input", "input_grad", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input_grad", input_grad);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "fully_connected_grad_input");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -1.125f, 5.625f, 10.125f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}