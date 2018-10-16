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
#include "api/CPP/softmax_loss_grad.hpp"
#include <api/CPP/data.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(softmax_loss_grad_f32_fw_gpu, basic1) {

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 4, 1 } });
    auto labels = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(labels, { 1.f, 3.f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("labels", labels),
        softmax_loss_grad("softmax_loss_grad", "input", "labels")
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax_loss_grad");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = { 8.f, -0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 3.f };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}