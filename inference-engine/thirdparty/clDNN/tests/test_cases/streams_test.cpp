/*
// Copyright (c) 2019 Intel Corporation
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

#include <gtest/gtest.h>
#include <api/tensor.hpp>
#include <api/engine.hpp>
#include <api/memory.hpp>
#include <api/input_layout.hpp>
#include <api/data.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

static engine_configuration streams_config(false,
                                                 false,
                                                 false,
                                                 "",
                                                 "",
                                                 true,
                                                 "",
                                                 "",
                                                 priority_mode_types::disabled,
                                                 throttle_mode_types::disabled,
                                                 true,
                                                 2);

TEST(gpu_streams, DISABLED_can_allocate_memory_for_stream)
{
    engine _engine(streams_config);

    ASSERT_NO_THROW(memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4})));
    ASSERT_NO_THROW(memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4}), 0));
    ASSERT_NO_THROW(memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4}), 1));
    ASSERT_ANY_THROW(memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4}), 2));

    auto mem0 = memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4}), 0);
    ASSERT_EQ(mem0.get_net_id(), 0);
    auto mem1 = memory::allocate(_engine, layout(data_types::f32, format::bfyx, {1, 2, 3, 4}), 1);
    ASSERT_EQ(mem1.get_net_id(), 1);
}

TEST(gpu_streams, can_create_networks_for_stream)
{
    engine _engine(streams_config);

    auto input = memory::allocate(_engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
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
            activation("relu", "input", activation_func::relu_negative_slope, activation_additional_params{ 0.5f, 0.f }, padding{ { 0, 0, 0, 0 }, 0 }));
    network network(_engine, topology, build_options());

    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    EXPECT_EQ(output_memory.get_net_id(), network.get_id());

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

TEST(gpu_streams, check_networks_can_use_the_same_weights) {
    engine _engine(streams_config);

    auto weights = memory::allocate(_engine, { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

    VVF<float> output_vec = {
            { 20.0f, 27.0f, 38.0f },
            { 17.0f, 19.0f, 19.0f } };

    layout input0_layout(data_types::f32, format::yxfb, { 1, 1, 5, 4 });

    topology topology(
            input_layout("input", input0_layout),
            data("weights", weights),
            convolution("conv", "input", { "weights" }, { 1,1,1,2 }));

    network network0(_engine, topology, build_options());
    network network1(_engine, topology, build_options());

    auto input0 = memory::allocate(_engine, input0_layout, network0.get_id());
    auto input1 = memory::allocate(_engine, input0_layout, network1.get_id());
    set_values(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(input1, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

    network0.set_input_data("input", input0);
    network1.set_input_data("input", input1);

    auto outputs0 = network0.execute();
    auto outputs1 = network1.execute();
    EXPECT_EQ(outputs0.size(), size_t(1));
    EXPECT_EQ(outputs1.size(), size_t(1));
    EXPECT_EQ(outputs0.begin()->first, "conv");
    EXPECT_EQ(outputs1.begin()->first, "conv");

    auto output_memory0 = outputs0.at("conv").get_memory();
    auto output_memory1 = outputs1.at("conv").get_memory();
    auto output_layout = output_memory0.get_layout();
    auto output_ptr0 = output_memory0.pointer<float>();
    auto output_ptr1 = output_memory1.pointer<float>();

    EXPECT_EQ(output_memory0.get_net_id(), network0.get_id());
    EXPECT_EQ(output_memory1.get_net_id(), network1.get_id());
    auto wmem0 = network0.get_output_memory("weights");
    auto wmem1 = network1.get_output_memory("weights");

    ASSERT_EQ(wmem0, wmem1);

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr0[y * x_size + x]);
            EXPECT_EQ(output_vec[y][x], output_ptr1[y * x_size + x]);
        }
    }
}
