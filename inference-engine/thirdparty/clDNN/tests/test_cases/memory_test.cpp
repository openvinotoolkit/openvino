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
#include <api/CPP/engine.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/data.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

#if 0
TEST(memory_tests, DISABLED_execution_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    network net(eng, tpl);
    
    while (true)
    {
        net.set_input_data("in", in);
        net.execute();
    }
}

TEST(memory_tests, DISABLED_network_creation_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    while (true)
    {
        network net(eng, tpl);
    }
}
#endif
TEST(memory_pool, basic_non_padded_relu_pipe) {
    // 5 relu's of size 1x4x1x1
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(activation("relu2", "relu1", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu3", activation_relu));
    topology.add(activation("relu5", "relu4", activation_relu));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t) 80);
 }


TEST(memory_pool, basic_non_padded_relu_and_pooling_pipe) {
    // uncomment this line to disable memory pool
    /*engine_configuration cfg{ false, false, false, std::string(), std::string(), true, std::string(),std::string(), 0, false };
    engine engine{ cfg };*/
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(pooling("pool1", "relu1",pooling_mode::max, { 1,1,3,3 }, { 1,1,2,2 }));
    topology.add(activation("relu2", "pool1", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu3", activation_relu));
    topology.add(activation("relu5", "relu4", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)1088);
}


TEST(memory_pool, multi_outputs_network) {
    //            -- relu -- relu1 -- relu4
    //     input<           
    //            -- relu2 --  relu3 -- relu5--relu6--relu7
    // neither of relu5, relu6 nor relu7 can share resource with relu4. 

    // uncomment this line to disable memory pool
    /*engine_configuration cfg{ false, false, false, std::string(), std::string(), true, std::string(),std::string(), 0, false };
    engine engine{ cfg };*/
    engine engine;
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_relu));
    topology.add(activation("relu1", "relu", activation_relu));
    topology.add(activation("relu2", "input", activation_relu));
    topology.add(activation("relu3", "relu2", activation_relu));
    topology.add(activation("relu4", "relu1", activation_relu));
    topology.add(activation("relu5", "relu3", activation_relu));
    topology.add(activation("relu6", "relu5", activation_relu));
    topology.add(activation("relu7", "relu6", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)2048);
}


TEST(memory_pool, oooq) {
    /*          -- relu1 - concat1- relu4 -- 
        input<  -- relu2 |                   >-- concat2 -- relu6
                -- relu3 --  relu5 --------- 
       neither of relu5, relu6 nor relu7 can share resource with relu4. */

    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), priority_mode_types::disabled, throttle_mode_types::disabled, true /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 4;
    auto y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu1", "input", activation_relu));
    topology.add(activation("relu2", "input", activation_relu));
    topology.add(activation("relu3", "input", activation_relu));
    topology.add(concatenation("concat1", { "relu1", "relu2"},concatenation::along_f));
    topology.add(activation("relu4", "concat1", activation_relu));
    topology.add(activation("relu5", "relu3", activation_relu));
    topology.add(concatenation("concat2", { "relu4", "relu5" }, concatenation::along_f));
    topology.add(activation("relu6", "concat2", activation_relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t) 2816);
}

TEST(memory_pool, shared_mem_pool_same_topology_twice) {
    /*          -- relu1 - concat1- relu4 --
    input<  -- relu2 |                   >-- concat2 -- relu6
    -- relu3 --  relu5 ---------
    neither of relu5, relu6 nor relu7 can share resource with relu4. */

    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), priority_mode_types::disabled, throttle_mode_types::disabled, true /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_num = 1;
    auto feature_num = 4;
    auto inp_x_size = 4;
    auto inp_y_size = 4;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_num)) } });

    set_values(input, 
    {   1.0f, 2.5f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 6.1f, 4.7f, 1.0f, 1.0f, 8.2f, 1.0f, 2.0f, 1.0f,
        5.0f, 2.0f, 2.0f, 3.0f, 5.0f, 2.0f, 2.0f, 3.0f, 1.1f, 2.4f, 1.0f, 1.0f, 4.0f, 6.0f, 3.0f, 3.6f,
        4.0f, 6.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.5f, 1.0f, 4.0f, 6.5f, 3.0f, 3.0f, 4.0f, 6.0f, 1.8f, 3.5f,
        3.0f, 5.0f, 1.0f, 1.0f, 1.3f, 1.0f, 0.4f, 1.3f, 4.0f, 7.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.9f, 4.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu1", "input", activation_relu));
    topology.add(activation("relu2", "input", activation_sqrt));
    topology.add(activation("relu3", "input", activation_square));
    topology.add(concatenation("concat1", { "relu1", "relu2" }, concatenation::along_f));
    topology.add(activation("relu4", "concat1", activation_relu));
    topology.add(activation("relu5", "relu3", activation_relu));
    topology.add(concatenation("concat2", { "relu4", "relu5" }, concatenation::along_f));
    topology.add(activation("relu6", "concat2", activation_linear, {1.0f, 0.5f}));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network_first(engine, topology, bo);
    network_first.set_input_data("input", input);
    auto outputs = network_first.execute();

    auto output_memory_first = outputs.at("relu6").get_memory();
    auto output_layout_first = output_memory_first.get_layout();
    auto output_ptr_first = output_memory_first.pointer<float>();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)2816);

    network network_second(engine, topology, bo);
    network_second.set_input_data("input", input);
    auto outputs_second = network_second.execute();

    auto output_memory_second = outputs_second.at("relu6").get_memory();
    auto output_layout_second = output_memory_second.get_layout();
    auto output_ptr_second = output_memory_second.pointer<float>();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)3584);
    EXPECT_EQ(output_layout_first, output_layout_second);

    int y_size = output_layout_first.size.spatial[1];
    int x_size = output_layout_first.size.spatial[0];
    int f_size = output_layout_first.size.feature[0];
    int b_size = output_layout_first.size.batch[0];
    int f_offset = y_size*x_size;
    int b_offset = f_size * f_offset;
    for (int b = 0; b < b_size; ++b)
    {
        for (int f = 0; f < f_size; ++f)
        {
            for (int y = 0; y < y_size; ++y) 
            {
                for (int x = 0; x < x_size; ++x) 
                {
                    int idx = b * b_offset + f * f_offset + y * x_size + x;
                    EXPECT_EQ(output_ptr_first[idx], output_ptr_second[idx]);
                }
            }
        }
    } 
}

TEST(memory_pool, shared_mem_pool_same_topology_twice_weights) {

    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), priority_mode_types::disabled, throttle_mode_types::disabled, true /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_num = 1;
    auto feature_num = 3;
    auto inp_x_size = 4;
    auto inp_y_size = 4;

    auto input= memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_num)) } });
    auto weights = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });
    
    std::vector<float> dummy_input_data_1 = {
       /*f0 xy*/ 0.8f, 0.65f, 0.1f, 1.0f, 1.0f, 0.5f, 0.11f, 0.33f, 0.66f, 0.11f, 0.22f, 0.33f, 0.99f, 0.8f, 0.7f, 0.5f,
       /*f1 xy*/ 0.48f, 0.05f, 0.35f, 1.0f, 1.0f, 0.51f, 0.51f, 0.13f, 0.86f, 0.10f, 0.29f, 0.53f, 0.99f, 0.4f, 0.3f, 0.1f,
       /*f2 xy*/ 0.98f, 0.35f, 0.3f, 0.01f, 0.9f, 0.55f, 0.15f, 0.39f, 0.36f, 0.01f, 0.32f, 0.4f, 0.3f, 0.2f, 0.1f, 0.5f,
    };

    set_values(input, dummy_input_data_1);
    set_values(weights, { 0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }, { 1, 1, 1, 2 }),
        softmax("softmax", "conv"));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network_first(engine, topology, bo);
    network_first.set_input_data("input", input);
    auto outputs = network_first.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)824);

    auto output_memory_first = outputs.at("softmax").get_memory();
    auto output_layout_first = output_memory_first.get_layout();
    auto output_ptr_first = output_memory_first.pointer<float>();

    network network_second(engine, topology, bo);
    network_second.set_input_data("input", input);
    auto outputs_second = network_second.execute();

    auto output_memory_second = outputs_second.at("softmax").get_memory();
    auto output_layout_second = output_memory_second.get_layout();
    auto output_ptr_second = output_memory_second.pointer<float>();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)1224);
    EXPECT_EQ(output_layout_first, output_layout_second);

    int y_size = output_layout_first.size.spatial[1];
    int x_size = output_layout_first.size.spatial[0];
    int f_size = output_layout_first.size.feature[0];
    int b_size = output_layout_first.size.batch[0];
    int f_offset = y_size * x_size;
    int b_offset = f_size * f_offset;
    for (int b = 0; b < b_size; ++b)
    {
        for (int f = 0; f < f_size; ++f)
        {
            for (int y = 0; y < y_size; ++y)
            {
                for (int x = 0; x < x_size; ++x)
                {
                    int idx = b * b_offset + f * f_offset + y * x_size + x;
                    EXPECT_EQ(output_ptr_first[idx], output_ptr_second[idx]);
                }
            }
        }
    }
}


TEST(memory_pool, shared_mem_pool_diff_batches) {

    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), priority_mode_types::disabled, throttle_mode_types::disabled, true /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_8 = 8;
    auto batch_1 = 1;
    auto feature_num = 3;
    auto inp_x_size = 4;
    auto inp_y_size = 4;
    auto dt = data_types::f32;
    auto fmt = format::bfyx;
    layout lay_batch_1 = { dt, fmt, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_1)) }};
    layout lay_batch_8 = { dt, fmt, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_8)) }};
    auto input_1 = memory::allocate(engine, lay_batch_1);
    auto input_8 = memory::allocate(engine, lay_batch_8);
    auto weights = memory::allocate(engine, { dt, fmt, { 1, 1, 3, 2 } });

    std::vector<float> dummy_input_data_1 = generate_random_1d<float>(batch_1*feature_num*inp_x_size*inp_y_size, 0, 1);
    std::vector<float> dummy_input_data_8 = generate_random_1d<float>(batch_8*feature_num*inp_x_size*inp_y_size, 0, 1);

    set_values(input_1, dummy_input_data_1);
    set_values(input_8, dummy_input_data_8);
    set_values(weights, { 0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f });

    topology topo(
        input_layout("input", input_8.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }, { 1, 1, 1, 2 }),
        softmax("softmax", "conv"));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network_first(engine, topo, bo);
    network_first.set_input_data("input", input_8);
    auto outputs = network_first.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)3928);

    topo.change_input_layout("input", input_1.get_layout());//change input layout to batch=1

    network network_second(engine, topo, bo);
    network_second.set_input_data("input", input_1);
    auto outputs_second = network_second.execute();

    EXPECT_EQ(engine.get_max_used_device_memory_size(), (uint64_t)3928);
}