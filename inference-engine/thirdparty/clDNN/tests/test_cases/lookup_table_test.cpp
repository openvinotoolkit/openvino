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

#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/lookup_table.hpp"
#include "api/CPP/arg_max_min.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace std;
using namespace tests;


TEST(lookup_table_base, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2;
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch_num, feature_num, x_size , y_size } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, {2, 1, 1, 1} });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(lookup_table("table", "input", "input2"));
    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    vector<float> input2_vec = { 11, 3 };
    set_values(input, input_vec);
    set_values(input2, input2_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.at("table").get_memory();
    auto output_ptr = output.pointer<float>();;
    float out_buffer[batch_num];
    for (uint32_t i = 0; i < batch_num; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    int size = x_size * y_size * feature_num;
    float value;
    for (int i = 0; i < batch_num; i++) {
        value = out_buffer[i];
        for (int j = 0; j < size; j++)
        {
            EXPECT_LE(input_vec[i*size + j], value);
        }
    }
}

TEST(lookup_table_num, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2, number_of_values = 3;
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 3, 1 } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(lookup_table("table", "input", "input2"));
    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    vector<float> input2_vec = { 11, 7, 3, 3, 7, 6};
    set_values(input, input_vec);
    set_values(input2, input2_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.at("table").get_memory();
    auto output_ptr = output.pointer<float>();;
    float out_buffer[batch_num*number_of_values];
    for (uint32_t i = 0; i < batch_num * number_of_values; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    int size = x_size * y_size * feature_num;
    float value;
    for (int i = 0; i < batch_num; i++) {
        int count = 0;
        int amount = 0;
        int same_values = 1;
        int j;
        for (j = 0; j < number_of_values; j++) {
            if (number_of_values - 1 == j) {
                if (input_vec[i*size + (int)input2_vec[i*number_of_values + j]] != input_vec[i*size + (int)input2_vec[i*number_of_values + j - 1]]) {
                    amount += j;
                }
                else
                    amount += same_values * (j - same_values + 1);
            }
            else if (input_vec[i*size + (int)input2_vec[i*number_of_values + j]] != input_vec[i*size + (int)input2_vec[i*number_of_values + j + 1]]) {
                if (same_values != j + 1) {
                    amount += same_values * (j - same_values + 1);
                    same_values = 1;
                }
            }
            else
                same_values++;
        }
        for (int j = 0; j < number_of_values; j++)
        {
            value = out_buffer[i*number_of_values + j];
            for (int k = 0; k < size; k++)
            {
                if (input_vec[i*size + k] > value)
                    count++;
            }
        }
        EXPECT_EQ(count, amount);
    }
}

TEST(lookup_table_with_arg_max, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2;
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max));
    topology.add(lookup_table("table", "input", "arg_max"));
    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.at("table").get_memory();
    auto output_ptr = output.pointer<float>();;
    float out_buffer[batch_num];
    for (uint32_t i = 0; i < batch_num; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    int size = x_size * y_size * feature_num;
    float value;
    for (int i = 0; i < batch_num; i++) {
        value = out_buffer[i];
        for (int j = 0; j < size; j++)
        {
            EXPECT_LE(input_vec[i*size + j], value);
        }
    }
}

TEST(lookup_table_axis, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2, number_of_values = 2;
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 3, 2, 2 } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(lookup_table("table", "input", "input2", lookup_table::batch));
    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    vector<float> input2_vec = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    set_values(input, input_vec);
    set_values(input2, input2_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));

    auto output = outputs.at("table").get_memory();
    auto output_ptr = output.pointer<float>();;
    const int out_size = y_size * feature_num * x_size * number_of_values;
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], (i%2==0 ? input_vec[i/2] : input_vec[(i/2+12)]));
    }
}