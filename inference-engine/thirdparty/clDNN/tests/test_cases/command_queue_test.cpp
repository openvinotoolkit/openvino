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
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/input_layout.hpp>
#include "test_utils/test_utils.h"
#include "api/arg_max_min.hpp"

using namespace cldnn;
using namespace tests;
using namespace std;

// Run some topology too see if command queue does work correctly
// Coppied from arg_max_gpu.base test.
void exexute_network(cldnn::engine engine)
{
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max));

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
    EXPECT_EQ(outputs.begin()->first, "arg_max");

    auto output = outputs.at("arg_max").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[batch_num];
    for (uint32_t i = 0; i < batch_num; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    int size = x_size * y_size * feature_num;
    int index;
    float value;
    for (int i = 0; i < batch_num; i++) {
        EXPECT_GE(out_buffer[i], 0);
        EXPECT_LT(out_buffer[i], size);
        index = (int)out_buffer[i];
        value = input_vec[i*size + (int)index];
        for (int j = 0; j < size; j++)
        {
            EXPECT_LE(input_vec[i*size + j], value);
        }
    }
}

TEST(command_queue_test, test_priority_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            false,          // decorate_kernel_names
            false,          // dump_custom_program
            "",             // options
            "",             // single_kernel
            true,           // primitives_parallelisation
            "",             // engine_log
            "",             // sources_dumps_dir
            priority_mode_types::low,
            throttle_mode_types::disabled);
    cldnn::engine engine(configuration);
    exexute_network(engine);
}

TEST(command_queue_test, test_throttle_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            false,          // decorate_kernel_names
            false,          // dump_custom_program
            "",             // options
            "",             // single_kernel
            true,           // primitives_parallelisation
            "",             // engine_log
            "",             // sources_dumps_dir
            priority_mode_types::disabled,
            throttle_mode_types::high);
    cldnn::engine engine(configuration);
    exexute_network(engine);
}

TEST(command_queue_test, test_priority_and_throttle_hints) {
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            false,          // decorate_kernel_names
            false,          // dump_custom_program
            "",             // options
            "",             // single_kernel
            true,           // primitives_parallelisation
            "",             // engine_log
            "",             // sources_dumps_dir
            priority_mode_types::high,
            throttle_mode_types::low);
    cldnn::engine engine(configuration);
    exexute_network(engine);
}