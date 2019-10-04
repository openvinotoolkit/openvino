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

#include <gtest/gtest.h>
#include "api/engine.hpp"
#include "test_utils/test_utils.h"
#include "api/input_layout.hpp"
#include "api/network.hpp"

using namespace tests;
using namespace cldnn;

TEST(events_pool, DISABLED_basic_test)
{
    /*
    This tests if the events pool works and there's no memory leak.
    */
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    topology topology;
    topology.add(input_layout("input", { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num))}}));
    topology.add(activation("relu", "input", activation_func::relu));
    topology.add(activation("relu1", "relu", activation_func::relu));
    topology.add(activation("relu2", "relu1", activation_func::relu));
    topology.add(activation("relu3", "relu2", activation_func::relu));
    topology.add(activation("relu4", "relu3", activation_func::relu));
    topology.add(activation("relu5", "relu4", activation_func::relu));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    for (int i = 0; i < 20; i++)
    {
        engine eng;// here we build new engine i times
        auto input = memory::allocate(eng, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });
        std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
        for (int j = 0; j < 20; j++) //then we build network j times
        {
            network network(eng, topology, bo);
            network.set_input_data("input", input);
            for(int k = 0; k < 20; k++) //and execute that network k times
                network.execute();  
        }
        EXPECT_EQ(eng.get_max_used_device_memory_size(), (uint64_t)80);
        eng.~engine();
    }
}