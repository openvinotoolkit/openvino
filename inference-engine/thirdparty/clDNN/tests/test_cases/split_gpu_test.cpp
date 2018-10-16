/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/CPP/split.hpp"
#include "api/CPP/scale.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t y, size_t x, int min, int max) {
    static std::default_random_engine generator(random_seed);
    int k = 8; // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(k * min, k * max);
    std::vector<T> v(b*f*x*y);
    for (size_t i = 0; i < b*f*x*y; ++i) {
        v[i] = (T)distribution(generator);
        v[i] /= k;
    }
    return v;
}

template<typename T>
void check_feature_map(cldnn::pointer<T> output_ptr, std::vector<T> &input_vec, size_t batch_num, size_t feature_num, size_t y_size, size_t x_size, size_t feature_id, size_t factor)
{
    for (size_t b = 0; b < batch_num; ++b) { //B
        for (size_t y = 0; y < y_size; ++y) { //Y
            for (size_t x = 0; x < x_size; ++x) { //X
                auto linear_id = x + x_size * (y + y_size * (feature_id + feature_num * b));
                auto output_linear_id = x + x_size * (y + y_size * b);
                EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id] * factor);
            }
        }
    }
}

TEST(split_gpu, basic_in2x3x2x2_split_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }

    engine engine;

    auto batch_num = 6;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 3;

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(split("split", "input",
    {
        { "out0", { 0, 0, 0, 0 } },
        { "out1", { 0, 1, 0, 0 } },
        { "out2", { 0, 2, 0, 0 } }
    } ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(3));

    for (unsigned int i = 0; i < 3; i++)
    {
        auto split_id = "split:out" + std::to_string(i);
        auto output = outputs.at(split_id).get_memory();
        auto output_ptr = output.pointer<float>();
        check_feature_map<float>(output_ptr, input_vec, batch_num, feature_num, y_size, x_size, i, 1);
    }
}

TEST(split_gpu, basic_in2x3x2x2_split_scale_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }
    //  Additional scale layer at the end

    engine engine;

    auto batch_num = 6;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 3;

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input0 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_input1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input0", scale_input0.get_layout()));
    topology.add(input_layout("scale_input1", scale_input1.get_layout()));
    topology.add(input_layout("scale_input2", scale_input2.get_layout()));
    topology.add(split("split", "input",
    {
        { "out0",{ 0, 0, 0, 0 } },
        { "out1",{ 0, 1, 0, 0 } },
        { "out2",{ 0, 2, 0, 0 } }
    }));
    topology.add(scale("scale0", "split:out0", "scale_input0"));
    topology.add(scale("scale1", "split:out1", "scale_input1"));
    topology.add(scale("scale2", "split:out2", "scale_input2"));

    std::vector<float> scale_input_vec0 = { 1.f };
    set_values(scale_input0, scale_input_vec0);
    std::vector<float> scale_input_vec1 = { 2.f };
    set_values(scale_input1, scale_input_vec1);
    std::vector<float> scale_input_vec2 = { 3.f };
    set_values(scale_input2, scale_input_vec2);

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input0", scale_input0);
    network.set_input_data("scale_input1", scale_input1);
    network.set_input_data("scale_input2", scale_input2);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(3));

    for (unsigned int i = 0; i < 3; i++)
    {
        auto split_id = "scale" + std::to_string(i);
        auto output = outputs.at(split_id).get_memory();
        auto output_ptr = output.pointer<float>();
        check_feature_map<float>(output_ptr, input_vec, batch_num, feature_num, y_size, x_size, i, i + 1);
    }
}