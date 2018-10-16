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
#include "api/CPP/concatenation.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(spatial_concatenate_f32_gpu, test01) {
    engine eng;

    memory input1 = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1,1,2,2 } });
    memory input2 = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1,1,2,2 } });

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        5.0f, 6.0f,
        7.0f, 8.0f
    });

    const auto expected_output = std::vector<float>{
        1.0f, 2.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 7.0f, 8.0f
    };

    topology tpl;
    tpl.add(input_layout("in1", input1.get_layout()));
    tpl.add(input_layout("in2", input2.get_layout()));
    tpl.add(concatenation("conc", { "in1", "in2" }, concatenation::along_x));

    network net(eng, tpl);
    net.set_input_data("in1", input1);
    net.set_input_data("in2", input2);

    auto outputs = net.execute();
    ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

    auto output_mem = outputs.at("conc").get_memory();
    auto output_layout = output_mem.get_layout();

    ASSERT_EQ(output_layout.size.batch[0], input1.get_layout().size.batch[0]);
    ASSERT_EQ(output_layout.size.feature[0], input1.get_layout().size.feature[0]);
    ASSERT_EQ(output_layout.size.spatial[1], input1.get_layout().size.spatial[1]);
    ASSERT_EQ(output_layout.size.spatial[0], input1.get_layout().size.spatial[0] + input2.get_layout().size.spatial[0]);

    ASSERT_EQ(output_mem.get_layout().get_linear_size(), expected_output.size());
    {
        auto out_ptr = output_mem.pointer<const float>();

        size_t idx = 0;
        for (auto const& value : out_ptr)
        {
            EXPECT_FLOAT_EQ(value, expected_output[idx++]);
        }
    }
}

TEST(spatial_concatenate_f32_gpu, test02) {
    engine eng;

    memory input1 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 } });
    memory input2 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 } });

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        5.0f, 6.0f,
        7.0f, 8.0f
    });

    const auto expected_output = std::vector<float>{
        1.0f, 2.0f, 
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    topology tpl;
    tpl.add(input_layout("in1", input1.get_layout()));
    tpl.add(input_layout("in2", input2.get_layout()));
    tpl.add(concatenation("conc", { "in1", "in2" }, concatenation::along_y));

    network net(eng, tpl);
    net.set_input_data("in1", input1);
    net.set_input_data("in2", input2);

    auto outputs = net.execute();
    ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

    auto output_mem = outputs.at("conc").get_memory();
    auto output_layout = output_mem.get_layout();

    ASSERT_EQ(output_layout.size.batch[0], input1.get_layout().size.batch[0]);
    ASSERT_EQ(output_layout.size.feature[0], input1.get_layout().size.feature[0]);
    ASSERT_EQ(output_layout.size.spatial[0], input1.get_layout().size.spatial[0]);
    ASSERT_EQ(output_layout.size.spatial[1], input1.get_layout().size.spatial[1] + input2.get_layout().size.spatial[1]);

    ASSERT_EQ(output_mem.get_layout().get_linear_size(), expected_output.size());
    {
        auto out_ptr = output_mem.pointer<const float>();

        size_t idx = 0;
        for (auto const& value : out_ptr)
        {
            EXPECT_FLOAT_EQ(value, expected_output[idx++]);
        }
    }
}

TEST(spatial_concatenate_f32_gpu, test03) {
    engine eng;

    memory input1 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 } });
    memory input2 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 } });

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        5.0f, 6.0f,
        7.0f, 8.0f
    });

    const auto expected_output = std::vector<float>{
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 2.0f, 0.0f,
        0.0f, 3.0f, 4.0f, 0.0f,
        0.0f, 5.0f, 6.0f, 0.0f,
        0.0f, 7.0f, 8.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    topology tpl;
    tpl.add(input_layout("in1", input1.get_layout()));
    tpl.add(input_layout("in2", input2.get_layout()));
    tpl.add(concatenation("conc", { "in1", "in2" }, concatenation::along_y, padding({ 0, 0, 1, 1 }, 0.0f)));

    network net(eng, tpl);
    net.set_input_data("in1", input1);
    net.set_input_data("in2", input2);

    auto outputs = net.execute();
    ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

    auto output_mem = outputs.at("conc").get_memory();
    auto output_layout = output_mem.get_layout();

    ASSERT_EQ(output_layout.size.batch[0], input1.get_layout().size.batch[0]);
    ASSERT_EQ(output_layout.size.feature[0], input1.get_layout().size.feature[0]);
    ASSERT_EQ(output_layout.size.spatial[0], input1.get_layout().size.spatial[0]);
    ASSERT_EQ(output_layout.size.spatial[1], input1.get_layout().size.spatial[1] + input2.get_layout().size.spatial[1]);

    ASSERT_EQ(output_mem.get_layout().get_linear_size(), expected_output.size());
    {
        auto out_ptr = output_mem.pointer<const float>();

        size_t idx = 0;
        for (auto const& value : out_ptr)
        {
            EXPECT_FLOAT_EQ(value, expected_output[idx++]);
        }
    }
}

TEST(spatial_concatenate_f32_gpu, test04) {
    engine eng;

    memory input1 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 }, padding({ 0,0,0,0 }, { 0,0,1,0 }) });
    memory input2 = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1,1,2,2 }, padding({ 0,0,0,1 }, 0.0f) });

    set_values(input1, {
        1.0f, 2.0f, 0.0f,
        3.0f, 4.0f, 0.0f
    });

    set_values(input2, {
        0.0f, 0.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
        0.0f, 0.0f
    });

    const auto expected_output = std::vector<float>{
        0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 6.0f,
        0.0f, 0.0f, 3.0f, 4.0f, 7.0f, 8.0f
    };

    topology tpl;
    tpl.add(input_layout("in1", input1.get_layout()));
    tpl.add(input_layout("in2", input2.get_layout()));
    tpl.add(concatenation("conc", { "in1", "in2" }, concatenation::along_x, padding({ 0,0,2,0 }, { 0,0,0,0 })));

    network net(eng, tpl);
    net.set_input_data("in1", input1);
    net.set_input_data("in2", input2);

    auto outputs = net.execute();
    ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

    auto output_mem = outputs.at("conc").get_memory();
    auto output_layout = output_mem.get_layout();

    ASSERT_EQ(output_layout.size.batch[0], input1.get_layout().size.batch[0]);
    ASSERT_EQ(output_layout.size.feature[0], input1.get_layout().size.feature[0]);
    ASSERT_EQ(output_layout.size.spatial[1], input1.get_layout().size.spatial[1]);
    ASSERT_EQ(output_layout.size.spatial[0], input1.get_layout().size.spatial[0] + input2.get_layout().size.spatial[0]);

    ASSERT_EQ(output_mem.get_layout().get_linear_size(), expected_output.size());
    {
        auto out_ptr = output_mem.pointer<const float>();

        size_t idx = 0;
        for (auto const& value : out_ptr)
        {
            EXPECT_FLOAT_EQ(value, expected_output[idx++]);
        }
    }
}