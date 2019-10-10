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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>

#include <api/data.hpp>
#include <api/reshape.hpp>
#include <api/input_layout.hpp>
#include <api/shuffle_channels.hpp>
#include <api/strided_slice.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(removing_output_node, multiple_outputs) {
    // Tests split with crop implementation
    //                                                   _ strided_slice(bfyx)
    //                                                  |
    //  INPUT(bfyx,3x2x1x1)--shuffle_channels(bfyx)-----|
    //                                                  |_
    //                                                     reshape(bfyx);

    const auto& engine = get_test_engine();
    auto batch_num = 6;
    auto feature_num = 1;
    auto x_size = 1;
    auto y_size = 1;
    int32_t axis = 0;
    int32_t group = 2;

    tensor initial_shape = tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num));
    tensor after_strided_slice = tensor(spatial(y_size, feature_num), feature(batch_num), batch(1));
    tensor after_reshape = tensor(feature(batch_num * feature_num * y_size * x_size));

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, initial_shape });
    auto begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(begin, {
            1, 0, 1, 0
    });
    set_values(end, {
            2, 2, 4, 4
    });
    set_values(strides, {
            1, 1, 1, 2
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(shuffle_channels("shuffle_channels", "input", group, axis));
    topology.add(reshape("reshape", "shuffle_channels", after_reshape));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", "shuffle_channels", "input2", "input3", "input4", {}, {}, { 1 }, {}));

    std::vector<float> input_vec = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> out_vec = { 0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f };
    set_values(input, input_vec);

    build_options bo;
    bo.set_option(build_option::outputs({ "shuffle_channels", "reshape", "strided_slice" }));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("reshape").get_memory();
    auto output_ptr = output.pointer<float>();

    ASSERT_TRUE(output.get_layout().size == after_reshape);

    for (size_t i = 0; i < out_vec.size(); i++)
        EXPECT_EQ(output_ptr[i], out_vec[i]);

    // checking the output node has the same name after output node deleting due to StridedSlice optimization
    ASSERT_TRUE(outputs.find("strided_slice") != outputs.end());
    auto output2 = outputs.at("strided_slice").get_memory();
    auto output_ptr2 = output.pointer<float>();

    ASSERT_TRUE(output2.get_layout().size == after_strided_slice);

    for (size_t i = 0; i < out_vec.size(); i++)
        EXPECT_EQ(output_ptr2[i], out_vec[i]);
}

TEST(removing_output_node, output_node_optimization) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    //  21  28  39
    //  18  20  20

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    VVF<float> output_vec = {
            { 20.0f, 27.0f, 38.0f },
            { 17.0f, 19.0f, 19.0f } };

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", "input", { "weights" }, { 1,1,1,2 }));
    topology.add(activation("relu", "conv", activation_func::relu));

    network network(engine, topology);
    network.set_input_data("input", input);

    // checking the output node has the same name after output node deleting due to ReLU optimization
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

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
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}
