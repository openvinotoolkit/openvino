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
#include <api/CPP/reorder.hpp>
#include "test_utils/test_utils.h"

#include <sstream>
#include <iomanip>

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

template<typename T>
void split_test(int batch_num, int feature_num, int x_size, int y_size, std::vector<cldnn::tensor> split_offsets)
{
    const auto& engine = get_test_engine();
    cldnn::tensor reference_input_size = { batch_num, feature_num, x_size, y_size };

    cldnn::memory input = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, reference_input_size });
    std::vector<std::pair<primitive_id, cldnn::tensor> > input_ids_offsets;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    
    // lambda exoression to create the primitive id for the splits
    auto create_split_id = [](size_t splitNum) {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << splitNum;

        return ss.str();
    };

    // Create the splits with the split ids for the topology
    for (size_t splitNum = 0; splitNum < split_offsets.size(); splitNum++) 
    {
        input_ids_offsets.push_back({ create_split_id(splitNum), split_offsets[splitNum]});
    }

    topology.add(split("split", "input", input_ids_offsets));

    std::vector<T> input_vec = generate_random_input<T>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    // The number of splits should match the expected number of splits
    EXPECT_EQ(outputs.size(), size_t(split_offsets.size()));
    
    std::vector<cldnn::tensor> expected_sizes;
    for (size_t splitNum = 0; splitNum < split_offsets.size(); splitNum++)  // Calculate the expected sizes
    {
        cldnn::tensor size;

        if (splitNum < (split_offsets.size() - 1))
        {
            size = split_offsets[splitNum + 1] - split_offsets[splitNum];
        }
        else
        {
            size = reference_input_size - split_offsets[splitNum];
        }

        // For all the other dimensions, copy from the split_input
        for (int dimension = 0; dimension < CLDNN_TENSOR_DIM_MAX; dimension++)
        {
            size.raw[dimension]
                = (size.raw[dimension] == 0) ? reference_input_size.raw[dimension] : size.raw[dimension];
        }

        expected_sizes.push_back(size);
    }

    pointer<T> input_ptr = input.pointer<T>();

    for (size_t splitNum = 0; splitNum < split_offsets.size(); splitNum++)
    {
        primitive_id split_id = "split:" + create_split_id(splitNum);
        cldnn::memory output = outputs.at(split_id).get_memory();
        auto prim = output.get_layout();
        EXPECT_EQ(prim.size, expected_sizes[splitNum]);
        auto output_ptr = output.pointer<T>();

        // Output tensor size
        auto output_batch = prim.size.batch[0];
        auto output_feature = prim.size.feature[0];
        auto output_x = prim.size.spatial[0];
        auto output_y = prim.size.spatial[1];

        // Input offsets, starting from which we will compare the output
        auto input_batch_offset = split_offsets[splitNum].batch[0];
        auto input_feature_offset = split_offsets[splitNum].feature[0];
        auto input_y_offset = split_offsets[splitNum].spatial[1];
        auto input_x_offset = split_offsets[splitNum].spatial[0];
        
        // iterator to iterate through input buffer
        auto input_batch_itr = input_batch_offset;
        auto input_feature_itr = input_feature_offset;
        auto input_y_itr = input_y_offset;
        auto input_x_itr = input_x_offset;
        
        for (auto b = 0; b < output_batch; ++b) {  // B
            
                // reset the input feature iterator
            input_feature_itr = input_feature_offset; 
            for (auto f = 0; f < output_feature; f++) {  // F
                
                // reset the input y iterator
                input_y_itr = input_y_offset;  
                for (auto y = 0; y < output_y; y++) {  // Y
                    
                    // reset the input x iterator
                    input_x_itr = input_x_offset;  
                    for (auto x = 0; x < output_x; x++) {  // X
                        auto linear_id = input_x_itr + x_size * (input_y_itr + y_size * (input_feature_itr + feature_num * input_batch_itr)); // index in input
                        auto output_linear_id = x + output_x * (y + output_y * (f + output_feature * b)); // index in output
                        EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                        input_x_itr++;  // update the input x iterator
                    }
                    input_y_itr++;  // update the input y iterator
                }
                input_feature_itr++;  // update the input feature iterator
            }
            input_batch_itr++;  // update the input batch iterator
        }
    }
}

TEST(split_gpu, split_1d_uneven_2_splits) {

    //  Input      : 2x4x3x3
    //  Output1    : 2x1x3x3
    //  Output2    : 2x3x3x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }

    auto batch_num = 2;
    auto feature_num = 4;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0}                                                
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}


TEST(split_gpu, basic_split_concat_optimization) {

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 25, 1, 256 } });
    tests::set_random_values<float>(input);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    std::vector<std::pair<primitive_id, tensor>> offsets;
    std::vector<primitive_id> ids;
    for (int i = 0; i < 25; i++)
    {
        auto id = "crop_" + std::to_string(i);
        ids.push_back("split:" + id);
        offsets.push_back({ id, {0, i, 0, 0} });
    }

    topology.add(split("split", "input", offsets));
    topology.add(concatenation("concat", ids, concatenation::along_f));
    topology.add(reorder("output", "concat", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    auto output_ptr = output.pointer<float>();
    auto input_ptr = input.pointer<float>();

    for (int i = 0; i < 25*256; ++i)
    {
        EXPECT_EQ(output_ptr[i], input_ptr[i]);
    }
}

TEST(split_gpu, split_1d_uneven_3_splits) {

    //  Input      : 2x8x3x3
    //  Output1    : 2x1x3x3
    //  Output2    : 2x3x3x3
    //  Output3    : 2x4x3x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 4, 0, 0 }

    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0},
                                                {0, 4, 0, 0},
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}

TEST(split_gpu, split_2d_uneven_2_splits) {

    //  Input      : 2x8x10x3
    //  Output1    : 2x1x4x3
    //  Output2    : 2x3x6x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 4, 0 }

    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}

TEST(split_gpu, split_2d_uneven_3_split3) {

    //  Input      : 2x8x10x3
    //  Output1    : 2x1x4x3
    //  Output2    : 2x3x3x3
    //  Output3    : 2x4x3x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 4, 0 }
    //  id: "out2", offsets: { 0, 4, 7, 0 }

    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0},
                                                {0, 4, 7, 0},
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}

TEST(split_gpu, split_3d_uneven_2_splits) {

    //  Input      : 2x8x10x3
    //  Output1    : 2x1x4x1
    //  Output2    : 2x7x6x2
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 4, 1 }

    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}

TEST(split_gpu, split_3d_uneven_3_splits) {

    //  Input      : 2x8x10x5
    //  Output1    : 2x1x4x1
    //  Output2    : 2x6x4x1
    //  Output3    : 2x1x2x1
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 4, 1 }
    //  id: "out2", offsets: { 0, 7, 8, 2 }

    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1},
                                                {0, 7, 8, 2}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets);
}

TEST(split_gpu, basic_in2x3x2x2_split_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }

    const auto& engine = get_test_engine();

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

    const auto& engine = get_test_engine();

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
