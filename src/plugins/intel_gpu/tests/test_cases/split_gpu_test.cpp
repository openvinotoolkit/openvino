// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/split.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

#include <sstream>
#include <iomanip>

using namespace cldnn;
using namespace ::tests;

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
void check_feature_map(T* output_ptr, std::vector<T> &input_vec, size_t batch_num, size_t feature_num, size_t y_size, size_t x_size, size_t feature_id, size_t factor)
{
    for (size_t b = 0; b < batch_num; ++b) { //B
        for (size_t y = 0; y < y_size; ++y) { //Y
            for (size_t x = 0; x < x_size; ++x) { //X
                auto linear_id = x + x_size * (y + y_size * (feature_id + feature_num * b));
                auto output_linear_id = x + x_size * (y + y_size * b);
                ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id] * factor);
            }
        }
    }
}

template<typename T>
void split_test(int batch_num, int feature_num, int x_size, int y_size, std::vector<cldnn::tensor> split_offsets,
                bool is_caching_test)
{
    auto& engine = get_test_engine();
    cldnn::tensor reference_input_size = { batch_num, feature_num, x_size, y_size };

    cldnn::memory::ptr input = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx, reference_input_size });
    std::vector<std::pair<primitive_id, cldnn::tensor> > input_ids_offsets;

    topology topology;
    topology.add(input_layout("input", input->get_layout()));

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

    topology.add(split("split", input_info("input"), input_ids_offsets));

    std::vector<T> input_vec = generate_random_input<T>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    cldnn::network::ptr network = get_network(engine, topology, ExecutionConfig(), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);

    auto outputs = network->execute();

    // The number of splits should match the expected number of splits
    ASSERT_EQ(outputs.size(), size_t(split_offsets.size()));

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
        for (int dimension = 0; dimension < cldnn::tensor_dim_max; dimension++)
        {
            size.raw[dimension]
                = (size.raw[dimension] == 0) ? reference_input_size.raw[dimension] : size.raw[dimension];
        }

        expected_sizes.push_back(size);
    }

    cldnn::mem_lock<T> input_ptr(input, get_test_stream());

    for (size_t splitNum = 0; splitNum < split_offsets.size(); splitNum++)
    {
        primitive_id split_id = "split:" + create_split_id(splitNum);
        cldnn::memory::ptr output = outputs.at(split_id).get_memory();
        auto prim = output->get_layout();
        ASSERT_EQ(prim.get_tensor(), expected_sizes[splitNum]);
        cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        // Output tensor size
        auto output_batch = prim.batch();
        auto output_feature = prim.feature();
        auto output_x = prim.spatial(0);
        auto output_y = prim.spatial(1);

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
                        ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
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

TEST(split_gpu_f32, split_1d_uneven_2_splits) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_1d_uneven_2_splits) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, basic_split_concat_optimization) {

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 25, 1, 256 } });
    tests::set_random_values<float>(input);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    std::vector<std::pair<primitive_id, tensor>> offsets;
    std::vector<input_info> inputs;
    for (int i = 0; i < 25; i++)
    {
        auto id = "crop_" + std::to_string(i);
        inputs.push_back(input_info("split:" + id));
        offsets.push_back({ id, {0, i, 0, 0} });
    }

    topology.add(split("split", input_info("input"), offsets));
    topology.add(concatenation("concat", inputs, 1));
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input, get_test_stream());

    for (int i = 0; i < 25*256; ++i)
    {
        ASSERT_EQ(output_ptr[i], input_ptr[i]);
    }
}

TEST(split_gpu_i64, basic_split_concat_optimization) {

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i64,format::bfyx,{ 1, 25, 1, 256 } });
    tests::set_random_values<int64_t>(input);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    std::vector<std::pair<primitive_id, tensor>> offsets;
    std::vector<input_info> inputs;
    for (int i = 0; i < 25; i++)
    {
        auto id = "crop_" + std::to_string(i);
        inputs.push_back(input_info("split:" + id));
        offsets.push_back({ id, {0, i, 0, 0} });
    }

    topology.add(split("split", input_info("input"), offsets));
    topology.add(concatenation("concat", inputs, 1));
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::i64));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    cldnn::mem_lock<int64_t> input_ptr(input, get_test_stream());

    for (int i = 0; i < 25*256; ++i)
    {
        ASSERT_EQ(output_ptr[i], input_ptr[i]);
    }
}

TEST(split_gpu_f32, split_1d_uneven_3_splits) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_1d_uneven_3_splits) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, split_2d_uneven_2_splits) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_2d_uneven_2_splits) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, split_2d_uneven_3_split3) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_2d_uneven_3_split3) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, split_3d_uneven_2_splits) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_3d_uneven_2_splits) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, split_3d_uneven_3_splits) {

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

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_i64, split_3d_uneven_3_splits) {

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

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, false);
}

TEST(split_gpu_f32, basic_in2x3x2x2_split_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 3;

    auto input = engine.allocate_memory({ data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(split("split", input_info("input"),
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

    ASSERT_EQ(outputs.size(), size_t(3));

    for (unsigned int i = 0; i < 3; i++)
    {
        auto split_id = "split:out" + std::to_string(i);
        auto output = outputs.at(split_id).get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());
        check_feature_map<float>(output_ptr.data(), input_vec, batch_num, feature_num, y_size, x_size, i, 1);
    }
}

TEST(split_gpu_i64, basic_in2x3x2x2_split_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 3;

    auto input = engine.allocate_memory({ data_types::i64,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(split("split", input_info("input"),
    {
        { "out0", { 0, 0, 0, 0 } },
        { "out1", { 0, 1, 0, 0 } },
        { "out2", { 0, 2, 0, 0 } }
    } ));

    std::vector<int64_t> input_vec = generate_random_input<int64_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(3));

    for (unsigned int i = 0; i < 3; i++)
    {
        auto split_id = "split:out" + std::to_string(i);
        auto output = outputs.at(split_id).get_memory();
        cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
        check_feature_map<int64_t>(output_ptr.data(), input_vec, batch_num, feature_num, y_size, x_size, i, 1);
    }
}

TEST(split_gpu_f32, basic_in2x3x2x2_split_scale_feature_bfyx) {
    //  Input      : 6x3x4x3
    //  3 x Outputs: 6x1x4x3
    //  Split params:
    //  id: "out0", offsets: { 0, 0, 0, 0 }
    //  id: "out1", offsets: { 0, 1, 0, 0 }
    //  id: "out2", offsets: { 0, 2, 0, 0 }
    //  Additional scale layer at the end

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 3;

    auto input = engine.allocate_memory({ data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input0 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_input1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_input2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("scale_input0", scale_input0->get_layout()));
    topology.add(input_layout("scale_input1", scale_input1->get_layout()));
    topology.add(input_layout("scale_input2", scale_input2->get_layout()));
    topology.add(split("split", input_info("input"),
    {
        { "out0",{ 0, 0, 0, 0 } },
        { "out1",{ 0, 1, 0, 0 } },
        { "out2",{ 0, 2, 0, 0 } }
    }));
    topology.add(eltwise("scale0", { input_info("split:out0"), input_info("scale_input0") }, eltwise_mode::prod));
    topology.add(eltwise("scale1", { input_info("split:out1"), input_info("scale_input1") }, eltwise_mode::prod));
    topology.add(eltwise("scale2", { input_info("split:out2"), input_info("scale_input2") }, eltwise_mode::prod));

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

    ASSERT_EQ(outputs.size(), size_t(3));

    for (unsigned int i = 0; i < 3; i++)
    {
        auto split_id = "scale" + std::to_string(i);
        auto output = outputs.at(split_id).get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());
        check_feature_map<float>(output_ptr.data(), input_vec, batch_num, feature_num, y_size, x_size, i, i + 1);
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(split_gpu_f32, split_1d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 4;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_i64, split_1d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 4;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0}
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_f32, split_1d_uneven_3_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0},
                                                {0, 4, 0, 0},
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_i64, split_1d_uneven_3_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 3;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 0, 0},
                                                {0, 4, 0, 0},
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_f32, split_2d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_i64, split_2d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0}
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_f32, split_2d_uneven_3_split3_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0},
                                                {0, 4, 7, 0},
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_i64, split_2d_uneven_3_split3_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 0},
                                                {0, 4, 7, 0},
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_f32, split_3d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_i64, split_3d_uneven_2_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1}
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}

TEST(split_gpu_f32, split_3d_uneven_3_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1},
                                                {0, 7, 8, 2}
                                               };

    split_test<float>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}
#endif
TEST(split_gpu_i64, split_3d_uneven_3_splits_cached) {
    auto batch_num = 2;
    auto feature_num = 8;
    auto x_size = 10;
    auto y_size = 3;
    std::vector<cldnn::tensor> split_offsets = {
                                                {0, 0, 0, 0},
                                                {0, 1, 4, 1},
                                                {0, 7, 8, 2}
                                               };

    split_test<int64_t>(batch_num, feature_num, x_size, y_size, split_offsets, true);
}
