// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shuffle_channels.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

template <typename T>
void test_multiple_outputs(bool is_caching_test) {
    // Tests split with crop implementation
    //                                                   _ strided_slice(bfyx)
    //                                                  |
    //  INPUT(bfyx,3x2x1x1)--shuffle_channels(bfyx)-----|
    //                                                  |_
    //                                                     reshape(bfyx);

    auto& engine = get_test_engine();
    auto batch_num = 6;
    auto feature_num = 1;
    auto x_size = 1;
    auto y_size = 1;
    int32_t axis = 0;
    int32_t group = 2;

    tensor initial_shape = tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num));
    tensor after_strided_slice = tensor(spatial(y_size, feature_num), feature(batch_num), batch(1));
    tensor after_reshape = tensor(feature(batch_num * feature_num * y_size * x_size));

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, initial_shape });
    auto begin = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto end = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto strides = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shuffle_channels("shuffle_channels", input_info("input"), group, axis));
    topology.add(reshape("reshape", input_info("shuffle_channels"), after_reshape));
    topology.add(data("input2", begin));
    topology.add(data("input3", end));
    topology.add(data("input4", strides));
    topology.add(strided_slice("strided_slice", input_info("shuffle_channels"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, { 1 }, {}, {}, {6, 1, 1, 1}));

    std::vector<T> input_vec = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<T> out_vec = { 0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f };
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "shuffle_channels", "reshape", "strided_slice" }));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("reshape").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    ASSERT_TRUE(output->get_layout().get_tensor() == after_reshape);

    for (size_t i = 0; i < out_vec.size(); i++)
        ASSERT_EQ(output_ptr[i], out_vec[i]);

    // checking the output node has the same name after output node deleting due to StridedSlice optimization
    ASSERT_TRUE(outputs.find("strided_slice") != outputs.end());
    auto output2 = outputs.at("strided_slice").get_memory();
    cldnn::mem_lock<T> output_ptr2(output, get_test_stream());

    ASSERT_TRUE(output2->get_layout().get_tensor() == after_strided_slice);

    for (size_t i = 0; i < out_vec.size(); i++)
        ASSERT_EQ(output_ptr2[i], out_vec[i]);
}

TEST(removing_output_node, DISABLED_multiple_outputs) { // Issue 129991
    test_multiple_outputs<float>(false);
}

template <typename T>
void test_output_node_optimization(bool is_caching_test) {
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

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,format::yxfb,{ 1, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    VVF<T> output_vec = {
            { 20.0f, 27.0f, 38.0f },
            { 17.0f, 19.0f, 19.0f } };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("input"), "weights", "", 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(activation("relu", input_info("conv"), activation_func::relu));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);

    // checking the output node has the same name after output node deleting due to ReLU optimization
    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::yxfb);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(removing_output_node, output_node_optimization) {
    test_output_node_optimization<float>(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(removing_output_node, DISABLED_multiple_outputs_cached) {  // Issue 129991
    test_multiple_outputs<float>(true);
}
#endif
TEST(removing_output_node, output_node_optimization_cached) {
    test_output_node_optimization<float>(true);
}
