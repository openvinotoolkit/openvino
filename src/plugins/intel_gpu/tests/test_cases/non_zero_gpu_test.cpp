// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include "ngraph/runtime/reference/non_zero.hpp"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;
template<typename T>
void test_count_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ngraph::runtime::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());

    set_values(input_mem, in_data);

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(count_nonzero("count_nonzero", "InputData")
    );

    network network(engine, topology);
    network.set_input_data("InputData", input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("count_nonzero").get_memory();

    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    EXPECT_EQ(count_non_zero, output_ptr[1]);
}

TEST(test_count_non_zero, 4d_fp32_1_2_1_5) {
    std::vector<float> in_data = {
        0.3f, 0.2f   , 0.3f, 0.0f, 0.0f,
        0.4f, 0.0001f, 0.1f, 0.9f, 0.10f
    };
    test_count_non_zero<float>(layout{ov::PartialShape{1, 2, 1, 5}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_gather_non_zero, 5d_fp16_1_3_2_1_2) {
    std::vector<FLOAT16> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f
    };
    test_count_non_zero<FLOAT16>(layout{ov::PartialShape{1, 3, 2, 1, 2}, data_types::f16, format::bfzyx}, in_data);
}

template<typename T>
void test_gather_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ngraph::runtime::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());
    auto in_rank = in_layout.get_shape().size();
    std::vector<int32_t> expected_results(count_non_zero * in_rank);
    ngraph::runtime::reference::non_zero<T, int32_t>(in_data.data(), expected_results.data(), in_layout.get_shape());

    auto output_shape_layout = layout{ov::PartialShape{4}, data_types::i32, format::bfyx};
    auto output_shape_mem = engine.allocate_memory(output_shape_layout);
    set_values(input_mem, in_data);

    std::vector<int32_t> output_shape_data = {(int32_t)in_rank, (int32_t)count_non_zero, 1, 1};

    set_values(output_shape_mem, output_shape_data);

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(input_layout("OutputShape", output_shape_layout));
    topology.add(
        gather_nonzero("gather_nonzero", "InputData", "OutputShape")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_mem);
    network.set_input_data("OutputShape", output_shape_mem);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    cldnn::mem_lock<int32_t> shape_ptr(output_shape_mem, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(test_gather_non_zero, 4d_fp32_1_3_3_1) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f,
        0.0f, 0.4f, 0.1f, 0.9f, 0.10f
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{1, 3, 3, 1}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_gather_non_zero, 4d_fp32_2_4_3_2) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f32, format::bfyx}, in_data);
}
TEST(test_gather_non_zero, 4d_fp16_2_4_3_2) {
    std::vector<FLOAT16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_gather_non_zero<FLOAT16>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_gather_non_zero, 5d_fp32_1_3_3_2_2) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{1, 3, 4, 2, 2}, data_types::f32, format::bfzyx}, in_data);
}

TEST(test_gather_non_zero, 6d_fp16_2_3_1_3_2_4) {
   std::vector<float> in_data = {
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        1.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        19.0f, 0.0f, 0.1f, 0.9f,  0.10f, -0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        11.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        -13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.1f,  1.2f, 0.3f, 99.0f,  12.1f, 11.1f,
       100.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  -0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{2, 3, 1, 3, 2, 4}, data_types::f32, format::bfwzyx}, in_data);
}

template<typename T>
void test_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ngraph::runtime::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());
    auto in_rank = in_layout.get_shape().size();
    std::vector<int32_t> expected_results(count_non_zero * in_rank);
    ngraph::runtime::reference::non_zero<T, int32_t>(in_data.data(), expected_results.data(), in_layout.get_shape());

    set_values(input_mem, in_data);

    std::vector<int32_t> output_shape_data = {(int32_t)in_rank, (int32_t)count_non_zero, 1, 1};

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(count_nonzero("count_nonzero", "InputData"));
    topology.add(gather_nonzero("gather_nonzero", "InputData", "count_nonzero"));

    network network(engine, topology);

    network.set_input_data("InputData", input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(test_non_zero, 1d_fp16_48) {
    std::vector<FLOAT16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<FLOAT16>(layout{ov::PartialShape{48}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_non_zero, 2d_fp32_2_34) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<float>(layout{ov::PartialShape{2, 24}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_non_zero, 3d_fp16_4_3_4) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<float>(layout{ov::PartialShape{4, 3, 4}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_non_zero, 4d_fp16_2_4_3_2) {
    std::vector<FLOAT16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<FLOAT16>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_non_zero, 5d_fp32_1_3_3_2_2) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
    };
    test_non_zero<float>(layout{ov::PartialShape{1, 3, 4, 2, 2}, data_types::f32, format::bfzyx}, in_data);
}

TEST(test_non_zero, 6d_fp16_2_3_1_3_2_4) {
    std::vector<float> in_data = {
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        1.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        19.0f, 0.0f, 0.1f, 0.9f,  0.10f, -0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        11.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        -13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.1f,  1.2f, 0.3f, 99.0f,  12.1f, 11.1f,
       100.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  -0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
    };
    test_non_zero<float>(layout{ov::PartialShape{2, 3, 1, 3, 2, 4}, data_types::f32, format::bfwzyx}, in_data);
}

TEST(test_non_zero, 6d_fp16_2_2_2_1_5_1) {
    std::vector<int32_t> in_data = {
        10, 12, 23, 1232, 11, 9, 10, 23, 0, 1,
        0,  12, 23, 0, 11, 9, 10, 23, 0, 1,
        10, 0,  2,  32, 11, 9, 10, 23, 0, 1,
        10, 12, 3, 12, 11, 9, 0, 23, 0, 1
    };
    test_non_zero<int32_t>(layout{ov::PartialShape{2, 2, 2, 1, 5, 1}, data_types::i32, format::bfwzyx}, in_data);
}
