// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <vector>
#include <iostream>

using namespace cldnn;
using namespace ::tests;

TEST(shape_of_gpu, bfyx) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", "input", 4, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}


TEST(shape_of_gpu, bfyx_i64) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", "input", 4, data_types::i64));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    std::vector<int64_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, yxfb) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::yxfb, tensor{1, 2, 3, 3}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", "input", 4, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(shape_of_gpu, bfzyx) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, tensor{1, 2, 3, 3, 4}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", "input", 5, data_types::i32));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("shape_of").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results = {1, 2, 4, 3, 3};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}
