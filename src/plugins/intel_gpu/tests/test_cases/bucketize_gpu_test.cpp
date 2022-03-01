// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/bucketize.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(bucketize_bfyx, wrb_true) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 10, 1}});
    auto buckets = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 4}});

    set_values(input, {8.f, 1.f, 2.f, 1.1f, 8.f, 10.f, 1.f, 10.2f, 0.f, 20.f});
    set_values(buckets, {1, 4, 10, 20});

    const bool with_right_bound = true;

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(input_layout("Buckets", buckets->get_layout()));
    topology.add(bucketize("bucketize", "Input", "Buckets", data_types::f32, with_right_bound));

    network network(engine, topology);

    network.set_input_data("Input", input);
    network.set_input_data("Buckets", buckets);

    auto outputs = network.execute();

    std::vector<int32_t> expected_out{2, 0, 1, 1, 2, 2, 0, 3, 0, 3};

    auto output = outputs.at("bucketize").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(expected_out.size(), output_ptr.size());
    for (std::size_t i = 0; i < expected_out.size(); i++) {
        EXPECT_EQ(expected_out[i], output_ptr[i]);
    }
}

TEST(bucketize_bfyx, wrb_false) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 10}});
    auto buckets = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 4}});

    set_values(input, {8, 1, 2, 1, 8, 5, 1, 5, 0, 20});
    set_values(buckets, {1, 4, 10, 20});

    const bool with_right_bound = false;

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(input_layout("Buckets", buckets->get_layout()));
    topology.add(bucketize("bucketize", "Input", "Buckets", data_types::i32, with_right_bound));

    network network(engine, topology);

    network.set_input_data("Input", input);
    network.set_input_data("Buckets", buckets);

    auto outputs = network.execute();

    std::vector<int32_t> expected_out{2, 1, 1, 1, 2, 2, 1, 2, 0, 4};

    auto output = outputs.at("bucketize").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(expected_out.size(), output_ptr.size());
    for (std::size_t i = 0; i < expected_out.size(); i++) {
        EXPECT_EQ(expected_out[i], output_ptr[i]);
    }
}
