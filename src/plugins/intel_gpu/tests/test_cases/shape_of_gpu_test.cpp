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

TEST(shape_of_gpu, basic) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = engine.allocate_memory({data_types::f32, format::yxfb, tensor{b, f, y, x}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of", "input", data_types::i32));

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
