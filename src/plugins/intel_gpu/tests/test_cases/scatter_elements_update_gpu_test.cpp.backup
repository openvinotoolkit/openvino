// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;


TEST(scatter_elements_update_gpu_fp16, d2411_axisF) {
    //  Dictionary : 2x4x1x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x1x1
    //  Axis : 1
    //  Output : 2x4x1x1
    //  Input values in fp16
    //
    //  Input:
    //  3.f, 6.f, 5.f, 4.f,
    //  1.f, 7.f, 2.f, 9.f
    //
    //  Indexes:
    //  0.f, 1.f
    //  2.f, 3.f
    //
    //  Updates:
    //  10.f, 11.f,
    //  12.f, 13.f
    //
    //  Output:
    //  10.f, 11.f, 5.f, 4.f,
    //  1.f, 7.f, 12.f, 13.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_elements_update::scatter_elements_update_axis::along_f;

    set_values(input1, {
        FLOAT16(3.0f), FLOAT16(6.0f), FLOAT16(5.0f), FLOAT16(4.0f),
        FLOAT16(1.0f), FLOAT16(7.0f), FLOAT16(2.0f), FLOAT16(9.0f)
    });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f)
    });

    set_values(input3, {
        FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f)
    });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_elements_update("scatter_elements_update", "InputData", "InputIndices", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_elements_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        10.f, 11.f, 5.f, 4.f,
        1.f, 7.f, 12.f, 13.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}
