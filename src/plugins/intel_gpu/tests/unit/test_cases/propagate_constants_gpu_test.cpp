// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>

using namespace cldnn;
using namespace ::tests;

//We expect additional reorder to be added in between "weights1" and "reshape1".
//This situation should be handled properly by propagate constants optimization phase
template <typename T>
void test_copy_dependecies_from_nodes(bool is_caching_test) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { T(1.1f), T(1.2f), T(1.3f), T(1.4f) });
    set_values(weights1, { T(2.1f), T(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", input_info("weights1"), tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", input_info("input"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", input_info("reshape1"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { input_info("reorder1"), input_info("weights2") }, 3));
    topology.add(convolution("conv2", input_info("reorder2"), "concat", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);

    auto outputs = network->execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs) {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        ASSERT_NEAR(7.8f, output[0], epsilon);
    }
}

TEST(propagate_constants, copy_dependecies_from_nodes) {
    test_copy_dependecies_from_nodes<ov::float16>(false);
}

TEST(propagate_constants, copy_dependecies_from_nodes_cached) {
    test_copy_dependecies_from_nodes<ov::float16>(true);
}
