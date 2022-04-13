// Copyright (C) 2018-2022 Intel Corporation
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
TEST(propagate_constants, copy_dependecies_from_nodes) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { FLOAT16(1.1f), FLOAT16(1.2f), FLOAT16(1.3f), FLOAT16(1.4f) });
    set_values(weights1, { FLOAT16(2.1f), FLOAT16(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", "weights1", tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", "input", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", "reshape1", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { "reorder1", "weights2" }, 3));
    topology.add(convolution("conv2", { "reorder2" }, { "concat" }));
    network network(engine, topology, build_opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs) {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        EXPECT_NEAR(7.8f, output[0], epsilon);
    }
}
