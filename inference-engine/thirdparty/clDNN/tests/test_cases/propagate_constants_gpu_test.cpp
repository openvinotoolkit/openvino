// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/concatenation.hpp>
#include <api/reorder.hpp>
#include <api/data.hpp>
#include <api/reshape.hpp>

using namespace cldnn;
using namespace tests;

//We expect additional reorder to be added in between "weights1" and "reshape1".
//This situation should be handled properly by propagate constants optimization phase
TEST(propagate_constants, copy_dependecies_from_nodes) {
    const auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { FLOAT16(1.1f), FLOAT16(1.2f), FLOAT16(1.3f), FLOAT16(1.4f) });
    set_values(weights1, { FLOAT16(2.1f), FLOAT16(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", "weights1", tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", "input", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", "reshape1", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { "reorder1", "weights2" }, concatenation::along_x));
    topology.add(convolution("conv2", { "reorder2" }, { "concat" }));
    network network(engine, topology, build_opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs)
    {
        auto output = it.second.get_memory().pointer<float>();
        EXPECT_NEAR(7.8f, output[0], epsilon);
    }
}