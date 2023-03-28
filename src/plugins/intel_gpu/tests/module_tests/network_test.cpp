// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/concatenation.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(network_test, model_with_scalar_input_is_not_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {};
    layout in_layout{input_shape, data_types::f32, format::bfyx};

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(broadcast("output", input_info("input"), {1, 2}, ov::AxisSet{}));

    network net(engine, topology);

    ASSERT_FALSE(net.is_dynamic());
}

TEST(network_test, model_with_empty_input_is_not_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {1, 0};
    layout in_layout{input_shape, data_types::f32, format::bfyx};
    auto const_mem = engine.allocate_memory({{1, 2}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input0", in_layout));
    topology.add(data("input1", const_mem));
    topology.add(concatenation("output", { input_info("input0"), input_info("input1") }, 1));

    network net(engine, topology, {ov::intel_gpu::allow_new_shape_infer(true)});

    ASSERT_FALSE(net.is_dynamic());
}

TEST(network_test, model_with_dynamic_input_is_dynamic) {
    auto& engine = get_test_engine();
    ov::PartialShape input_shape = {1, -1};
    layout in_layout{input_shape, data_types::f32, format::bfyx};
    auto const_mem = engine.allocate_memory({{1, 2}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input0", in_layout));
    topology.add(data("input1", const_mem));
    topology.add(concatenation("output", { input_info("input0"), input_info("input1") }, 1));

    network net(engine, topology, {ov::intel_gpu::allow_new_shape_infer(true)});

    ASSERT_TRUE(net.is_dynamic());
}
