// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_gather_tests {
TEST(skip_gather_at_runtime, not_skip_if_cpuimpl) {
    auto& engine = get_test_engine();

    ov::Shape in1_shape = { 3, 3 };
    ov::Shape in2_shape = { 2, 2 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::i32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx}); // data
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::i32, format::bfyx}); // Indexes

    int64_t axis = 1;
    set_values(input1, {0, 1, 2, 10, 11, 12, 20, 21, 22 });
    set_values(input2, {1, 0, 2, 1});

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(gather("gather", input_info("input1"), input_info("input2"), axis, 0, ov::Shape{}));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gather", {format::bfyx, "", impl_types::cpu}} }));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto gather_inst = network.get_primitive("gather");
    auto impl = gather_inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(gather_inst->can_be_optimized(), false);

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {1, 0, 2, 1,  11, 10, 12, 11,  21, 20, 22, 21};

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]) << i;
    }
}
}  // skip_gather_tests
