// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/permute.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace bounded_shape_mem_alloc_tests {

TEST(dyn_shape_bounded_shape_mem, reorder) {
    auto& engine = get_test_engine();
    auto input_lay = layout{ov::PartialShape{ov::Dimension(1, 10), ov::Dimension(1, 20)}, data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_lay),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    const auto reorder_mem = network.get_primitive("reorder")->output_memory_ptr();
    ASSERT_NE(reorder_mem, nullptr);
    ASSERT_EQ(reorder_mem->get_layout().count(), 10*20);
}

TEST(dyn_shape_bounded_shape_mem, permute) {
    auto& engine = get_test_engine();
    auto input_lay = layout{ov::PartialShape{ov::Dimension(1, 10), ov::Dimension(1,15), ov::Dimension(1,5), ov::Dimension(1, 20)}, data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_lay),
                      permute("permute", input_info("input"), {0, 2, 3, 1}));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    const auto permute_mem = network.get_primitive("permute")->output_memory_ptr();
    ASSERT_NE(permute_mem, nullptr);
    ASSERT_EQ(permute_mem->get_layout().count(), 10*15*5*20);
}
}
