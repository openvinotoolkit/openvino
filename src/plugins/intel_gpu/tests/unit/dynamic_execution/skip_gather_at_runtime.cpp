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

    auto in1_layout = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape{-1}, data_types::i32, format::bfyx};

    topology topology(input_layout("input1", in1_layout),
                      input_layout("input2", in2_layout),
                      gather("gather",
                             input_info("input1"),
                             input_info("input2"),
                             0,                                       // axis
                             in1_layout.get_partial_shape().size(),   // input rank
                             ov::Shape{},                             // output shape
                             0,                                       // batch_dim
                             true),                                   // support_neg_ind
                      reorder("reorder", input_info("gather"), format::bfyx, data_types::f32)); /*output padding*/

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gather", {format::bfyx, "", impl_types::cpu}} }));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    ASSERT_EQ(gather_inst->can_be_optimized(), false);
}
}  // skip_gather_tests
