// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace is_valid_fusion_tests {
TEST(eltwise_activation_fusing_test, basic_dynamic_rank4) {
    // is_valid_fusion() should work properly when conv->add->prelu case
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx};
    auto weights = engine.allocate_memory(weight_layout);
    set_values<ov::float16>(weights, {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            //
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
    });

    layout in_layout = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    set_values(input_mem, {11.0f,  11.0f, 11.0f, 11.0f,
                           11.0f,  11.0f, 11.0f, 11.0f,
                           11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref = { 77.0f,  143.0f, 143.0f, 77.0f,
                               143.0f, 275.0f, 275.0f, 143.0f,
                               143.0f, 275.0f, 275.0f, 143.0f,
                               77.0f,  143.0f, 143.0f, 77.0f };

    auto const1 = engine.allocate_memory(layout{ov::PartialShape({1, 1, 1, 1}), data_types::f32, format::bfyx});
    set_values(const1, {11.0f});
    auto const2 = engine.allocate_memory(layout{ov::PartialShape({1, 1, 1, 1}), data_types::f32, format::bfyx});
    set_values(const2, {0.1f});
    std::vector<float> values_to_subtract = {};

    auto in_layout_0 = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", in_layout_0),
                      data("weights", weights),
                      data("const1", const1),
                      data("const2", const2),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("reorder"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      eltwise("eltwise", input_info("conv"), input_info("const1"), eltwise_mode::sum),
                      activation("prelu", input_info("eltwise"), "const2", activation_func::relu_negative_slope),
                      reorder("output", input_info("prelu"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_ptr(output_mem, get_test_stream());

    for (size_t i = 0; i < output_mem->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_ptr[i], ref[i]);
    }
}
}  // is_valid_fusion_tests
