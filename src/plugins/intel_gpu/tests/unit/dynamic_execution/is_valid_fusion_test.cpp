// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "softmax_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace is_valid_fusion_tests {
TEST(eltwise_activation_fusing_test, basic) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f32, format::bfyx};

    auto weights = engine.allocate_memory(weight_layout);
    set_values<FLOAT16>(weights, {
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

    layout input_layout_1 = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem_1 = engine.allocate_memory(input_layout_1);
    set_values(input_mem_1, {11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref_output_1 = { 66, 132, 132, 66, 132, 264, 264, 132, 132, 264, 264, 132, 66, 132, 132, 66};
    
    auto const1 = engine.allocate_memory(layout{ov::PartialShape({1, 1, 1, 1}), data_types::f32, format::bfyx});
    set_values(const1, {11.0f});
    auto const2 = engine.allocate_memory(layout{ov::PartialShape({1, 1, 1, 1}), data_types::f32, format::bfyx});
    set_values(const2, {0.05f});
    std::vector<float> values_to_subtract = {};

    auto in_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", in_layout),
                      data("weights", weights),
                      data("const1", const1),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("input"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT,
                                  padding{{0, 0, 0, 0}, 0}),
                      eltwise("eltwise", input_info("conv"), input_info("const1"), eltwise_mode::sum),
                      activation("relu", input_info("eltwise"), activation_func::relu),
                      reorder("output", input_info("relu"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    //config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem_1);
    auto outputs_1 = network.execute();
    auto output_mem_1 = outputs_1.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_1_ptr(output_mem_1, get_test_stream());
    for (size_t i = 0; i < output_mem_1->get_layout().get_buffer_size().count(); ++i) {
        ASSERT_EQ(output_mem_1_ptr[i], ref_output_1[i]);
    }
}
}  // memory_realloc_tests
