// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_max_suppression.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "non_max_suppression_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct non_max_suppression_test_params {
    layout in0_layout;
    layout in1_layout;
    layout data_layout;
    float max_output_boxes_per_class;
    int32_t selected_indices_num;
    bool center_point_box;
    bool sort_result_descending;
    std::vector<input_info> inputs;
    size_t num_outputs;
    std::vector<layout> expected_layouts;
};

class non_max_suppression_test : public testing::TestWithParam<non_max_suppression_test_params> { };

TEST_P(non_max_suppression_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.in0_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", p.in1_layout);
    auto data_mem = engine.allocate_memory(p.data_layout);
    set_values(data_mem, {p.max_output_boxes_per_class});
    auto data_prim = std::make_shared<data>("const", data_mem);
    auto non_max_suppression_prim = std::make_shared<non_max_suppression>("output",
                                                                          p.inputs[0],
                                                                          p.inputs[1],
                                                                          p.selected_indices_num,
                                                                          p.center_point_box,
                                                                          p.sort_result_descending,
                                                                          "const",
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          p.num_outputs);
    non_max_suppression_prim->output_paddings = {padding(), padding(), padding()};
    non_max_suppression_prim->output_data_types = {optional_data_type{}, optional_data_type{p.in1_layout.data_type}, optional_data_type{}};

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& data_node = prog.get_or_create(data_prim);
    auto& non_max_suppression_node = prog.get_or_create(non_max_suppression_prim);
    program_wrapper::add_connection(prog, input0_layout_node, non_max_suppression_node);
    program_wrapper::add_connection(prog, input1_layout_node, non_max_suppression_node);
    program_wrapper::add_connection(prog, data_node, non_max_suppression_node);

    auto params = non_max_suppression_node.get_kernel_impl_params();
    auto res = non_max_suppression_inst::calc_output_layouts<ov::PartialShape>(non_max_suppression_node, *params);

    ASSERT_EQ(res.size(), p.num_outputs);
    for (size_t i = 0; i < p.expected_layouts.size(); i++)
        ASSERT_EQ(res[i], p.expected_layouts[i]);
}

INSTANTIATE_TEST_SUITE_P(smoke, non_max_suppression_test,
    testing::ValuesIn(std::vector<non_max_suppression_test_params>{
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 2, 3}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::f32, format::bfyx},
            1.f, 4, false, true, {input_info("input0", 0), input_info("input1", 0)}, 3,
            {layout{ov::PartialShape{4, 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{4, 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::f32, format::bfyx},
            1.f, 4, false, true, {input_info("input0", 0), input_info("input1", 0)}, 3,
            {layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
    }));

}  // shape_infer_tests
