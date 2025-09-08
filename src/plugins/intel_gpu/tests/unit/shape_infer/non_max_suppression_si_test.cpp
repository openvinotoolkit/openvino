// Copyright (C) 2022 Intel Corporation
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
    std::vector<layout> in_layouts;
    float max_output_boxes_per_class;
    int32_t selected_indices_num;
    bool center_point_box;
    bool sort_result_descending;
    size_t num_outputs;
    std::vector<layout> expected_layouts;
};

class non_max_suppression_test : public testing::TestWithParam<non_max_suppression_test_params> { };

TEST_P(non_max_suppression_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    for (size_t i = 0; i < 2; i++) {
        auto prim_id = "input" + std::to_string(i);
        auto input_layout_prim = std::make_shared<input_layout>(prim_id, p.in_layouts[i]);
        input_prims.push_back(input_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    for (size_t i = 2; i < p.in_layouts.size(); i++) {
        auto prim_id = "const" + std::to_string(i);
        auto data_mem = engine.allocate_memory(p.in_layouts[i]);
        set_values(data_mem, {p.max_output_boxes_per_class});
        auto data_prim = std::make_shared<data>(prim_id, data_mem);
        input_prims.push_back(data_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto non_max_suppression_prim = std::make_shared<non_max_suppression>("output",
                                                                          input_prim_ids[0],
                                                                          input_prim_ids[1],
                                                                          p.selected_indices_num,
                                                                          p.center_point_box,
                                                                          p.sort_result_descending,
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          primitive_id(),
                                                                          p.num_outputs);
    non_max_suppression_prim->output_paddings = {padding(), padding(), padding()};
    non_max_suppression_prim->output_data_types = {optional_data_type{}, optional_data_type{p.in_layouts[1].data_type}, optional_data_type{}};
    if (p.in_layouts.size() > 2) {
        non_max_suppression_prim->num_select_per_class = input_prim_ids[2].pid;
    }

    cldnn::program prog(engine);

    auto& non_max_suppression_node = prog.get_or_create(non_max_suppression_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, non_max_suppression_node);
    }

    auto params = non_max_suppression_node.get_kernel_impl_params();
    auto res = non_max_suppression_inst::calc_output_layouts<ov::PartialShape>(non_max_suppression_node, *params);

    ASSERT_EQ(res.size(), p.num_outputs);
    for (size_t i = 0; i < p.expected_layouts.size(); i++)
        ASSERT_EQ(res[i], p.expected_layouts[i]);
}

INSTANTIATE_TEST_SUITE_P(smoke, non_max_suppression_test,
    testing::ValuesIn(std::vector<non_max_suppression_test_params>{
        {
            {layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{2, 2, 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::f32, format::bfyx}},
            1.f, 4, false, true, 3,
            {layout{ov::PartialShape{4, 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{4, 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{2, 2, 3}, data_types::f32, format::bfyx}},
            1.f, 4, false, true, 3,
            {layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
             layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::f32, format::bfyx}},
            1.f, 4, false, true, 3,
            {layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
             layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::f32, format::bfyx}},
            1.f, 4, false, true, 3,
            {layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::i32, format::bfyx},
             layout{ov::PartialShape{ov::Dimension::dynamic(), 3}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1}, data_types::i32, format::bfyx}}
        },
    }));

}  // shape_infer_tests
