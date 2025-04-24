// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "concatenation_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct concatenation_test_params {
    std::vector<layout> input_layouts;
    int64_t axis;
    layout expected_layout;
};

class concatenation_test : public testing::TestWithParam<concatenation_test_params> { };

TEST_P(concatenation_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<input_layout>> input_prims;
    std::vector<input_info> input_prims_ids;
    for (size_t i = 0; i < p.input_layouts.size(); i++) {
        auto prim_id = "data" + std::to_string(i);
        auto data_layout_prim = std::make_shared<input_layout>(prim_id, p.input_layouts[i]);
        input_prims.push_back(data_layout_prim);
        input_prims_ids.push_back(input_info(prim_id));
    }

    auto concatenation_prim = std::make_shared<concatenation>("output", input_prims_ids, p.axis);
    auto& concatenation_node = prog.get_or_create(concatenation_prim);
    for (size_t i = 0; i < p.input_layouts.size(); i++) {
        auto& input_node = prog.get_or_create(input_prims[i]);
        program_wrapper::add_connection(prog, input_node, concatenation_node);
    }

    auto params = concatenation_node.get_kernel_impl_params();
    auto res = concatenation_inst::calc_output_layouts<ov::PartialShape>(concatenation_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, concatenation_test,
    testing::ValuesIn(std::vector<concatenation_test_params>{
        {
            {
                layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx},
                layout{ov::PartialShape{1, 3, 3, 4}, data_types::f32, format::bfyx},
            },
            1,
            layout{ov::PartialShape{1, 5, 3, 4}, data_types::f32, format::bfyx}
        },
        {
            {
                layout{ov::PartialShape{4, 2}, data_types::f32, format::bfyx},
                layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
                layout{ov::PartialShape{2, 2}, data_types::f32, format::bfyx},
            },
            0,
            layout{ov::PartialShape{9, 2}, data_types::f32, format::bfyx}
        },
        {
            {
                layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx},
                layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
                layout{ov::PartialShape{2, 2}, data_types::f32, format::bfyx},
            },
            0,
            layout{ov::PartialShape{ov::Dimension(5, -1), ov::Dimension(2, 2)}, data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
