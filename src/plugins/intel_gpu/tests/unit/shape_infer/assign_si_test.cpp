// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/assign.hpp>

#include "assign_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct assign_test_params {
    layout input_layout;
    std::string variable_id;
    layout expected_layout;
};

class assign_test : public testing::TestWithParam<assign_test_params> { };

TEST_P(assign_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    const auto variable_layout = p.input_layout;

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto inputs = std::vector<input_info>{ input_info("input") };
    auto assign_prim = std::make_shared<assign>("assign", inputs, p.variable_id, variable_layout);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& assign_node = prog.get_or_create(assign_prim);
    program_wrapper::add_connection(prog, input_layout_node, assign_node);

     auto res = assign_inst::calc_output_layouts<ov::PartialShape>(assign_node, *assign_node.get_kernel_impl_params());
     ASSERT_EQ(res.size(), 1);
     ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, assign_test,
    testing::ValuesIn(std::vector<assign_test_params>{
        {
            layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx},
            "v0",
            layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            "v0",
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}
        }
    }));


}  // shape_infer_tests
