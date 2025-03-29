// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/roll.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "roll_inst.hpp"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct roll_test_params {
    layout input_layout;
};

class roll_test : public testing::TestWithParam<roll_test_params> { };

TEST_P(roll_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto roll_prim = std::make_shared<roll>("output", input_info("input0"), tensor(1));

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& roll_node = prog.get_or_create(roll_prim);
    program_wrapper::add_connection(prog, input0_layout_node, roll_node);
    auto res = roll_inst::calc_output_layouts<ov::PartialShape>(roll_node, *roll_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, roll_test,
    testing::ValuesIn(std::vector<roll_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx} },
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx} },
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx} },
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx} },
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx} }
    }));

}  // shape_infer_tests
