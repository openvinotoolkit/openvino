// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reverse.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "reverse_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct reverse_test_params {
    layout input_layout;
    reverse_mode mode;
};

class reverse_test : public testing::TestWithParam<reverse_test_params> { };

TEST_P(reverse_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input1_layout = layout{{1}, data_types::i32, format::bfyx};
    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", input1_layout);
    auto reverse_prim = std::make_shared<reverse>("output", input_info("input0"), input_info("input1"), p.mode);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& reverse_node = prog.get_or_create(reverse_prim);
    program_wrapper::add_connection(prog, input0_layout_node, reverse_node);
    program_wrapper::add_connection(prog, input1_layout_node, reverse_node);
    auto res = reverse_inst::calc_output_layouts<ov::PartialShape>(reverse_node, *reverse_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, reverse_test,
    testing::ValuesIn(std::vector<reverse_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, reverse_mode::index },
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, reverse_mode::mask },
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, reverse_mode::index },
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, reverse_mode::mask },
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, reverse_mode::index }
    }));

}  // shape_infer_tests
