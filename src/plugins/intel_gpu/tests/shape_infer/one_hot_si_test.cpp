// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/one_hot.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "one_hot_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct one_hot_test_params {
    layout in_layout;
    int64_t axis;
    int64_t depth;
    layout expected_layout;
};

class one_hot_test : public testing::TestWithParam<one_hot_test_params> { };

TEST_P(one_hot_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto one_hot_prim = std::make_shared<one_hot>("output", input_info("input"), tensor(), p.axis, p.depth);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& one_hot_node = prog.get_or_create(one_hot_prim);
    program_wrapper::add_connection(prog, input_layout_node, one_hot_node);
    auto res = one_hot_inst::calc_output_layouts<ov::PartialShape>(one_hot_node, *one_hot_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, one_hot_test,
    testing::ValuesIn(std::vector<one_hot_test_params>{
        {layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, 3, 5, layout{ov::PartialShape{1, 2, 3, 5}, data_types::f32, format::bfyx}},
        {layout{ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx}, -1, 5, layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}},
    }));

}  // shape_infer_tests
