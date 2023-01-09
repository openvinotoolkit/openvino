// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "softmax_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct softmax_test_params {
    layout input_layout;
    int64_t axis;
};

class softmax_si_test : public testing::TestWithParam<softmax_test_params> { };

TEST_P(softmax_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto softmax_prim = std::make_shared<softmax>("output", input_info("input0"), p.axis);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& softmax_node = prog.get_or_create(softmax_prim);
    program_wrapper::add_connection(prog, input0_layout_node, softmax_node);
    auto res = softmax_inst::calc_output_layouts<ov::PartialShape>(softmax_node, *softmax_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, softmax_si_test,
    testing::ValuesIn(std::vector<softmax_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, 1},
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, 2},
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, 4},
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, 2},
        { layout{ov::PartialShape::dynamic(5), data_types::f16, format::bfzyx}, -1}
    }));

}  // shape_infer_tests
