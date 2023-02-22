// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/bucketize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "bucketize_inst.hpp"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct bucketize_test_params {
    layout input_layout;
    data_types out_dt;
    bool with_right_bound;
};

class bucketize_test : public testing::TestWithParam<bucketize_test_params> { };

TEST_P(bucketize_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", layout({3}, p.input_layout.data_type, format::bfyx));
    auto inputs = std::vector<input_info>{ input_info("input0"), input_info("input1")};
    auto bucketize_prim = std::make_shared<bucketize>("output", inputs, p.out_dt, p.with_right_bound);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& bucketize_node = prog.get_or_create(bucketize_prim);
    program_wrapper::add_connection(prog, input0_layout_node, bucketize_node);
    program_wrapper::add_connection(prog, input1_layout_node, bucketize_node);
    auto res = bucketize_inst::calc_output_layouts<ov::PartialShape>(bucketize_node, *bucketize_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    expected_layout.data_type = p.out_dt;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, bucketize_test,
    testing::ValuesIn(std::vector<bucketize_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, data_types::i64, true },
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, data_types::i64, false },
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, data_types::i32, true },
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, data_types::i64, true},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, data_types::i32, false}
    }));

}  // shape_infer_tests
