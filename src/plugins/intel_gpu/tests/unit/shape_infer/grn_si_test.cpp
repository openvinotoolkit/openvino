// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/grn.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "grn_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct grn_test_params {
    layout input_layout;
    data_types out_dt;
};

class grn_test : public testing::TestWithParam<grn_test_params> { };

TEST_P(grn_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto grn_prim = std::make_shared<grn>("output", input_info("input0"), 0.1f, p.out_dt);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& grn_node = prog.get_or_create(grn_prim);
    program_wrapper::add_connection(prog, input0_layout_node, grn_node);
    auto res = grn_inst::calc_output_layouts<ov::PartialShape>(grn_node, *grn_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    auto expected_layout = p.input_layout;
    expected_layout.data_type = p.out_dt;
    ASSERT_EQ(res[0], expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, grn_test,
    testing::ValuesIn(std::vector<grn_test_params>{
        { layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, data_types::f32},
        { layout{ov::PartialShape{1, 2, 3, 4}, data_types::f16, format::bfyx}, data_types::f32},
        { layout{ov::PartialShape{1, 2, 3, 4, 5}, data_types::f32, format::bfzyx}, data_types::f32},
        { layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx}, data_types::f16},
        { layout{ov::PartialShape::dynamic(5), data_types::f32, format::bfzyx}, data_types::f32}
    }));

}  // shape_infer_tests
