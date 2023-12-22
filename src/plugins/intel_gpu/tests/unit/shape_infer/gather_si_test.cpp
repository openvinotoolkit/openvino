// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gather_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct gather_test_params {
    layout in0_layout;
    layout in1_layout;
    int64_t axis;
    int64_t batch_dim;
    layout expected_layout;
};

class gather_test : public testing::TestWithParam<gather_test_params> { };

TEST_P(gather_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.in0_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", p.in1_layout);
    auto gather_prim = std::make_shared<gather>("output", input_info("input0"), input_info("input1"), p.axis, 0, ov::Shape{}, p.batch_dim);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& gather_node = prog.get_or_create(gather_prim);
    program_wrapper::add_connection(prog, input0_layout_node, gather_node);
    program_wrapper::add_connection(prog, input1_layout_node, gather_node);
    auto res = gather_inst::calc_output_layouts<ov::PartialShape>(gather_node, *gather_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gather_test,
    testing::ValuesIn(std::vector<gather_test_params>{
        {
            layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, layout{ov::PartialShape{4, 5}, data_types::f32, format::bfyx},
            1, 0,
            layout{ov::PartialShape{1, 4, 5, 3}, data_types::f32, format::bfyx}
        },
    }));

INSTANTIATE_TEST_SUITE_P(optimized, gather_test,
    testing::ValuesIn(std::vector<gather_test_params>{
        {
            layout{ov::PartialShape{3, 4, 2, 2}, data_types::f32, format::bfyx}, layout{ov::PartialShape{1}, data_types::f32, format::bfyx},
            0, 0,
            layout{ov::PartialShape{1, 4, 2, 2}, data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
