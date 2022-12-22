// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "reorder_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct reorder_si_test_params {
    layout data_layout;
    layout expected_layout;
};

class reorder_si_test : public testing::TestWithParam<reorder_si_test_params> { };

TEST_P(reorder_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto reorder_prim = std::make_shared<reorder>("output", input_info("data"), p.expected_layout);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& reorder_node = prog.get_or_create(reorder_prim);
    program_wrapper::add_connection(prog, data_node, reorder_node);

    auto res = reorder_inst::calc_output_layouts<ov::PartialShape>(reorder_node, *reorder_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke_taylor, reorder_si_test,
    testing::ValuesIn(std::vector<reorder_si_test_params>{
        {
            layout{ov::PartialShape{2, 3}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 3, 1, 1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 1, 4, 5, 3}, data_types::f32, format::bfzyx},
            layout{ov::PartialShape{2, 1, 20, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 4, 5, 3}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 4, 1, 1, 5, 3}, data_types::f32, format::bfwzyx}
        },
        {
            layout{ov::PartialShape{2, 4, 5, 3}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 4, 1, 5, 3}, data_types::f32, format::b_fs_zyx_fsv16}
        },
        {
            layout{ov::PartialShape{2, 4}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{2, 4, 1, 1, 1}, data_types::f32, format::b_fs_zyx_fsv16}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx}
        },
    }));

}  // shape_infer_tests
