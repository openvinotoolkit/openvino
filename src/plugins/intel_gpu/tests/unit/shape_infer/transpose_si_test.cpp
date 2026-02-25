// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "permute_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct transpose_test_params {
    layout data_layout;
    std::vector<uint16_t> permute_order_data;
    layout expected_layout;
};

class transpose_test : public testing::TestWithParam<transpose_test_params> { };

TEST_P(transpose_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto permute_prim = std::make_shared<permute>("output", input_info("data"), p.permute_order_data);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& permute_node = prog.get_or_create(permute_prim);
    program_wrapper::add_connection(prog, data_node, permute_node);

    auto res = permute_inst::calc_output_layouts<ov::PartialShape>(permute_node, *permute_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, transpose_test,
    testing::ValuesIn(std::vector<transpose_test_params>{
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
            {2, 0, 1},
            layout{ov::PartialShape{4, 2, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 3, 4}, data_types::f32, format::bfyx},
            {},
            layout{ov::PartialShape{4, 3, 2}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(), data_types::f32, format::bfyx},
            {0, 1, 2},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            {},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
