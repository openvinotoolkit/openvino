// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "shape_of_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct shape_of_test_params {
    layout data_layout;
    data_types out_dt;
    layout expected_layout;
};

class shape_of_test : public testing::TestWithParam<shape_of_test_params> { };

TEST_P(shape_of_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto shape_of_prim = std::make_shared<shape_of>("output", input_info("data"), p.out_dt);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& shape_of_node = prog.get_or_create(shape_of_prim);
    program_wrapper::add_connection(prog, data_node, shape_of_node);

    auto res = shape_of_inst::calc_output_layouts<ov::PartialShape>(shape_of_node, *shape_of_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, shape_of_test,
    testing::ValuesIn(std::vector<shape_of_test_params>{
        {
            layout{ov::PartialShape{1000, 256, 10, 15}, data_types::f32, format::bfyx}, data_types::i64,
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx},
        },
        {
            layout{ov::PartialShape{3, 5}, data_types::f32, format::bfyx}, data_types::i32,
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
        },
        {
            layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}, data_types::i64,
            layout{ov::PartialShape{2}, data_types::i64, format::bfyx},
        }
    }));

}  // shape_infer_tests
