// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gather_elements_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct gather_elements_test_params {
    layout data_layout;
    layout indices_layout;
    int64_t axis;
    layout expected_layout;
};

class gather_elements_test : public testing::TestWithParam<gather_elements_test_params> { };

TEST_P(gather_elements_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto indices_layout_prim = std::make_shared<input_layout>("indices", p.indices_layout);
    auto gather_prim = std::make_shared<gather_elements>("output", input_info("data"), input_info("indices"), p.axis);

    cldnn::program prog(engine);

    auto& data_layout_node = prog.get_or_create(data_layout_prim);
    auto& indices_layout_node = prog.get_or_create(indices_layout_prim);
    auto& gather_elements_node = prog.get_or_create(gather_prim);
    program_wrapper::add_connection(prog, data_layout_node, gather_elements_node);
    program_wrapper::add_connection(prog, indices_layout_node, gather_elements_node);
    auto res = gather_elements_inst::calc_output_layouts<ov::PartialShape>(gather_elements_node, *gather_elements_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gather_elements_test,
    testing::ValuesIn(std::vector<gather_elements_test_params>{
        {
            layout{ov::PartialShape{3, 7, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 10, 5}, data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape{3, 10, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 10, 5}, data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape{3, 10, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{3, 7, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape{3, -1, 5}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
