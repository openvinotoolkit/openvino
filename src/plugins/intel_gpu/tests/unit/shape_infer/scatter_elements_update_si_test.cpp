// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "scatter_elements_update_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct scatter_elements_update_test_params {
    layout data_layout;
    layout indices_layout;
    layout updates_layout;
    int64_t axis;
    layout expected_layout;
};

class scatter_elements_update_test : public testing::TestWithParam<scatter_elements_update_test_params> { };

TEST_P(scatter_elements_update_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto data_layout_prim = std::make_shared<input_layout>("data", p.data_layout);
    auto indices_layout_prim = std::make_shared<input_layout>("indices", p.indices_layout);
    auto updates_layout_prim = std::make_shared<input_layout>("updates", p.updates_layout);
    auto scatter_elements_update_prim = std::make_shared<scatter_elements_update>("output", input_info("data"), input_info("indices"), input_info("updates"), p.axis);

    cldnn::program prog(engine);

    auto& data_node = prog.get_or_create(data_layout_prim);
    auto& incides_node = prog.get_or_create(indices_layout_prim);
    auto& updates_node = prog.get_or_create(updates_layout_prim);
    auto& scatter_elements_update_node = prog.get_or_create(scatter_elements_update_prim);
    program_wrapper::add_connection(prog, data_node, scatter_elements_update_node);
    program_wrapper::add_connection(prog, incides_node, scatter_elements_update_node);
    program_wrapper::add_connection(prog, updates_node, scatter_elements_update_node);

    auto res = scatter_elements_update_inst::calc_output_layouts<ov::PartialShape>(scatter_elements_update_node, *scatter_elements_update_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, scatter_elements_update_test,
    testing::ValuesIn(std::vector<scatter_elements_update_test_params>{
        {
            layout{ov::PartialShape{1000, 256, 7, 7}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{125, 20, 7, 6}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{125, 20, 7, 6}, data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape{1000, 256, 7, 7}, data_types::f32, format::bfyx},
        },
        {
            layout{ov::PartialShape{3, 5}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 2}, data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape{3, 5}, data_types::f32, format::bfyx},
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            1,
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
        }
    }));

}  // shape_infer_tests
