// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/roi_pooling.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "roi_pooling_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

const float position_sensitive = false;
const pooling_mode mode = pooling_mode::bilinear;

struct roi_pooling_test_params {
    layout in0_layout;
    layout in1_layout;
    int pooled_height;
    int pooled_width;
    float spatial_scale;
    layout expected_layout;
};

class roi_pooling_test : public testing::TestWithParam<roi_pooling_test_params> { };

TEST_P(roi_pooling_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.in0_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", p.in1_layout);
    auto roi_pooling_prim = std::make_shared<roi_pooling>("roi_pooling",
                                                          input_info("input0"),
                                                          input_info("input1"),
                                                          mode,
                                                          position_sensitive,
                                                          p.pooled_width,
                                                          p.pooled_height,
                                                          p.spatial_scale,
                                                          0, 1, 1);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);

    auto& roi_pooling_node = prog.get_or_create(roi_pooling_prim);
    program_wrapper::add_connection(prog, input0_layout_node, roi_pooling_node);
    program_wrapper::add_connection(prog, input1_layout_node, roi_pooling_node);

    auto res = roi_pooling_inst::calc_output_layouts<ov::PartialShape>(roi_pooling_node, *roi_pooling_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, roi_pooling_test,
    testing::ValuesIn(std::vector<roi_pooling_test_params>{
        {
            layout{ov::PartialShape{1, 3, 8, 8}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{3, 5}, data_types::f32, format::bfyx},
            3, 3, 0.625f,
            layout{ov::PartialShape{3, 3, 3, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            layout{ov::PartialShape{-1 , 5}, data_types::f32, format::bfyx},
            5, 5, 1.f,
            layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5}, data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
