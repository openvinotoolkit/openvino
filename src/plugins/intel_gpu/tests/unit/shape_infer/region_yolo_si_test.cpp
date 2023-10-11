// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/region_yolo.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "region_yolo_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct region_yolo_test_params {
    layout in_layout;
    uint32_t coords;
    uint32_t classes;
    uint32_t num;
    std::vector<int64_t> mask;
    int32_t axis;
    int32_t end_axis;
    bool do_softmax;
    layout expected_layout;
};

class region_yolo_test : public testing::TestWithParam<region_yolo_test_params> { };

TEST_P(region_yolo_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_layout_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto region_yolo_prim = std::make_shared<region_yolo>("region_yolo",
                                                          input_info("input"),
                                                          p.coords,
                                                          p.classes,
                                                          p.num,
                                                          p.mask,
                                                          static_cast<uint32_t>(p.mask.size()),
                                                          p.axis,
                                                          p.end_axis,
                                                          p.do_softmax);

    cldnn::program prog(engine);

    auto& input_layout_node = prog.get_or_create(input_layout_prim);
    auto& region_yolo_node = prog.get_or_create(region_yolo_prim);
    program_wrapper::add_connection(prog, input_layout_node, region_yolo_node);
    auto res = region_yolo_inst::calc_output_layouts<ov::PartialShape>(region_yolo_node, *region_yolo_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, region_yolo_test,
    testing::ValuesIn(std::vector<region_yolo_test_params>{
        {
            layout{ov::PartialShape{1, 255, 26, 26}, data_types::f32, format::bfyx},
            4, 80, 6, { 0, 1, 2 }, 1, 3, false,
            layout{ov::PartialShape{1, 255, 26, 26}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            4, 80, 6, { 0, 1, 2 }, 1, 3, false,
            layout{ov::PartialShape{-1, 255, -1, -1}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},
            4, 80, 6, { 0, 1, 2 }, 1, 3, true,
            layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
