// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/bucketize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "roi_align_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct roi_align_test_params {
    layout input_layout;
    int num_roi;
    int pooled_h;
    int pooled_w;
    layout expected_layout;
};

class roi_align_test : public testing::TestWithParam<roi_align_test_params> { };

TEST_P(roi_align_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.input_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", layout{ov::PartialShape{p.num_roi, 4}, p.input_layout.data_type, p.input_layout.format});
    auto input2_layout_prim = std::make_shared<input_layout>("input2", layout{ov::PartialShape{p.num_roi}, p.input_layout.data_type, p.input_layout.format});
    auto inputs = std::vector<input_info>{ input_info("input0"), input_info("input1"), input_info("input2")};
    auto roi_align_prim = std::make_shared<roi_align>("output", inputs,
                                p.pooled_h, p.pooled_w, 2, 1.0f,
                                roi_align::PoolingMode::avg, roi_align::AlignedMode::half_pixel_for_nn);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& input2_layout_node = prog.get_or_create(input2_layout_prim);
    auto& roi_align_node = prog.get_or_create(roi_align_prim);
    program_wrapper::add_connection(prog, input0_layout_node, roi_align_node);
    program_wrapper::add_connection(prog, input1_layout_node, roi_align_node);
    program_wrapper::add_connection(prog, input2_layout_node, roi_align_node);

    auto res = roi_align_inst::calc_output_layouts<ov::PartialShape>(roi_align_node, *roi_align_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, roi_align_test,
    testing::ValuesIn(std::vector<roi_align_test_params>{
        { layout{ov::PartialShape{1, 2, 3, 4},      data_types::f16, format::bfyx},     -1, 3, 3, layout{ov::PartialShape{-1, 2,  3, 3},    data_types::f16, format::bfyx}},
        { layout{ov::PartialShape{1, 2, 3, 4},      data_types::f32, format::bfyx},     10, 2, 2, layout{ov::PartialShape{10, 2,  2, 2},    data_types::f32, format::bfyx}},
        { layout{ov::PartialShape::dynamic(4),      data_types::f16, format::bfyx},     10, 7, 7, layout{ov::PartialShape{10, -1, 7, 7},    data_types::f16, format::bfyx}},
        { layout{ov::PartialShape::dynamic(4),      data_types::f32, format::bfyx},     -1, 2, 2, layout{ov::PartialShape{-1, -1, 2, 2},    data_types::f32, format::bfyx}}
    }));

}  // shape_infer_tests
