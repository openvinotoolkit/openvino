// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/proposal.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "proposal_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

const float iou_threshold = 0.7f;
const int base_bbox_size = 16;
const int min_bbox_size = 12;
const int feature_stride = 16;
const int pre_nms_topn = 6000;
const int post_nms_topn = 300;
const float coordinates_offset = 1.0f;
const float box_coordinate_scale = 1.0f;
const float box_size_scale = 1.0f;
const bool swap_xy = false;
const bool initial_clip = false;
const bool clip_before_nms = true;
const bool clip_after_nms = false;
const bool round_ratios = true;
const bool shift_anchors = false;
const bool normalize = true;
const std::vector<float> ratios = { 0.5f, 1.0f, 2.0f };
const std::vector<float> scales = { 2.0f, 4.0f, 8.0f, 16.0f, 32.0f };

struct proposal_test_params {
    std::vector<layout> in_layouts;
    data_types output_data_type;
    size_t num_outputs;
    std::vector<layout> expected_layouts;
};

class proposal_test : public testing::TestWithParam<proposal_test_params> { };

TEST_P(proposal_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    cldnn::program prog(engine);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    for (size_t i = 0; i < p.in_layouts.size(); i++) {
        auto prim_id = "input" + std::to_string(i);
        auto input_layout_prim = std::make_shared<input_layout>(prim_id, p.in_layouts[i]);
        input_prims.push_back(input_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto proposal_prim = std::make_shared<proposal>("depth_to_space",
                                                    input_prim_ids[0],
                                                    input_prim_ids[1],
                                                    input_prim_ids[2],
                                                    0,
                                                    iou_threshold,
                                                    base_bbox_size,
                                                    min_bbox_size,
                                                    feature_stride,
                                                    pre_nms_topn,
                                                    post_nms_topn,
                                                    ratios,
                                                    scales,
                                                    coordinates_offset,
                                                    box_coordinate_scale,
                                                    box_size_scale,
                                                    false,
                                                    swap_xy,
                                                    initial_clip,
                                                    clip_before_nms,
                                                    clip_after_nms,
                                                    round_ratios,
                                                    shift_anchors,
                                                    normalize,
                                                    p.output_data_type,
                                                    p.num_outputs);
    std::vector<padding> output_paddings;
    std::vector<optional_data_type> output_data_types;
    for (size_t i = 0; i < p.num_outputs; i++) {
        output_paddings.push_back(padding());
        output_data_types.push_back(optional_data_type{p.output_data_type});
    }
    proposal_prim->output_paddings = output_paddings;
    proposal_prim->output_data_types = output_data_types;
    auto& proposal_node = prog.get_or_create(proposal_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, proposal_node);
    }

    auto res = proposal_inst::calc_output_layouts<ov::PartialShape>(proposal_node, *proposal_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), p.num_outputs);
    for (size_t i = 0; i < p.expected_layouts.size(); i++)
        ASSERT_EQ(res[i], p.expected_layouts[i]);
}

INSTANTIATE_TEST_SUITE_P(smoke, proposal_test,
    testing::ValuesIn(std::vector<proposal_test_params>{
        {
            {layout{ov::PartialShape{-1, 30, -1, -1}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{-1, 60, -1, -1}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{3}, data_types::f32, format::bfyx}},
             data_types::f32, 2,
            {layout{ov::PartialShape{ov::Dimension::dynamic(), 5}, data_types::f32, format::bfyx},
             layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx}}
        },
        {
            {layout{ov::PartialShape{1, 24, -1, -1}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1, 48, -1, -1}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{3}, data_types::f32, format::bfyx}},
             data_types::f32, 2,
            {layout{ov::PartialShape{300, 5}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{300}, data_types::f32, format::bfyx}}
        },
    }));

}  // shape_infer_tests
