// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/detection_output.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "detection_output_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

const int background_label_id = 0;
const float nms_threshold = 0.5f;
const float eta = 1.f;
const float confidence_threshold = 0.3f;
const prior_box_code_type code_type = prior_box_code_type::corner;
const int input_width = 1;
const int input_height = 1;
const bool decrease_label_id = false;
const bool clip_before_nms = false;
const bool clip_after_nms = false;
const float objectness_score = 0.0f;

struct detection_output_test_params {
    std::vector<layout> in_layouts;
    int num_classes;
    int top_k;
    bool variance_encoded_in_target;
    int keep_top_k;
    bool share_location;
    bool normalized;
    layout expected_layout;
};

class detection_output_test : public testing::TestWithParam<detection_output_test_params> { };

TEST_P(detection_output_test, shape_infer) {
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

    int prior_info_size = p.normalized != 0 ? 4 : 5;
    int prior_coordinates_offset = p.normalized != 0 ? 0 : 1;

    auto detection_output_prim = std::make_shared<detection_output>("detection_output",
                                                                    input_prim_ids,
                                                                    p.num_classes,
                                                                    p.keep_top_k,
                                                                    p.share_location,
                                                                    background_label_id,
                                                                    nms_threshold,
                                                                    p.top_k,
                                                                    eta,
                                                                    code_type,
                                                                    p.variance_encoded_in_target,
                                                                    confidence_threshold,
                                                                    prior_info_size,
                                                                    prior_coordinates_offset,
                                                                    p.normalized,
                                                                    input_width,
                                                                    input_height,
                                                                    decrease_label_id,
                                                                    clip_before_nms,
                                                                    clip_after_nms,
                                                                    objectness_score);
    auto& detection_output_node = prog.get_or_create(detection_output_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, detection_output_node);
    }

    auto res = detection_output_inst::calc_output_layouts<ov::PartialShape>(detection_output_node, *detection_output_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, detection_output_test,
    testing::ValuesIn(std::vector<detection_output_test_params>{
        {
            {layout{ov::PartialShape{1, 60}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1, 165}, data_types::f32, format::bfyx},
             layout{ov::PartialShape{1, 1, 60}, data_types::f32, format::bfyx}},
            11, 75, true, 50, true, true,
            layout{ov::PartialShape{1, 1, 50, 7}, data_types::f32, format::bfyx}
        },
        {
            {layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx},
             layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx},
             layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx}},
            11, 75, true, 50, true, true,
            layout{ov::PartialShape{1, 1, ov::Dimension::dynamic(), 7}, data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
