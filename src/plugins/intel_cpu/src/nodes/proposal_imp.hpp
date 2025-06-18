// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>

namespace ov::Extensions::Cpu {

struct proposal_conf {
    size_t feat_stride_ = 0UL;
    size_t base_size_ = 0UL;
    size_t min_size_ = 0UL;
    int pre_nms_topn_ = 0;
    int post_nms_topn_ = 0;
    float nms_thresh_ = 0.0F;
    float box_coordinate_scale_ = 0.0F;
    float box_size_scale_ = 0.0F;
    std::vector<float> scales;
    std::vector<float> ratios;
    bool normalize_ = false;

    size_t anchors_shape_0 = 0UL;

    // Framework specific parameters
    float coordinates_offset = 0.0F;
    bool swap_xy = false;
    bool initial_clip = false;     // clip initial bounding boxes
    bool clip_before_nms = false;  // clip bounding boxes before nms step
    bool clip_after_nms = false;   // clip bounding boxes after nms step
    bool round_ratios = false;     // round ratios during anchors generation stage
    bool shift_anchors = false;    // shift anchors by half size of the box
};

namespace XARCH {

void proposal_exec(const float* input0,
                   const float* input1,
                   std::vector<size_t> dims0,
                   std::array<float, 4> img_info,
                   const float* anchors,
                   int* roi_indices,
                   float* output0,
                   float* output1,
                   proposal_conf& conf);

}  // namespace XARCH
}  // namespace ov::Extensions::Cpu
