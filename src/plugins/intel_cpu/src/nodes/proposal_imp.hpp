// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>

namespace ov::Extensions::Cpu {

struct proposal_conf {
    size_t feat_stride_;
    size_t base_size_;
    size_t min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float box_coordinate_scale_;
    float box_size_scale_;
    std::vector<float> scales;
    std::vector<float> ratios;
    bool normalize_;

    size_t anchors_shape_0;

    // Framework specific parameters
    float coordinates_offset;
    bool swap_xy;
    bool initial_clip;     // clip initial bounding boxes
    bool clip_before_nms;  // clip bounding boxes before nms step
    bool clip_after_nms;   // clip bounding boxes after nms step
    bool round_ratios;     // round ratios during anchors generation stage
    bool shift_anchors;    // shift anchors by half size of the box
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
