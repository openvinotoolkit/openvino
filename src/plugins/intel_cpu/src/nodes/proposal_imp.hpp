// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <vector>
#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

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

void enumerate_proposals_cpu(const float *bottom4d, const float *d_anchor4d, const float *anchors, float *proposals,
                             const int num_anchors, const int bottom_H, const int bottom_W, const float img_H,
                             const float img_W, const float min_box_H, const float min_box_W, const int feat_stride,
                             const float box_coordinate_scale, const float box_size_scale, float coordinates_offset,
                             bool initial_clip, bool swap_xy, bool clip_before_nms);

void unpack_boxes(const float *p_proposals, float *unpacked_boxes, int pre_nms_topn, bool store_prob);

void nms_cpu(const int num_boxes, int is_dead[], const float *boxes, int index_out[], std::size_t *const num_out,
             const int base_index, const float nms_thresh, const int max_num_out, float coordinates_offset);

void retrieve_rois_cpu(const int num_rois, const int item_index, const int num_proposals, const float *proposals,
                       const int roi_indices[], float *rois, int post_nms_topn_, bool normalize, float img_h,
                       float img_w, bool clip_after_nms, float *probs);

} // namespace node
} // namespace intel_cpu
} // namespace ov
