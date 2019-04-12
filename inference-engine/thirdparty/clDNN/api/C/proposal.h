/*
// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef PROPOSAL_H
#define PROPOSAL_H

#include "cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

#define CLDNN_ROI_VECTOR_SIZE 5

CLDNN_BEGIN_PRIMITIVE_DESC(proposal)
    int max_proposals;
    float iou_threshold;
    int base_bbox_size;
    int min_bbox_size;
    int feature_stride;
    int pre_nms_topn;
    int post_nms_topn;
    cldnn_float_arr ratios;
    cldnn_float_arr scales;
    float coordinates_offset;
    float box_coordinate_scale;
    float box_size_scale;
    uint32_t swap_xy;
    uint32_t initial_clip;
    uint32_t clip_before_nms;
    uint32_t clip_after_nms;
    uint32_t round_ratios;
    uint32_t shift_anchors;
    uint32_t normalize;
CLDNN_END_PRIMITIVE_DESC(proposal)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(proposal);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif /* PROPOSAL_H */
