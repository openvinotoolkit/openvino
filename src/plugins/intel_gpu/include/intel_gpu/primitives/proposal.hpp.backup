// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

#define CLDNN_ROI_VECTOR_SIZE 5

struct proposal : public primitive_base<proposal> {
    CLDNN_DECLARE_PRIMITIVE(proposal)

    proposal(const primitive_id& id,
             const primitive_id& cls_scores,
             const primitive_id& bbox_pred,
             const primitive_id& image_info,
             int max_proposals,
             float iou_threshold,
             int min_bbox_size,
             int feature_stride,
             int pre_nms_topn,
             int post_nms_topn,
             const std::vector<float>& ratios_param,
             const std::vector<float>& scales_param,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, ext_prim_id, output_padding),
          max_proposals(max_proposals),
          iou_threshold(iou_threshold),
          base_bbox_size(16),
          min_bbox_size(min_bbox_size),
          feature_stride(feature_stride),
          pre_nms_topn(pre_nms_topn),
          post_nms_topn(post_nms_topn),
          ratios(ratios_param),
          scales(scales_param),
          coordinates_offset(1.0f),
          box_coordinate_scale(1.0f),
          box_size_scale(1.0f),
          for_deformable(false),
          swap_xy(false),
          initial_clip(false),
          clip_before_nms(true),
          clip_after_nms(false),
          round_ratios(true),
          shift_anchors(false),
          normalize(false) {}

    proposal(const primitive_id& id,
             const primitive_id& cls_scores,
             const primitive_id& bbox_pred,
             const primitive_id& image_info,
             int max_proposals,
             float iou_threshold,
             int base_bbox_size,
             int min_bbox_size,
             int feature_stride,
             int pre_nms_topn,
             int post_nms_topn,
             const std::vector<float>& ratios_param,
             const std::vector<float>& scales_param,
             float coordinates_offset,
             float box_coordinate_scale,
             float box_size_scale,
             bool for_deformable,
             bool swap_xy,
             bool initial_clip,
             bool clip_before_nms,
             bool clip_after_nms,
             bool round_ratios,
             bool shift_anchors,
             bool normalize,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, ext_prim_id, output_padding),
          max_proposals(max_proposals),
          iou_threshold(iou_threshold),
          base_bbox_size(base_bbox_size),
          min_bbox_size(min_bbox_size),
          feature_stride(feature_stride),
          pre_nms_topn(pre_nms_topn),
          post_nms_topn(post_nms_topn),
          ratios(ratios_param),
          scales(scales_param),
          coordinates_offset(coordinates_offset),
          box_coordinate_scale(box_coordinate_scale),
          box_size_scale(box_size_scale),
          for_deformable(for_deformable),
          swap_xy(swap_xy),
          initial_clip(initial_clip),
          clip_before_nms(clip_before_nms),
          clip_after_nms(clip_after_nms),
          round_ratios(round_ratios),
          shift_anchors(shift_anchors),
          normalize(normalize) {}

    proposal(const primitive_id& id,
             const primitive_id& cls_scores,
             const primitive_id& bbox_pred,
             const primitive_id& image_info,
             const primitive_id& second_output,
             int max_proposals,
             float iou_threshold,
             int base_bbox_size,
             int min_bbox_size,
             int feature_stride,
             int pre_nms_topn,
             int post_nms_topn,
             const std::vector<float>& ratios_param,
             const std::vector<float>& scales_param,
             float coordinates_offset,
             float box_coordinate_scale,
             float box_size_scale,
             bool for_deformable,
             bool swap_xy,
             bool initial_clip,
             bool clip_before_nms,
             bool clip_after_nms,
             bool round_ratios,
             bool shift_anchors,
             bool normalize,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
            : primitive_base(id, {cls_scores, bbox_pred, image_info, second_output}, ext_prim_id, output_padding),
              max_proposals(max_proposals),
              iou_threshold(iou_threshold),
              base_bbox_size(base_bbox_size),
              min_bbox_size(min_bbox_size),
              feature_stride(feature_stride),
              pre_nms_topn(pre_nms_topn),
              post_nms_topn(post_nms_topn),
              ratios(ratios_param),
              scales(scales_param),
              coordinates_offset(coordinates_offset),
              box_coordinate_scale(box_coordinate_scale),
              box_size_scale(box_size_scale),
              for_deformable(for_deformable),
              swap_xy(swap_xy),
              initial_clip(initial_clip),
              clip_before_nms(clip_before_nms),
              clip_after_nms(clip_after_nms),
              round_ratios(round_ratios),
              shift_anchors(shift_anchors),
              normalize(normalize) {}

    int max_proposals;
    float iou_threshold;
    int base_bbox_size;
    int min_bbox_size;
    int feature_stride;
    int pre_nms_topn;
    int post_nms_topn;
    std::vector<float> ratios;
    std::vector<float> scales;
    float coordinates_offset;
    float box_coordinate_scale;
    float box_size_scale;
    bool for_deformable;
    bool swap_xy;
    bool initial_clip;
    bool clip_before_nms;
    bool clip_after_nms;
    bool round_ratios;
    bool shift_anchors;
    bool normalize;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
