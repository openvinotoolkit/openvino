// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include <vector>

namespace cldnn {


#define CLDNN_ROI_VECTOR_SIZE 5

struct proposal : public primitive_base<proposal> {
    CLDNN_DECLARE_PRIMITIVE(proposal)

    proposal() : primitive_base("", {}) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION

    proposal(const primitive_id& id,
             const input_info& cls_scores,
             const input_info& bbox_pred,
             const input_info& image_info,
             int max_proposals,
             float iou_threshold,
             int min_bbox_size,
             int feature_stride,
             int pre_nms_topn,
             int post_nms_topn,
             const std::vector<float>& ratios_param,
             const std::vector<float>& scales_param,
             const padding& output_padding = padding())
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, {output_padding}),
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
             const input_info& cls_scores,
             const input_info& bbox_pred,
             const input_info& image_info,
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
             const padding& output_padding = padding())
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, {output_padding}),
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
             const input_info& cls_scores,
             const input_info& bbox_pred,
             const input_info& image_info,
             const input_info& second_output,
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
             const padding& output_padding = padding())
            : primitive_base(id, {cls_scores, bbox_pred, image_info, second_output}, {output_padding}),
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

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, max_proposals);
        seed = hash_combine(seed, iou_threshold);
        seed = hash_combine(seed, base_bbox_size);
        seed = hash_combine(seed, min_bbox_size);
        seed = hash_combine(seed, feature_stride);
        seed = hash_combine(seed, pre_nms_topn);
        seed = hash_combine(seed, post_nms_topn);
        seed = hash_range(seed, ratios.begin(), ratios.end());
        seed = hash_range(seed, scales.begin(), scales.end());
        seed = hash_combine(seed, coordinates_offset);
        seed = hash_combine(seed, box_coordinate_scale);
        seed = hash_combine(seed, box_size_scale);
        seed = hash_combine(seed, for_deformable);
        seed = hash_combine(seed, swap_xy);
        seed = hash_combine(seed, initial_clip);
        seed = hash_combine(seed, clip_before_nms);
        seed = hash_combine(seed, clip_after_nms);
        seed = hash_combine(seed, round_ratios);
        seed = hash_combine(seed, shift_anchors);
        seed = hash_combine(seed, normalize);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << max_proposals;
        ob << iou_threshold;
        ob << base_bbox_size;
        ob << min_bbox_size;
        ob << feature_stride;
        ob << pre_nms_topn;
        ob << post_nms_topn;
        ob << ratios;
        ob << scales;
        ob << coordinates_offset;
        ob << box_coordinate_scale;
        ob << box_size_scale;
        ob << for_deformable;
        ob << swap_xy;
        ob << initial_clip;
        ob << clip_before_nms;
        ob << clip_after_nms;
        ob << round_ratios;
        ob << shift_anchors;
        ob << normalize;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> max_proposals;
        ib >> iou_threshold;
        ib >> base_bbox_size;
        ib >> min_bbox_size;
        ib >> feature_stride;
        ib >> pre_nms_topn;
        ib >> post_nms_topn;
        ib >> ratios;
        ib >> scales;
        ib >> coordinates_offset;
        ib >> box_coordinate_scale;
        ib >> box_size_scale;
        ib >> for_deformable;
        ib >> swap_xy;
        ib >> initial_clip;
        ib >> clip_before_nms;
        ib >> clip_after_nms;
        ib >> round_ratios;
        ib >> shift_anchors;
        ib >> normalize;
    }
};

}  // namespace cldnn
