// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include <vector>
#include "intel_gpu/graph/serialization/utils.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"

namespace cldnn {
#define CLDNN_ROI_VECTOR_SIZE 5

struct proposal : public primitive_base<proposal> {
    CLDNN_DECLARE_PRIMITIVE(proposal)

    proposal() : primitive_base("", {}),
                 max_proposals(0),
                 iou_threshold(0.0f),
                 base_bbox_size(16),
                 min_bbox_size(0),
                 feature_stride(0),
                 pre_nms_topn(0),
                 post_nms_topn(0),
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
             int min_bbox_size,
             int feature_stride,
             int pre_nms_topn,
             int post_nms_topn,
             const std::vector<float>& ratios_param,
             const std::vector<float>& scales_param)
        : primitive_base(id, {cls_scores, bbox_pred, image_info}),
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
             data_types output_data_type = data_types::f32,
             const size_t num_outputs = 1)
        : primitive_base(id, {cls_scores, bbox_pred, image_info}, num_outputs, {optional_data_type{output_data_type}}),
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
             bool normalize)
            : primitive_base(id, {cls_scores, bbox_pred, image_info, second_output}),
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

        membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
            save(ob);
        }
        seed = hash_range(seed, mem_buf.begin(), mem_buf.end());

        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const proposal>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(max_proposals) &&
               cmp_fields(iou_threshold) &&
               cmp_fields(base_bbox_size) &&
               cmp_fields(min_bbox_size) &&
               cmp_fields(feature_stride) &&
               cmp_fields(pre_nms_topn) &&
               cmp_fields(post_nms_topn) &&
               cmp_fields(ratios) &&
               cmp_fields(scales) &&
               cmp_fields(coordinates_offset) &&
               cmp_fields(box_coordinate_scale) &&
               cmp_fields(box_size_scale) &&
               cmp_fields(for_deformable) &&
               cmp_fields(swap_xy) &&
               cmp_fields(initial_clip) &&
               cmp_fields(clip_before_nms) &&
               cmp_fields(clip_after_nms) &&
               cmp_fields(round_ratios) &&
               cmp_fields(shift_anchors) &&
               cmp_fields(normalize);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<proposal>::save(ob);
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
        primitive_base<proposal>::load(ib);
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
