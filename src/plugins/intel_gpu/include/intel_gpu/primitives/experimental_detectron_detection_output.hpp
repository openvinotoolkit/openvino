// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <utility>
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief experimental detectron detection output
struct experimental_detectron_detection_output : public primitive_base<experimental_detectron_detection_output> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_detection_output)

    experimental_detectron_detection_output() : primitive_base("", {}) {}

    /// @brief Constructs experimental_detectron_detection_output primitive
    /// @param id This primitive id
    /// @param input_rois input rois
    /// @param input_deltas input deltas
    /// @param input_scores input scores
    /// @param input_im_info image info
    /// @param output_classes ROI scores
    /// @param output_scores  minimum box width and height
    /// @param score_threshold a threshold to consider only detections whose score are larger than the threshold
    /// @param nms_threshold a threshold to be used in the NMS stage
    /// @param num_classes the number of detected classes
    /// @param post_nms_count the maximum number of detections per class
    /// @param max_detections_per_image the maximum number of detections per image
    /// @param class_agnostic_box_regression specifies whether to delete background classes or not
    /// @param max_delta_log_wh the maximum delta of logarithms for width and height
    /// @param deltas_weights the weights for bounding boxes sizes deltas
    experimental_detectron_detection_output(const primitive_id& id,
                                            const input_info& input_rois,
                                            const input_info& input_deltas,
                                            const input_info& input_scores,
                                            const input_info& input_im_info,
                                            float score_threshold,
                                            float nms_threshold,
                                            int num_classes,
                                            int post_nms_count,
                                            int max_detections_per_image,
                                            bool class_agnostic_box_regression,
                                            float max_delta_log_wh,
                                            std::vector<float> deltas_weights)
        : primitive_base{id, {input_rois, input_deltas, input_scores, input_im_info}},
          score_threshold{score_threshold},
          nms_threshold{nms_threshold},
          num_classes{num_classes},
          post_nms_count{post_nms_count},
          max_detections_per_image{max_detections_per_image},
          class_agnostic_box_regression{class_agnostic_box_regression},
          max_delta_log_wh{max_delta_log_wh},
          deltas_weights{std::move(deltas_weights)} {}

    float score_threshold = 0.0f;
    float nms_threshold = 0.0f;
    int num_classes = 0;
    int post_nms_count = 0;
    int max_detections_per_image = 0;
    bool class_agnostic_box_regression = false;
    float max_delta_log_wh  = 0.0f;
    std::vector<float> deltas_weights;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, score_threshold);
        seed = hash_combine(seed, nms_threshold);
        seed = hash_combine(seed, num_classes);
        seed = hash_combine(seed, post_nms_count);
        seed = hash_combine(seed, max_detections_per_image);
        seed = hash_combine(seed, class_agnostic_box_regression);
        seed = hash_combine(seed, max_delta_log_wh);
        seed = hash_range(seed, deltas_weights.begin(), deltas_weights.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const experimental_detectron_detection_output>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(score_threshold) &&
               cmp_fields(nms_threshold) &&
               cmp_fields(num_classes) &&
               cmp_fields(post_nms_count) &&
               cmp_fields(max_detections_per_image) &&
               cmp_fields(class_agnostic_box_regression) &&
               cmp_fields(max_delta_log_wh) &&
               cmp_fields(deltas_weights);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<experimental_detectron_detection_output>::save(ob);
        ob << score_threshold;
        ob << nms_threshold;
        ob << num_classes;
        ob << post_nms_count;
        ob << max_detections_per_image;
        ob << class_agnostic_box_regression;
        ob << max_delta_log_wh;
        ob << deltas_weights;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<experimental_detectron_detection_output>::load(ib);
        ib >> score_threshold;
        ib >> nms_threshold;
        ib >> num_classes;
        ib >> post_nms_count;
        ib >> max_detections_per_image;
        ib >> class_agnostic_box_regression;
        ib >> max_delta_log_wh;
        ib >> deltas_weights;
    }
};
}  // namespace cldnn
