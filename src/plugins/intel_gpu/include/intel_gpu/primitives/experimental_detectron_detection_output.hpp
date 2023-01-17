// Copyright (C) 2022-2023 Intel Corporation
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
                                            const input_info& output_classes,
                                            const input_info& output_scores,
                                            float score_threshold,
                                            float nms_threshold,
                                            int num_classes,
                                            int post_nms_count,
                                            int max_detections_per_image,
                                            bool class_agnostic_box_regression,
                                            float max_delta_log_wh,
                                            std::vector<float> deltas_weights,
                                            const padding& output_padding = {})
        : primitive_base{id,
                         {input_rois, input_deltas, input_scores, input_im_info, output_classes, output_scores},
                         {output_padding}},
          output_classes{output_classes.pid},
          output_scores{output_scores.pid},
          score_threshold{score_threshold},
          nms_threshold{nms_threshold},
          num_classes{num_classes},
          post_nms_count{post_nms_count},
          max_detections_per_image{max_detections_per_image},
          class_agnostic_box_regression{class_agnostic_box_regression},
          max_delta_log_wh{max_delta_log_wh},
          deltas_weights{std::move(deltas_weights)} {}

    primitive_id output_classes;
    primitive_id output_scores;
    float score_threshold;
    float nms_threshold;
    int num_classes;
    int post_nms_count;
    int max_detections_per_image;
    bool class_agnostic_box_regression;
    float max_delta_log_wh;
    std::vector<float> deltas_weights;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!output_classes.empty())
            ret.emplace_back(output_classes);

        if (!output_scores.empty())
            ret.emplace_back(output_scores);

        return ret;
    }
};
}  // namespace cldnn
