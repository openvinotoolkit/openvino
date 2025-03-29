// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"

#include <vector>

namespace cldnn {

/// @brief Performs non max suppression of input boxes and returns indices of selected boxes.
/// @detail Filters out boxes that have high intersection-over-union (IOU) with previously
/// selected boxes with higher score. Boxes with score higher than score_threshold are
/// filtered out. This filtering happens per class.
struct non_max_suppression : public primitive_base<non_max_suppression> {
    CLDNN_DECLARE_PRIMITIVE(non_max_suppression)

    enum Rotation {
        NONE,
        CLOCKWISE,
        COUNTERCLOCKWISE
    };

    non_max_suppression() : primitive_base("", {}),
                            selected_indices_num(0),
                            center_point_box(false),
                            sort_result_descending(false) {}

    /// @brief Creates non max suppression primitive.
    /// @param id This primitive id.
    /// @param boxes_positions Id of primitive with bounding boxes.
    /// @param boxes_score Id of primitive with boxes scores per class.
    /// @param selected_indices_num Number of selected indices.
    /// @param center_point_box If true boxes are represented as [center x, center y, width, height].
    /// @param sort_result_descending Specifies whether it is necessary to sort selected boxes across batches or not.
    /// @param num_select_per_class Id of primitive producing number of boxes to select per class.
    /// @param iou_threshold Id of primitive producing threshold value for IOU.
    /// @param score_threshold Id of primitive producing threshold value for scores.
    /// @param soft_nms_sigma Id of primitive specifying the sigma parameter for Soft-NMS.
    /// @param second_output Id of primitive specifying output for scores for each selected box.
    /// @param third_output Id of primitive specifying output for total number of selected boxes.
    non_max_suppression(const primitive_id& id,
                        const input_info& boxes_positions,
                        const input_info& boxes_score,
                        int selected_indices_num,
                        bool center_point_box = false,
                        bool sort_result_descending = true,
                        const primitive_id& num_select_per_class = primitive_id(),
                        const primitive_id& iou_threshold = primitive_id(),
                        const primitive_id& score_threshold = primitive_id(),
                        const primitive_id& soft_nms_sigma = primitive_id(),
                        const primitive_id& second_output = primitive_id(),
                        const primitive_id& third_output = primitive_id(),
                        const size_t num_outputs = 1)
        : primitive_base(id, {boxes_positions, boxes_score}, num_outputs, {optional_data_type()})
        , selected_indices_num(selected_indices_num)
        , center_point_box(center_point_box)
        , sort_result_descending(sort_result_descending)
        , num_select_per_class(num_select_per_class)
        , iou_threshold(iou_threshold)
        , score_threshold(score_threshold)
        , soft_nms_sigma(soft_nms_sigma)
        , second_output(second_output)
        , third_output(third_output) {}

    int selected_indices_num;
    bool center_point_box;
    bool sort_result_descending;
    primitive_id num_select_per_class;
    primitive_id iou_threshold;
    primitive_id score_threshold;
    primitive_id soft_nms_sigma;
    primitive_id second_output;
    primitive_id third_output;
    Rotation rotation{Rotation::NONE};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, center_point_box);
        seed = hash_combine(seed, sort_result_descending);
        seed = hash_combine(seed, num_select_per_class.empty());
        seed = hash_combine(seed, iou_threshold.empty());
        seed = hash_combine(seed, score_threshold.empty());
        seed = hash_combine(seed, soft_nms_sigma.empty());
        seed = hash_combine(seed, second_output.empty());
        seed = hash_combine(seed, third_output.empty());
        seed = hash_combine(seed, rotation);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const non_max_suppression>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(selected_indices_num) &&
               cmp_fields(center_point_box) &&
               cmp_fields(sort_result_descending) &&
               cmp_fields(num_select_per_class.empty()) &&
               cmp_fields(iou_threshold.empty()) &&
               cmp_fields(score_threshold.empty()) &&
               cmp_fields(soft_nms_sigma.empty()) &&
               cmp_fields(second_output.empty()) &&
               cmp_fields(third_output.empty()) &&
               cmp_fields(rotation);
        #undef cmp_fields
    }

    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!num_select_per_class.empty())
            ret.push_back(num_select_per_class);
        if (!iou_threshold.empty())
            ret.push_back(iou_threshold);
        if (!score_threshold.empty())
            ret.push_back(score_threshold);
        if (!soft_nms_sigma.empty())
            ret.push_back(soft_nms_sigma);
        if (!second_output.empty())
            ret.push_back(second_output);
        if (!third_output.empty())
            ret.push_back(third_output);

        return ret;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<non_max_suppression>::save(ob);
        ob << selected_indices_num;
        ob << center_point_box;
        ob << sort_result_descending;
        ob << num_select_per_class;
        ob << iou_threshold;
        ob << score_threshold;
        ob << soft_nms_sigma;
        ob << second_output;
        ob << third_output;
        ob << make_data(&rotation, sizeof(rotation));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<non_max_suppression>::load(ib);
        ib >> selected_indices_num;
        ib >> center_point_box;
        ib >> sort_result_descending;
        ib >> num_select_per_class;
        ib >> iou_threshold;
        ib >> score_threshold;
        ib >> soft_nms_sigma;
        ib >> second_output;
        ib >> third_output;
        ib >> make_data(&rotation, sizeof(rotation));
    }
};

struct non_max_suppression_gather : primitive_base<non_max_suppression_gather> {
    CLDNN_DECLARE_PRIMITIVE(non_max_suppression_gather)

    non_max_suppression_gather() : primitive_base("", {}) {}

    /// @brief Constructs non_max_suppression_gather primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    non_max_suppression_gather(const primitive_id& id,
                  const std::vector<input_info>& inputs,
                  const size_t num_outputs = 1)
        : primitive_base(id, inputs, num_outputs, {optional_data_type()}) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        return true;
    }
};
}  // namespace cldnn
