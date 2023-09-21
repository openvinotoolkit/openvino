// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/nms_rotated.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "openvino/reference/non_max_suppression.hpp"

using namespace ov;

namespace nms_v13 {

struct InfoForNMSRotated {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    element::Type output_type;
    bool clockwise;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const ov::TensorVector& inputs, size_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port].get_shape();
    const auto scores_ps = inputs[scores_port].get_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.size() > 0 && scores_ps.size() > 0) {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto num_boxes = num_boxes_boxes;
        const auto num_classes = scores_ps[1];

        result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0];
    }
    return result;
}

InfoForNMSRotated get_info_for_nms_eval(const std::shared_ptr<op::v13::NMSRotated>& nms,
                                        const ov::TensorVector& inputs) {
    InfoForNMSRotated result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port].get_shape();
    result.scores_shape = inputs[scores_port].get_shape();
    result.boxes_data = get_floats(inputs[boxes_port], result.boxes_shape);
    result.scores_data = get_floats(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);
    result.sort_result_descending = nms->get_sort_result_descending();
    result.output_type = nms->get_output_type();
    result.clockwise = nms->get_clockwise();
    return result;
}
}  // namespace nms_v13

template <element::Type_t ET>
bool evaluate(const std::shared_ptr<op::v13::NMSRotated>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto info = nms_v13::get_info_for_nms_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ov::reference::nms_rotated::non_max_suppression(info.boxes_data.data(),
                                                    info.boxes_shape,
                                                    info.scores_data.data(),
                                                    info.scores_shape,
                                                    info.max_output_boxes_per_class,
                                                    info.iou_threshold,
                                                    info.score_threshold,
                                                    info.soft_nms_sigma,
                                                    selected_indices.data(),
                                                    info.out_shape,
                                                    selected_scores.data(),
                                                    info.out_shape,
                                                    &valid_outputs,
                                                    info.sort_result_descending,
                                                    info.clockwise);

    auto selected_scores_type = (outputs.size() < 2) ? element::f32 : outputs[1].get_element_type();

    ov::reference::nms_rotated::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

template <>
bool evaluate_node<op::v13::NMSRotated>(std::shared_ptr<Node> node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<op::v1::Select>(node) || ov::is_type<op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case element::Type_t::boolean:
        return evaluate<element::Type_t::boolean>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::bf16:
        return evaluate<element::Type_t::bf16>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::f16:
        return evaluate<element::Type_t::f16>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::f64:
        return evaluate<element::Type_t::f64>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::f32:
        return evaluate<element::Type_t::f32>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::i4:
        return evaluate<element::Type_t::i4>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::i8:
        return evaluate<element::Type_t::i8>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::i16:
        return evaluate<element::Type_t::i16>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::i32:
        return evaluate<element::Type_t::i32>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::i64:
        return evaluate<element::Type_t::i64>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u1:
        return evaluate<element::Type_t::u1>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u4:
        return evaluate<element::Type_t::u4>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u8:
        return evaluate<element::Type_t::u8>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u16:
        return evaluate<element::Type_t::u16>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u32:
        return evaluate<element::Type_t::u32>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    case element::Type_t::u64:
        return evaluate<element::Type_t::u64>(ov::as_type_ptr<op::v13::NMSRotated>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
