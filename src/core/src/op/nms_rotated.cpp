// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include <cstring>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "nms_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/nms_rotated.hpp"

namespace ov {

namespace op {
namespace nms_rotated {
namespace validate {
namespace {
void input_types(const Node* op) {
    const auto inputs_size = op->get_input_size();

    NODE_VALIDATION_CHECK(op, inputs_size == 5, "Expected 5 inputs to be provided.");

    NODE_VALIDATION_CHECK(op,
                          op->get_input_element_type(0).is_real() || op->get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(op,
                          op->get_input_element_type(1).is_real() || op->get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(
        op,
        op->get_input_element_type(2).is_integral_number() || op->get_input_element_type(0).is_dynamic(),
        "Expected integer number type as element type for the 'max_output_boxes_per_class' input.");

    NODE_VALIDATION_CHECK(op,
                          op->get_input_element_type(3).is_real() || op->get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the "
                          "'iou_threshold' input.");

    NODE_VALIDATION_CHECK(op,
                          op->get_input_element_type(4).is_real() || op->get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the "
                          "'score_threshold_ps' input.");
}
}  // namespace
}  // namespace validate
}  // namespace nms_rotated
}  // namespace op
// ------------------------------ v13 ------------------------------

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                const Output<Node>& scores,
                                const Output<Node>& max_output_boxes_per_class,
                                const Output<Node>& iou_threshold,
                                const Output<Node>& score_threshold,
                                const bool sort_result_descending,
                                const element::Type& output_type,
                                const bool clockwise)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type},
      m_clockwise{clockwise} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v13::NMSRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_NMSRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 5, "Number of inputs must be 5");

    return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 m_sort_result_descending,
                                                 m_output_type,
                                                 m_clockwise);
}

bool op::v13::NMSRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_NMSRotated_visit_attributes);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("clockwise", m_clockwise);
    return true;
}

void op::v13::NMSRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v13_NMSRotated_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto output_shapes = shape_infer(this, input_shapes);

    nms_rotated::validate::input_types(this);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    set_output_type(0, m_output_type, output_shapes[0]);
    set_output_type(1, element::f32, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

// Temporary evaluate, for testing purpose
OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
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
    bool clockwise = true;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NMSRotated produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes, const Shape& boxes_shape) {
    auto result = host_tensor_2_vector<float>(boxes);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = host_tensor_2_vector<float>(scores);
    return result;
}

InfoForNMSRotated get_info_for_nms_eval(const op::v13::NMSRotated* nms,
                                        const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMSRotated result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? host_tensor_2_vector<int64_t>(inputs[2])[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? host_tensor_2_vector<float>(inputs[3])[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? host_tensor_2_vector<float>(inputs[4])[0] : 0.0f;
    result.soft_nms_sigma = 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();
    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();
    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);
    result.out_shape_size = shape_size(result.out_shape);
    result.sort_result_descending = nms->get_sort_result_descending();
    result.output_type = nms->get_output_type();
    result.clockwise = nms->get_clockwise();

    return result;
}
}  // namespace

bool op::v13::NMSRotated::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    auto info = get_info_for_nms_eval(this, inputs);

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

    auto selected_scores_type = (outputs.size() < 3) ? element::f32 : outputs[1]->get_element_type();

    ov::reference::nms_rotated::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

bool op::v13::NMSRotated::has_evaluate() const {
    OV_OP_SCOPE(v3_nms_rotated_has_evaluate);
    switch (get_input_element_type(0)) {
    case ov::element::bf16:
    case ov::element::f16:
    case ov::element::f32:
    case ov::element::f64:
        return true;
    default:
        break;
    }
    return false;
}
OPENVINO_SUPPRESS_DEPRECATED_END

}  // namespace ov
