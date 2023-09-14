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

namespace ov {
namespace {
constexpr size_t max_output_boxes_port = 2;
constexpr size_t iou_threshold_port = 3;
constexpr size_t score_threshold_port = 4;
constexpr size_t soft_nms_sigma_port = 5;

inline bool is_float_type_admissible(const element::Type& t) {
    return t == element::dynamic || t == element::f32 || t == element::f16 || t == element::bf16;
}
}  // namespace
namespace op {
namespace nms {
namespace validate {
namespace {
void input_types(const Node* op) {
    NODE_VALIDATION_CHECK(op,
                          is_float_type_admissible(op->get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(op,
                          is_float_type_admissible(op->get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");
    const auto inputs_size = op->get_input_size();
    if (inputs_size > 3) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(3)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'iou_threshold' input.");
    }

    if (inputs_size > 4) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(4)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'score_threshold_ps' input.");
    }

    if (inputs_size > 5) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(5)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'soft_nms_sigma' input.");
    }
}
}  // namespace
}  // namespace validate
}  // namespace nms
}  // namespace op
// ------------------------------ v13 ------------------------------
op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const Output<Node>& soft_nms_sigma,
                                             const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v13::NMSRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_NMSRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 6,
                          "Number of inputs must be 2, 3, 4, 5 or 6");

    switch (new_args.size()) {
    case 2:
        return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                           new_args.at(1),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 3:
        return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 4:
        return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 5:
        return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    default:
        return std::make_shared<op::v13::NMSRotated>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           new_args.at(5),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    }
}

int64_t op::v13::NMSRotated::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    if (inputs().size() < 3) {
        return 0;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto max_output_boxes_input = get_constant_from_source(input_value(max_output_boxes_port));
    OPENVINO_SUPPRESS_DEPRECATED_END
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

float op::v13::NMSRotated::iou_threshold_from_input() const {
    float iou_threshold = 0.0f;

    if (inputs().size() < 4) {
        return iou_threshold;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto iou_threshold_input = get_constant_from_source(input_value(iou_threshold_port));
    OPENVINO_SUPPRESS_DEPRECATED_END
    iou_threshold = iou_threshold_input->cast_vector<float>().at(0);

    return iou_threshold;
}

float op::v13::NMSRotated::score_threshold_from_input() const {
    float score_threshold = 0.0f;

    if (inputs().size() < 5) {
        return score_threshold;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto score_threshold_input = get_constant_from_source(input_value(score_threshold_port));
    OPENVINO_SUPPRESS_DEPRECATED_END
    score_threshold = score_threshold_input->cast_vector<float>().at(0);

    return score_threshold;
}

float op::v13::NMSRotated::soft_nms_sigma_from_input() const {
    float soft_nms_sigma = 0.0f;

    if (inputs().size() < 6) {
        return soft_nms_sigma;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto soft_nms_sigma_input = get_constant_from_source(input_value(soft_nms_sigma_port));
    OPENVINO_SUPPRESS_DEPRECATED_END
    soft_nms_sigma = soft_nms_sigma_input->cast_vector<float>().at(0);

    return soft_nms_sigma;
}

bool op::v13::NMSRotated::is_soft_nms_sigma_constant_and_default() const {
    auto soft_nms_sigma_node = input_value(soft_nms_sigma_port).get_node_shared_ptr();
    if (inputs().size() < 6 || !op::util::is_constant(soft_nms_sigma_node)) {
        return false;
    }
    const auto soft_nms_sigma_input = as_type_ptr<op::v0::Constant>(soft_nms_sigma_node);
    return soft_nms_sigma_input->cast_vector<float>().at(0) == 0.0f;
}

bool op::v13::NMSRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_NMSRotated_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v13::NMSRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v13_NMSRotated_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto output_shapes = shape_infer(this, input_shapes);

    nms::validate::input_types(this);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    set_output_type(0, m_output_type, output_shapes[0]);
    set_output_type(1, element::f32, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

std::ostream& operator<<(std::ostream& s, const op::v13::NMSRotated::BoxEncodingType& type) {
    return s << as_string(type);
}

template <>
NGRAPH_API EnumNames<op::v13::NMSRotated::BoxEncodingType>&
EnumNames<op::v13::NMSRotated::BoxEncodingType>::get() {
    static auto enum_names = EnumNames<op::v13::NMSRotated::BoxEncodingType>(
        "op::v13::NMSRotated::BoxEncodingType",
        {{"corner", op::v13::NMSRotated::BoxEncodingType::CORNER},
         {"center", op::v13::NMSRotated::BoxEncodingType::CENTER}});
    return enum_names;
}
}  // namespace ov
