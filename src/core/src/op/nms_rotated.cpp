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
}
}  // namespace
}  // namespace validate
}  // namespace nms
}  // namespace op
// ------------------------------ v13 ------------------------------

op::v13::NMSRotated::NMSRotated(const Output<Node>& boxes,
                                const Output<Node>& scores,
                                const Output<Node>& max_output_boxes_per_class,
                                const Output<Node>& iou_threshold,
                                const Output<Node>& score_threshold,
                                const op::v13::NMSRotated::BoxEncodingType box_encoding,
                                const bool sort_result_descending,
                                const element::Type& output_type,
                                const bool clockwise)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
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
                                                 m_box_encoding,
                                                 m_sort_result_descending,
                                                 m_output_type,
                                                 m_clockwise);
}

bool op::v13::NMSRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_NMSRotated_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
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
NGRAPH_API EnumNames<op::v13::NMSRotated::BoxEncodingType>& EnumNames<op::v13::NMSRotated::BoxEncodingType>::get() {
    static auto enum_names =
        EnumNames<op::v13::NMSRotated::BoxEncodingType>("op::v13::NMSRotated::BoxEncodingType",
                                                        {{"corner", op::v13::NMSRotated::BoxEncodingType::CORNER},
                                                         {"center", op::v13::NMSRotated::BoxEncodingType::CENTER}});
    return enum_names;
}
}  // namespace ov
