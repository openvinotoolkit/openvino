// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"

#include <cstring>

#include "itt.hpp"
#include "matrix_nms_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {

op::v8::MatrixNms::MatrixNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : Op({boxes, scores}),
      m_attrs{attrs} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MatrixNms::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_MatrixNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MatrixNms>(new_args.at(0), new_args.at(1), m_attrs);
}

void op::v8::MatrixNms::validate() {
    OV_OP_SCOPE(v8_MatrixNms_validate);

    const auto& nms_attrs = this->get_attrs();
    const auto output_type = nms_attrs.output_type;

    auto is_float_type_admissible = [](const element::Type& t) {
        return t == element::f32 || t == element::f16 || t == element::bf16;
    };

    // validate dtype of each input
    NODE_VALIDATION_CHECK(this,
                          output_type == element::i64 || output_type == element::i32,
                          "Output type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(this,
                          this->get_input_element_type(0).compatible(this->get_input_element_type(1)),
                          "Expected 'boxes', 'scores' type is same.");
}

void op::v8::MatrixNms::validate_and_infer_types() {
    OV_OP_SCOPE(v8_MatrixNms_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    validate();

    const auto& output_type = get_attrs().output_type;
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, output_type, output_shapes[1]);
    set_output_type(2, output_type, output_shapes[2]);
}

bool op::v8::MatrixNms::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_MatrixNms_visit_attributes);

    visitor.on_attribute("sort_result_type", m_attrs.sort_result_type);
    visitor.on_attribute("output_type", m_attrs.output_type);
    visitor.on_attribute("nms_top_k", m_attrs.nms_top_k);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("sort_result_across_batch", m_attrs.sort_result_across_batch);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("background_class", m_attrs.background_class);
    visitor.on_attribute("decay_function", m_attrs.decay_function);
    visitor.on_attribute("gaussian_sigma", m_attrs.gaussian_sigma);
    visitor.on_attribute("post_threshold", m_attrs.post_threshold);
    visitor.on_attribute("normalized", m_attrs.normalized);

    return true;
}

void op::v8::MatrixNms::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}

std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v8::MatrixNms::DecayFunction>& EnumNames<op::v8::MatrixNms::DecayFunction>::get() {
    static auto enum_names =
        EnumNames<op::v8::MatrixNms::DecayFunction>("op::v8::MatrixNms::DecayFunction",
                                                    {{"gaussian", op::v8::MatrixNms::DecayFunction::GAUSSIAN},
                                                     {"linear", op::v8::MatrixNms::DecayFunction::LINEAR}});
    return enum_names;
}

std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::SortResultType& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v8::MatrixNms::SortResultType>& EnumNames<op::v8::MatrixNms::SortResultType>::get() {
    static auto enum_names =
        EnumNames<op::v8::MatrixNms::SortResultType>("op::v8::MatrixNms::SortResultType",
                                                     {{"classid", op::v8::MatrixNms::SortResultType::CLASSID},
                                                      {"score", op::v8::MatrixNms::SortResultType::SCORE},
                                                      {"none", op::v8::MatrixNms::SortResultType::NONE}});
    return enum_names;
}

AttributeAdapter<op::v8::MatrixNms::DecayFunction>::~AttributeAdapter() = default;
AttributeAdapter<op::v8::MatrixNms::SortResultType>::~AttributeAdapter() = default;
}  // namespace ov
