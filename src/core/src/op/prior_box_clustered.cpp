// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/prior_box_clustered.hpp"

#include "itt.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "prior_box_clustered_shape_inference.hpp"

using namespace std;

namespace ov {
op::v0::PriorBoxClustered::PriorBoxClustered(const Output<Node>& layer_shape,
                                             const Output<Node>& image_shape,
                                             const Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::v0::PriorBoxClustered::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PriorBoxClustered_validate_and_infer_types);

    const auto input_shapes = prior_box::validate::inputs_et(this);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.widths.size() == m_attrs.heights.size(),
                          "Size of heights vector: ",
                          m_attrs.heights.size(),
                          " doesn't match size of widths vector: ",
                          m_attrs.widths.size());

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::f32, output_shapes.front());
    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v0::PriorBoxClustered::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PriorBoxClustered_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClustered>(new_args.at(0), new_args.at(1), m_attrs);
}

bool op::v0::PriorBoxClustered::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PriorBoxClustered_visit_attributes);

    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("step_w", m_attrs.step_widths);
    visitor.on_attribute("step_h", m_attrs.step_heights);
    visitor.on_attribute("width", m_attrs.widths);
    visitor.on_attribute("height", m_attrs.heights);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variances);
    return true;
}

namespace prior_box_clustered {
namespace {
template <element::Type_t ET>
bool evaluate(const Tensor& arg0, const Tensor& arg1, Tensor& out, op::v0::PriorBoxClustered::Attributes attrs) {
    using T = fundamental_type_for<ET>;
    ov::reference::prior_box_clustered(arg0.data<T>(), arg1.data<T>(), out.data<float>(), out.get_shape(), attrs);
    return true;
}

bool evaluate_prior_box(const Tensor& arg0,
                        const Tensor& arg1,
                        Tensor& out,
                        const op::v0::PriorBoxClustered::Attributes& attrs) {
    bool rc = true;
    switch (arg0.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace prior_box_clustered

bool op::v0::PriorBoxClustered::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_PriorBoxClustered_evaluate);
    return prior_box_clustered::evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v0::PriorBoxClustered::has_evaluate() const {
    OV_OP_SCOPE(v0_PriorBoxClustered_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        break;
    }
    return false;
}

void op::v0::PriorBoxClustered::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace ov
