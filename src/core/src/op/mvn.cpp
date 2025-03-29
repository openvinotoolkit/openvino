// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn.hpp"

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/reference/mvn.hpp"

// ------------------------------ V0 ------------------------------
ov::op::v0::MVN::MVN(const Output<Node>& data, bool across_channels, bool normalize_variance, double eps)
    : Op({data}),
      m_eps{eps},
      m_across_channels{across_channels},
      m_normalize_variance{normalize_variance} {
    constructor_validate_and_infer_types();
}

ov::op::v0::MVN::MVN(const Output<Node>& data, AxisSet reduction_axes, bool normalize_variance, double eps)
    : Op({data}),
      m_eps{eps},
      m_across_channels{false},
      m_normalize_variance{normalize_variance},
      m_reduction_axes{reduction_axes} {
    constructor_validate_and_infer_types();
    const size_t chanelAxis = 1;
    m_across_channels = (m_reduction_axes.count(chanelAxis) > 0);
}

void ov::op::v0::MVN::validate_and_infer_types() {
    OV_OP_SCOPE(v0_MVN_validate_and_infer_types);
    // if m_across_channels is true we should calculate mean and variance per batch
    // else we calculate these per channel
    if (m_reduction_axes.empty() && input_value(0).get_partial_shape().rank().is_static()) {
        AxisSet reduction_axes;
        size_t start_axis = m_across_channels ? 1 : 2;
        for (int64_t i = start_axis; i < input_value(0).get_partial_shape().rank().get_length(); ++i) {
            reduction_axes.insert(i);
        }
        set_reduction_axes(reduction_axes);
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> ov::op::v0::MVN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_MVN_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the MVN op but got ",
                          new_args.size());
    return std::make_shared<op::v0::MVN>(new_args.at(0), m_reduction_axes, m_normalize_variance, m_eps);
}

bool ov::op::v0::MVN::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_MVN_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("across_channels", m_across_channels);
    visitor.on_attribute("normalize_variance", m_normalize_variance);
    visitor.on_attribute("reduction_axes", m_reduction_axes);
    return true;
}

// ------------------------------ V6 ------------------------------

namespace ov {
template <>
OPENVINO_API EnumNames<ov::op::MVNEpsMode>& EnumNames<ov::op::MVNEpsMode>::get() {
    static auto enum_names = EnumNames<ov::op::MVNEpsMode>(
        "op::MVNEpsMode",
        {{"OUTSIDE_SQRT", ov::op::MVNEpsMode::OUTSIDE_SQRT}, {"INSIDE_SQRT", ov::op::MVNEpsMode::INSIDE_SQRT}});
    return enum_names;
}

AttributeAdapter<op::MVNEpsMode>::~AttributeAdapter() = default;
}  // namespace ov

std::ostream& ov::op::operator<<(std::ostream& s, const ov::op::MVNEpsMode& type) {
    return s << as_string(type);
}

ov::op::v6::MVN::MVN(const Output<Node>& data,
                     const Output<Node>& reduction_axes,
                     bool normalize_variance,
                     float eps,
                     MVNEpsMode eps_mode)
    : Op({data, reduction_axes}),
      m_normalize_variance{normalize_variance},
      m_eps{eps},
      m_eps_mode{eps_mode} {
    constructor_validate_and_infer_types();
}

void ov::op::v6::MVN::validate_and_infer_types() {
    OV_OP_SCOPE(v6_MVN_validate_and_infer_types);
    const auto& data = get_input_partial_shape(0);
    const auto& axes = get_input_partial_shape(1);

    if (axes.is_static()) {
        NODE_VALIDATION_CHECK(this, is_vector(axes.to_shape()), "Expected 1D tensor for the 'axes' input. Got: ", axes);

        const auto data_rank = data.rank();
        NODE_VALIDATION_CHECK(this,
                              data_rank.is_dynamic() || cmp::ge(data_rank.get_length(), axes.get_shape()[0]),
                              "Expected rank for the 'data' input to be higher than axes shape. Got: ",
                              data);
    }

    set_output_type(0, get_input_element_type(0), data);
}

std::shared_ptr<ov::Node> ov::op::v6::MVN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_MVN_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 2,
                          "Expected 2 element in new_args for the MVN op but got ",
                          new_args.size());
    return std::make_shared<op::v6::MVN>(new_args.at(0), new_args.at(1), m_normalize_variance, m_eps, m_eps_mode);
}

bool ov::op::v6::MVN::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_MVN_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("normalize_variance", m_normalize_variance);
    visitor.on_attribute("eps_mode", m_eps_mode);
    return true;
}

namespace mvn {
namespace {
template <ov::element::Type_t ET>
bool evaluate(ov::TensorVector& outputs,
              const ov::TensorVector& inputs,
              bool normalize_variance,
              float eps,
              ov::op::MVNEpsMode eps_mode) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::AxisSet reduction_axes;
    auto rank = inputs[0].get_shape().size();
    if (inputs[1].get_element_type() == ov::element::i64) {
        reduction_axes = ov::reference::mvn_6_reduction_axes<int64_t>(inputs[1], rank);
    } else if (inputs[1].get_element_type() == ov::element::i32) {
        reduction_axes = ov::reference::mvn_6_reduction_axes<int32_t>(inputs[1], rank);
    } else {
        OPENVINO_THROW("Unexpected indices type");
    }
    ov::reference::mvn_6<T>(inputs[0].data<T>(),
                            outputs[0].data<T>(),
                            inputs[0].get_shape(),
                            reduction_axes,
                            normalize_variance,
                            eps,
                            eps_mode);
    return true;
}

bool evaluate_mvn(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  bool normalize_variance,
                  float eps,
                  ov::op::MVNEpsMode eps_mode) {
    bool rc = true;
    switch (inputs[0].get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_mvn, f32, outputs, inputs, normalize_variance, eps, eps_mode);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace mvn

bool ov::op::v6::MVN::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v6_MVN_evaluate);
    return mvn::evaluate_mvn(outputs, inputs, get_normalize_variance(), get_eps(), get_eps_mode());
}

bool ov::op::v6::MVN::has_evaluate() const {
    OV_OP_SCOPE(v6_MVN_has_evaluate);
    switch (get_input_element_type(0)) {
    case ov::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
