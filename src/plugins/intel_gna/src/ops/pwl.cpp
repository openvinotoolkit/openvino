// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "reference/pwl.hpp"

namespace ov {
namespace intel_gna {
namespace op {

Pwl::Pwl(const ngraph::Output<ngraph::Node>& data,
         const ngraph::Output<ngraph::Node>& m,
         const ngraph::Output<ngraph::Node>& b,
         const ngraph::Output<ngraph::Node>& knots)
    : Op({data, m, b, knots}) {
    constructor_validate_and_infer_types();
}

bool Pwl::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<ngraph::Node> Pwl::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Pwl>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void Pwl::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

template <typename T1, typename T2>
bool Pwl::evaluate(ov::TensorVector& outputs,
                   const ov::TensorVector& inputs) const {
    using A1 = typename ov::element_type_traits<T1::value>::value_type;
    using A2 = typename ov::element_type_traits<T2::value>::value_type;
    ov::intel_gna::op::reference::pwl(inputs[0].data<A2>(),
                                      outputs[0].data<A2>(),
                                      shape_size(get_input_shape(0)),
                                      inputs[1].data<A1>(),
                                      inputs[2].data<A1>(),
                                      inputs[3].data<A1>(),
                                      shape_size(get_input_shape(1)));
    return true;
}

bool Pwl::evaluate(ov::TensorVector& outputs,
                   const ov::TensorVector& inputs,
                   const ov::EvaluationContext& evaluation_context) const {
    return evaluate_pwl(std::tuple<std::integral_constant<ov::element::Type_t, ov::element::f32>,
                                   std::integral_constant<ov::element::Type_t, ov::element::f64>>(),
                        std::tuple<std::integral_constant<ov::element::Type_t, ov::element::i32>,
                                   std::integral_constant<ov::element::Type_t, ov::element::i64>,
                                   std::integral_constant<ov::element::Type_t, ov::element::u32>,
                                   std::integral_constant<ov::element::Type_t, ov::element::u64>,
                                   std::integral_constant<ov::element::Type_t, ov::element::f16>,
                                   std::integral_constant<ov::element::Type_t, ov::element::f32>,
                                   std::integral_constant<ov::element::Type_t, ov::element::f64>>(),
                        outputs,
                        inputs);
}

bool Pwl::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ov::element::i32:
    case ov::element::i64:
    case ov::element::u32:
    case ov::element::u64:
    case ov::element::f16:
    case ov::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

void Pwl::set_base_node(const std::shared_ptr<ngraph::Node>& base_node) {
    m_base_node = base_node;
}

std::shared_ptr<ngraph::Node> Pwl::get_base_node() {
    return m_base_node;
}

} // namespace op
} // namespace intel_gna
} // namespace ov
