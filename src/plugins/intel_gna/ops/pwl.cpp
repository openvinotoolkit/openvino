// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "reference/pwl.hpp"

NGRAPH_RTTI_DEFINITION(GNAPluginNS::Pwl, "Pwl", 0);

namespace GNAPluginNS {

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
bool Pwl::evaluate(const ov::HostTensorVector& outputs,
                   const ov::HostTensorVector& inputs) const {
    outputs[0]->set_unary(inputs[0]);
    using A1 = typename ov::element_type_traits<T1::value>::value_type;
    using A2 = typename ov::element_type_traits<T2::value>::value_type;
    GNAPluginNS::runtime::reference::pwl(inputs[0]->get_data_ptr<A2>(),
                                         outputs[0]->get_data_ptr<A2>(),
                                         shape_size(get_input_shape(0)),
                                         inputs[1]->get_data_ptr<A1>(),
                                         inputs[2]->get_data_ptr<A1>(),
                                         inputs[3]->get_data_ptr<A1>(),
                                         shape_size(get_input_shape(1)));
    return true;
}

bool Pwl::evaluate(const ov::HostTensorVector& outputs,
                   const ov::HostTensorVector& inputs) const {
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

} // namespace GNAPluginNS
