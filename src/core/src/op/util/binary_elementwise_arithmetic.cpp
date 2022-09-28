// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::BinaryElementwiseArithmetic);

ov::op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob)
    : m_autob(autob) {}

ov::op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                       const Output<Node>& arg1,
                                                                       const AutoBroadcastSpec& autob)
    : Op({arg0, arg1}),
      m_autob(autob) {}

void ov::op::util::BinaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic(
    const op::AutoBroadcastSpec& autob) {
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, autob);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          args_et,
                          ").");

    set_output_type(0, args_et, args_pshape);
}

void ov::op::util::BinaryElementwiseArithmetic::validate_and_infer_types() {
    OV_OP_SCOPE(v0_util_BinaryElementwiseArithmetic_validate_and_infer_types);
    validate_and_infer_elementwise_arithmetic(m_autob);
}

bool ov::op::util::BinaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_util_BinaryElementwiseArithmetic_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}

bool ov::op::util::BinaryElementwiseArithmetic::evaluate_upper(const HostTensorVector& output_values) const {
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(output_values, 1));
    HostTensorVector lower_output_tensors;
    for (const auto& output : output_values)
        lower_output_tensors.push_back(
            std::make_shared<HostTensor>(output->get_element_type(), output->get_partial_shape()));
    if (!ngraph::interval_bound_evaluator(this, lower_output_tensors, output_values))
        return false;
    return true;
}

bool ov::op::util::BinaryElementwiseArithmetic::evaluate_lower(const HostTensorVector& output_values) const {
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(output_values, 1));
    HostTensorVector upper_output_tensors;
    for (const auto& output : output_values)
        upper_output_tensors.push_back(
            std::make_shared<HostTensor>(output->get_element_type(), output->get_partial_shape()));
    if (!ngraph::interval_bound_evaluator(this, output_values, upper_output_tensors))
        return false;
    return true;
}
