// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/fake_quant_internal.hpp"

#include <memory>
#include <ngraph/opsets/opset5.hpp>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::internal::FakeQuantInternal);

op::internal::FakeQuantInternal::FakeQuantInternal(const Output<Node>& x,
                                                   const Output<Node>& scale,
                                                   const std::string& op_type,
                                                   const int bit_length)
    : Op({x, scale}),
      m_op_type(op_type),
      m_bit_length(bit_length) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::FakeQuantInternal::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    return make_shared<FakeQuantInternal>(new_args.at(0), new_args.at(1), m_op_type, m_bit_length);
}

bool op::internal::FakeQuantInternal::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("op_type", m_op_type);
    visitor.on_attribute("bit_length", m_bit_length);
    return true;
}

void op::internal::FakeQuantInternal::validate_and_infer_types() {
    const auto x_ps = get_input_partial_shape(0);

    set_output_type(0, get_input_element_type(0), x_ps);
}
