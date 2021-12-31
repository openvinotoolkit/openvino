// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/fake_dequant_internal.hpp"

#include <memory>
#include <ngraph/opsets/opset5.hpp>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::internal::FakeDequantInternal);

op::internal::FakeDequantInternal::FakeDequantInternal(const Output<Node>& x,
                                                       const Output<Node>& scale,
                                                       const int bit_length,
                                                       const float max_range)
    : Op({x, scale}),
      m_bit_length(bit_length),
      m_max_range(max_range) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::FakeDequantInternal::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    return make_shared<FakeDequantInternal>(new_args.at(0), new_args.at(1), m_bit_length, m_max_range);
}

bool op::internal::FakeDequantInternal::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("bit_length", m_bit_length);
    visitor.on_attribute("max_range", m_max_range);
    return true;
}

void op::internal::FakeDequantInternal::validate_and_infer_types() {
    const auto x_ps = get_input_partial_shape(0);

    set_output_type(0, get_input_element_type(0), x_ps);
}
