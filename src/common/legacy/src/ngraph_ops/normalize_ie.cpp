// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/normalize_ie.hpp"

#include <memory>
#include <string>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::NormalizeIE);

op::NormalizeIE::NormalizeIE(const Output<Node>& data, const Output<Node>& weights, float eps, bool across_spatial,
                             bool channel_shared, const ngraph::element::Type output_type)
    : Op({data, weights}), m_eps(eps), m_across_spatial(across_spatial), m_channel_shared(channel_shared), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void op::NormalizeIE::validate_and_infer_types() {
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, m_output_type, arg_shape);

    const PartialShape& input_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_shape.rank().is_dynamic() || (input_shape.rank().get_length() >= 2 && input_shape.rank().get_length() <= 4),
                          "Argument must have rank >= 2 and <= 4 (argument shape: ", input_shape, ").");
}

shared_ptr<Node> op::NormalizeIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::NormalizeIE>(new_args.at(0), new_args.at(1), m_eps, m_across_spatial, m_channel_shared, m_output_type);
}

bool op::NormalizeIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("channel_shared", m_channel_shared);
    visitor.on_attribute("across_spatial", m_across_spatial);
    return true;
}
