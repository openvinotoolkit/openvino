// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"

namespace ov {
namespace intel_gna {
namespace op {

Identity::Identity(const ngraph::Output<ngraph::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> Identity::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Identity>(new_args.at(0));
}

void Identity::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool Identity::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

} // namespace op
} // namespace intel_gna
} // namespace ov
