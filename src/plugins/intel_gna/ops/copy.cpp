// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "copy.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

#include <cmath>
#include <cstddef>

NGRAPH_RTTI_DEFINITION(GNAPluginNS::Copy, "Copy", 0);

namespace GNAPluginNS {

Copy::Copy() : m_is_delayed_copy(false) {
}

Copy::Copy(const ngraph::Output<ngraph::Node>& arg, bool is_delayed_copy) : Op({arg}), m_is_delayed_copy(is_delayed_copy) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> Copy::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Copy>(new_args.at(0), m_is_delayed_copy);
}

void Copy::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool Copy::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("m_is_delayed_copy", m_is_delayed_copy);
    return true;
}

} // namespace GNAPluginNS
