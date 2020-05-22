// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo Operation::type_info;

//! [op:ctor]
Operation::Operation(const ngraph::Output<ngraph::Node> &arg, int64_t add) : Op({arg}), add(add) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Operation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> Operation::copy_with_new_args(const ngraph::NodeVector &new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }

    return std::make_shared<Operation>(new_args.at(0), add);
}
//! [op:copy]

//! [op:visit_attributes]
bool Operation::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("add", add);
    return true;
}
//! [op:visit_attributes]
