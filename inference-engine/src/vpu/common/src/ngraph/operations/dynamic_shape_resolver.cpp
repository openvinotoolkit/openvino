// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

namespace ngraph { namespace op {

constexpr NodeTypeInfo DynamicShapeResolver::type_info;

DynamicShapeResolver::DynamicShapeResolver(const Output<Node>& tensorWithData, const Output<Node>& tensorWithDims)
    : Op(OutputVector{tensorWithData, tensorWithDims}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> DynamicShapeResolver::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicShapeResolver>(new_args.at(0), new_args.at(1));
}

void DynamicShapeResolver::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2, "(", get_friendly_name(), ") supports only ", 2, " inputs, but ", get_input_size(), " provided");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static(), "(", get_friendly_name(), ") does not support dynamic shape for data tensor");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(1).is_static(), "(", get_friendly_name(), ") does not support dynamic shape for dims tensor");

    const auto& dimsElementType = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this, dimsElementType.is_integral_number() && dimsElementType.is_static(), "(", get_friendly_name(), ") supports only integral "
        "number type for dims tensor, but ", dimsElementType, " provided");

    const auto& dataShape = get_input_shape(0);
    const auto& dimsShape = get_input_shape(1);
    NODE_VALIDATION_CHECK(this, dimsShape.size() == 1 && dimsShape.front() == dataShape.size(), "(", get_friendly_name(), ") inputs shapes mismatch: first "
        "input shape = ", dataShape, " second input shape = ", dimsShape, " but ", dataShape, " and ", Shape{dataShape.size()}, " are expected");

    set_output_type(0, get_input_element_type(0), get_input_shape(0));
}

bool DynamicShapeResolver::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

}  // namespace op
}  // namespace ngraph
