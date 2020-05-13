// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo OutShapeOfReshape::type_info;

OutShapeOfReshape::OutShapeOfReshape(
        const Output<Node>& inDataShape,
        const Output<Node>& outShapeDescriptor,
        bool specialZero) : Op({inDataShape, outShapeDescriptor}), m_specialZero(specialZero) {
    constructor_validate_and_infer_types();
}

void OutShapeOfReshape::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have only 2 inputs, provided: ", get_input_size());

    const auto& inDataShapeTensorShape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic input data shape");
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have input data shape tensor with rank 1, provided: ",
                          inDataShapeTensorShape.rank().get_length());

    const auto& outShapeDescriptorTensorShape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic output shape descriptor");
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have output shape descriptor tensor with rank 1, provided: ",
                          outShapeDescriptorTensorShape.rank().get_length());

    const auto& inDataShapeTensorType = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          inDataShapeTensorType.is_static() &&
                          inDataShapeTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") input data type needs to be an integral type. Got: ",
                          inDataShapeTensorType);
    const auto& outShapeDescriptorTensorType = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          outShapeDescriptorTensorType.is_static() &&
                          outShapeDescriptorTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") shape descriptor type needs to be an integral type. Got: ",
                          outShapeDescriptorTensorType);

    set_output_type(0, element::i64, outShapeDescriptorTensorShape);
}

std::shared_ptr<Node> OutShapeOfReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<OutShapeOfReshape>(new_args.at(0), new_args.at(1), m_specialZero);
}

bool OutShapeOfReshape::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("special_zero", m_specialZero);
    return true;
}


}  // namespace op
}  // namespace vpu
}  // namespace ngraph
