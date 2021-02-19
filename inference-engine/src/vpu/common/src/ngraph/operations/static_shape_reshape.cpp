// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include "vpu/ngraph/operations/static_shape_reshape.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <ngraph/validation_util.hpp>

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeReshape::type_info;

StaticShapeReshape::StaticShapeReshape(const Output<Node>& arg, const Output<Node>& pattern, bool special_zero)
    : ::ngraph::opset3::Reshape(arg, pattern, special_zero) {
    constructor_validate_and_infer_types();
}

StaticShapeReshape::StaticShapeReshape(const std::shared_ptr<ngraph::opset3::Reshape>& reshape)
    : StaticShapeReshape(reshape->input_value(0), reshape->input_value(1), reshape->get_special_zero()) {
}

void StaticShapeReshape::validate_and_infer_types() {
    if (get_output_partial_shape(0).is_static()) {
        return;
    }

    opset3::Reshape::validate_and_infer_types();

    set_input_is_relevant_to_shape(1);
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static(), "StaticShapeReshape (", get_friendly_name(), ") ",
                          "input#0 is expected to be of static shape, got: ", get_input_partial_shape(0));

    auto outputShape = get_output_partial_shape(0);
    NODE_VALIDATION_CHECK(this, outputShape.rank().is_static(), "StaticShapeReshape (", get_friendly_name(), ") ",
                          "output is expected to be of static rank");
    for (size_t i = 0; i < outputShape.rank().get_length(); i++) {
        outputShape[i] = outputShape[i].get_max_length();
    }

    NODE_VALIDATION_CHECK(this, outputShape.is_static(),
                          "StaticShapeReshape (", get_friendly_name(), ") can't evaluate output shape");

    set_output_type(0, get_input_element_type(0), outputShape);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
