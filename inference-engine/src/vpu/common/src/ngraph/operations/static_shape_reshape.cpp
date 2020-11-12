// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include "vpu/ngraph/operations/static_shape_reshape.hpp"
#include "vpu/ngraph/utilities.hpp"

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
    const auto& targetShape = input_value(1);
    if (as_type_ptr<opset3::Constant>(targetShape.get_node_shared_ptr())) {
        opset3::Reshape::validate_and_infer_types();
        return;
    }

    NODE_VALIDATION_CHECK(this, get_input_element_type(1).is_integral_number(), "Pattern must be an integral number.");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(1).rank().compatible(1), "Pattern must have rank 1, got ", get_input_partial_shape(1).rank(), ".");

    set_input_is_relevant_to_shape(1);

    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static(), "StaticShapeReshape (", get_friendly_name(), ") ",
                          "input#0 is expected to be of static shape, got: ", get_input_partial_shape(0));

    if (get_output_partial_shape(0).is_static()) {
        return;
    }

    const auto& inputShape = get_input_shape(0);

    auto outputDimensionsValues = ::vpu::evaluateTargetShape(targetShape);
    NODE_VALIDATION_CHECK(this, !outputDimensionsValues.empty(), "StaticShapeReshape (", get_friendly_name(), ") can't evaluate output shape");

    for (std::size_t i = 0; i < outputDimensionsValues.size(); ++i) {
        if (outputDimensionsValues[i] == 0 && m_special_zero) {
            NODE_VALIDATION_CHECK(this, inputShape[i] <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
                                  "StaticShapeReshape (", get_friendly_name(), ") out of range input shape dimension value: ", inputShape[i]);
            outputDimensionsValues[i] = static_cast<std::int64_t>(inputShape[i]);
        }
    }

    NODE_VALIDATION_CHECK(this, std::none_of(outputDimensionsValues.cbegin(), outputDimensionsValues.cend(),
        [](std::int64_t dimension) { return dimension < -1; }), "Dim size cannot be less than -1, got ", ngraph::PartialShape(outputDimensionsValues));
    const auto negativeDimsCount = std::count_if(outputDimensionsValues.cbegin(), outputDimensionsValues.cend(),
        [](std::int64_t dimension) { return dimension == -1; });
    NODE_VALIDATION_CHECK(this, negativeDimsCount <= 1, "More than one dimension has size of -1 (", negativeDimsCount, ")");

    const auto& inputShapeVolume = shape_size(inputShape);
    if (negativeDimsCount == 1) {
        const auto& outputShapeVolume = std::abs(std::accumulate(
            outputDimensionsValues.cbegin(),
            outputDimensionsValues.cend(),
            static_cast<std::int64_t>(1),
            std::multiplies<std::int64_t>())); //shape_size(outputDimensionsValues);
        NODE_VALIDATION_CHECK(this, inputShapeVolume % outputShapeVolume == 0, "StaticShapeReshape (", get_friendly_name(), ") ",
                              "output shape volume does not evenly divide the input shape volume: input shape volume = ", inputShapeVolume, " output shape ",
                              "volume = ", outputShapeVolume);
        NODE_VALIDATION_CHECK(this, outputShapeVolume != 0, "StaticShapeReshape (", get_friendly_name(), ") ",
                              "output shape volume is equal to 0");

        const auto actualValue = inputShapeVolume / outputShapeVolume;
        NODE_VALIDATION_CHECK(this, actualValue <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
                              "StaticShapeReshape (", get_friendly_name(), ") out of range output shape dimension value: ", actualValue);
        std::replace(outputDimensionsValues.begin(), outputDimensionsValues.end(),
            static_cast<std::int64_t>(-1), static_cast<std::int64_t>(actualValue));
    }

    const auto& outputShape = ngraph::PartialShape(outputDimensionsValues);
    NODE_VALIDATION_CHECK(this, inputShapeVolume == shape_size(outputDimensionsValues), "Requested output shape (upper-bound) ", outputShape,
                          " is incompatible with input shape ", get_input_shape(0), " (upper-bound)");

    set_output_type(0, get_input_element_type(0), outputShape);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
