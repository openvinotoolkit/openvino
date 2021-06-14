// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/multiply_to_group_convolution.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void MultiplyToGroupConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

bool MultiplyToGroupConvolutionTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    auto input = multiply->get_input_node_shared_ptr(0);
    auto constant = multiply->get_input_node_shared_ptr(1);
    auto inputIndex = 0;
    if (!is_type<opset1::Constant>(constant)) {
        input = multiply->get_input_node_shared_ptr(1);
        constant = multiply->get_input_node_shared_ptr(0);
        inputIndex = 1;
    }

    auto dequantization = NetworkHelper::getDequantization(multiply, inputIndex);
    if (dequantization.subtractConvert != nullptr) {
        dequantization = NetworkHelper::foldDequantization(multiply, inputIndex);
    }

    const element::Type weightsPrecision = updatePrecisions ? precisionsOnWeights[0] : dequantization.data.get_element_type();

    const size_t inputChannelsCount = input->get_output_shape(0)[1];
    const size_t outputChannelsCount = multiply->get_output_shape(0)[1];
    const size_t group = outputChannelsCount / groupSize;
    const size_t weightsSize = outputChannelsCount * inputChannelsCount / group;
    std::vector<float> weightsBuffer(weightsSize);
    const size_t kernelsCount = inputChannelsCount / group;

    if (group == 1ul) {
        for (size_t outputChannel = 0ul; outputChannel < outputChannelsCount; ++outputChannel) {
            for (size_t kernel = 0ul; kernel < kernelsCount; ++kernel) {
                const float value = (outputChannel == kernel) ? 1.f : 0.f;
                weightsBuffer[kernelsCount * outputChannel + kernel] = value;
            }
        }
    } else {
        const size_t channelsInGroup = outputChannelsCount / group;
        for (size_t outputChannel = 0ul; outputChannel < outputChannelsCount; ++outputChannel) {
            const size_t groupIndex = outputChannel / channelsInGroup;
            for (size_t kernel = 0ul; kernel < kernelsCount; ++kernel) {
                const size_t outputChannelIndexInGroup = outputChannel - groupIndex * channelsInGroup;
                const float value = (outputChannelIndexInGroup == kernel) ? 1.f : 0.f;
                weightsBuffer[kernelsCount * outputChannel + kernel] = value;
            }
        }
    }

    const Shape shape = multiply->output(0).get_shape();

    Shape weightsShape = Shape(shape.size() + 1, 1ul);
    weightsShape[0] = group;
    weightsShape[1] = outputChannelsCount / group;
    weightsShape[2] = inputChannelsCount / group;
    const auto weightsNode = std::make_shared<opset1::Constant>(weightsPrecision, weightsShape, weightsBuffer);

    const size_t spatialDimsSize = shape.size() - 2;
    ngraph::Strides strides(spatialDimsSize, 1ul);
    ngraph::CoordinateDiff pads(spatialDimsSize, 0ul);
    ngraph::Strides dilations(spatialDimsSize, 1ul);

    const auto convolution = std::make_shared<op::TypeRelaxed<opset1::GroupConvolution>>(
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{ element::f32 },
        ngraph::op::TemporaryReplaceOutputType(dequantization.data, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(weightsNode, element::f32).get(),
        strides,
        pads,
        pads,
        dilations);
    convolution->set_friendly_name(multiply->get_friendly_name() + "/GroupConvolution");

    std::shared_ptr<Node> lastNode = convolution;
    if (dequantization.subtract != nullptr) {
        lastNode = std::make_shared<opset1::Add>(
            convolution,
            fold<opset1::Negative>(foldConvert(dequantization.subtractConstant, element::f32)));
        lastNode->set_friendly_name(convolution->get_friendly_name() + "/Add");
    }

    lastNode = multiply->copy_with_new_inputs({ lastNode, constant });

    replace_node(multiply, lastNode);
    NetworkHelper::copyInfo(multiply, lastNode);

    return true;
}

bool MultiplyToGroupConvolutionTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    const Shape shape = operation->output(0).get_shape();
    if ((shape.size() != 4ul) && (shape.size() != 5ul)) {
        return false;
    }

    Shape constShape;
    int inputIndex;
    if (is_type<opset1::Constant>(operation->get_input_node_shared_ptr(1))) {
        inputIndex = 0;
        constShape = operation->get_input_shape(1);
        if (is_type<opset1::Constant>(operation->get_input_node_shared_ptr(0)) ||
            (is_type<opset1::Subtract>(operation->get_input_node_shared_ptr(0))  &&
            is_type<opset1::Constant>(operation->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0)))) {
            return false;
        }
    } else if (is_type<opset1::Constant>(operation->get_input_node_shared_ptr(0))) {
        inputIndex = 1;
        constShape = operation->get_input_shape(0);
    } else {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation, inputIndex);

    if (dequantization.empty()) {
        return false;
    }

    const Shape outShape = operation->get_output_shape(0);
    if (outShape[1] % groupSize != 0) {
        return false;
    }

    for (size_t i = 2; i < constShape.size(); ++i) {
        if (constShape[i] != 1) {
            return false;
        }
    }

    if (updatePrecisions) {
        const element::Type parentPrecision = dequantization.data.get_element_type();
        if (std::find(precisionsOnActivations.begin(), precisionsOnActivations.end(), parentPrecision) == precisionsOnActivations.end()) {
            return false;
        }
    }

    return true;
}

bool MultiplyToGroupConvolutionTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    const auto parent0 = layer->get_input_node_shared_ptr(0);
    const auto parent1 = layer->get_input_node_shared_ptr(1);

    if (!is_type<opset1::Constant>(parent0) && !is_type<opset1::Constant>(parent1)) {
        return false;
    }

    const Shape shape = layer->output(0).get_shape();
    return (shape.size() == 4ul) || (shape.size() == 5ul);
}

void MultiplyToGroupConvolutionTransformation::setGroupSize(const size_t groupSize) {
    this->groupSize = groupSize;
}

size_t MultiplyToGroupConvolutionTransformation::getGroupSize() const {
    return groupSize;
}

bool MultiplyToGroupConvolutionTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
