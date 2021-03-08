// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/split.hpp"
#include "ngraph/node.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
SplitTransformation::SplitTransformation(const Params& params) : LayerTransformation(params) {}

void SplitTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Split>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

bool SplitTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> split = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    auto dequantization = NetworkHelper::getDequantization(split);

    OutputVector inputs(split->get_input_size());
    for (size_t i = 0; i < split->get_input_size(); ++i) {
        inputs[i] = split->get_input_node_shared_ptr(i);
    }

    const size_t dequantizationIndex = NetworkHelper::getChildInputIndex(dequantization.multiply, split);
    inputs[dequantizationIndex] = dequantization.data;

    std::shared_ptr<ngraph::Node> newSplit = split->clone_with_new_inputs(inputs);
    newSplit->set_friendly_name(split->get_friendly_name());

    const ngraph::Shape subConstShape = dequantization.subtract ?
        dequantization.subtract->get_input_node_shared_ptr(1)->get_shape() : Shape{};
    std::vector<float> subValues = dequantization.subtract ? as_type_ptr<opset1::Constant>(
        dequantization.subtract->get_input_node_shared_ptr(1))->cast_vector<float>() : std::vector<float>();

    const ngraph::Shape mulConstShape = dequantization.multiply->get_input_node_shared_ptr(1)->get_shape();
    std::vector<float> mulValues = as_type_ptr<opset1::Constant>(
        dequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<float>();

    int64_t SplitedAxis = as_type_ptr<opset1::Constant>(split->get_input_node_shared_ptr(1))->cast_vector<int64_t>()[0];
    size_t axis = SplitedAxis > 0 ? SplitedAxis : split->get_input_shape(0).size() + SplitedAxis;
    size_t outputSize = newSplit->get_output_size();

    const auto subSplitLengths = getConstSplitLengths(inputs, subConstShape, outputSize);
    const auto mulSplitLengths = getConstSplitLengths(inputs, mulConstShape, outputSize);

    std::vector<std::shared_ptr<ngraph::Node>> lastNodes(outputSize);
    ngraph::OutputVector replacement;
    for (size_t i = 0; i < outputSize; ++i) {
        Output<Node> previous = newSplit->output(i);

        if (dequantization.convert != nullptr) {
            const std::shared_ptr<ngraph::Node> convert =
                dequantization.convert->clone_with_new_inputs({ newSplit->output(i) });
            previous = convert;
        }

        if (dequantization.subtract != nullptr) {
            std::shared_ptr<ngraph::opset1::Constant> subConst;
            if (!subSplitLengths.empty()) {
                const auto newSubConstShape = getConstSplitShape(subSplitLengths, subConstShape, axis, i);

                std::vector<float> newSubValues(
                    subValues.begin() + subSplitLengths[i],
                    subValues.begin() + subSplitLengths[i + 1]);

                subConst = as_type_ptr<ngraph::opset1::Constant>(std::make_shared<ngraph::opset1::Constant>(
                    dequantization.subtract->get_input_element_type(1),
                    newSubConstShape,
                    newSubValues));
            } else {
                subConst = as_type_ptr<ngraph::opset1::Constant>(dequantization.subtract->get_input_node_shared_ptr(1)->clone_with_new_inputs({}));
            }
            const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(previous, subConst);
            previous = subtract;
        }

        std::shared_ptr<ngraph::opset1::Constant> mulConst;
        if (!mulSplitLengths.empty()) {
            const auto newMulConstShape = getConstSplitShape(mulSplitLengths, mulConstShape, axis, i);

            std::vector<float> newMulValues(
                mulValues.begin() + mulSplitLengths[i],
                mulValues.begin() + mulSplitLengths[i + 1]);

            mulConst = as_type_ptr<ngraph::opset1::Constant>(std::make_shared<ngraph::opset1::Constant>(
                dequantization.multiply->get_input_element_type(1), newMulConstShape, newMulValues));
        } else {
            mulConst = as_type_ptr<ngraph::opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}));
        }
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(previous, mulConst);

        lastNodes.push_back(multiply);
        replacement.push_back(multiply);
    }

    replace_node(split, replacement);
    updateOutputs(context, lastNodes, newSplit);
    return true;
}

std::vector<size_t> SplitTransformation::getConstSplitLengths(
    const OutputVector& inputs,
    const ngraph::Shape& constShape,
    const size_t outputSize) const {
    int64_t axis = as_type_ptr<opset1::Constant>(inputs[1].get_node_shared_ptr())->cast_vector<int64_t>()[0];
    size_t splitedAxis = axis > 0 ? axis : inputs[0].get_shape().size() + axis;

    if ((!constShape.empty()) && (constShape[splitedAxis] != 1)) {
        std::vector<size_t> result(outputSize + 1);
        result[0] = 0;
        for (size_t i = 1; i < result.size(); ++i) {
            result[i] = result[i - 1] + constShape[splitedAxis] / outputSize;
        }
        return result;
    } else {
        return std::vector<size_t>();
    }
}

ngraph::Shape SplitTransformation::getConstSplitShape(
    const std::vector<size_t>& constSplitLengths,
    const ngraph::Shape& constShape, const size_t axis,
    const size_t idx) const {
    Shape result(constShape);
    result[axis] = constSplitLengths[idx + 1] - constSplitLengths[idx];
    return result;
}

void SplitTransformation::updateOutputs(
    TransformationContext& context,
    std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
    std::shared_ptr<ngraph::Node> originalNode) const {
    const size_t outputSize = context.function->get_output_size();
    if (outputSize == 1) {
        updateOutput(context, lastNodes[0], originalNode);
    } else {
        const std::string originalName = originalNode->get_friendly_name();
        for (auto& lastNode : lastNodes) {
            for (size_t i = 0; i < outputSize; ++i) {
                std::shared_ptr<ngraph::Node> result = context.function->get_output_op(i);
                std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
                if (outputNode.get() == lastNode.get()) {
                    std::ostringstream oss;
                    oss << i;
                    originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
                    lastNode->set_friendly_name(originalName + "." + oss.str());
                    break;
                }
            }
        }
    }
}

bool SplitTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool SplitTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return (!NetworkHelper::getDequantization(layer).empty()) && LayerTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
