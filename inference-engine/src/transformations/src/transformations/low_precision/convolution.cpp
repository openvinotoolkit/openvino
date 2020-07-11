// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/convolution.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConvolutionTransformation::ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {
}

void ConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Convolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>()}));
}

size_t handledCount = 0;

void ConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();

    if (!WeightableLayerTransformation::canBeTransformed(context, convolution)) {
        return;
    }

    convolution = separateInStandaloneBranch(convolution);

    {
        const FakeQuantizeDequantization dequantization = getDequantization(convolution);

        std::shared_ptr<opset1::Subtract> subtract = dequantization.subtract == nullptr ?
            nullptr :
            as_type_ptr<opset1::Subtract>(optimizeSubtract(dequantization.subtract));

        // workaround normalizes shape of Subtract to match CPU plugin expectations
        if (subtract && subtract->get_output_partial_shape(0) != subtract->get_input_partial_shape(1)) {
            size_t length = subtract->get_output_partial_shape(0).rank().get_length();

            // Insert explicit broadcast for channel dimension [1] and immediately fold it
            Shape broadcastShape(subtract->get_output_partial_shape(0).rank().get_length(), 1);
            broadcastShape[1] = subtract->get_output_shape(0)[1];

            std::shared_ptr<Node> newShift = fold<opset1::Broadcast>(
                subtract->input_value(1).get_node_shared_ptr(),
                std::make_shared<opset1::Constant>(
                    element::i64,
                    Shape{ length },
                    broadcastShape));

            const auto newSubtract = as_type_ptr<opset1::Subtract>(subtract->clone_with_new_inputs({
                subtract->input_value(0).get_node_shared_ptr(),
                newShift }));
            replace_node(subtract, newSubtract);

            newSubtract->set_output_type(0, subtract->get_output_element_type(0), newSubtract->get_output_partial_shape(0));
            subtract = newSubtract;
        }

        std::shared_ptr<ngraph::opset1::Multiply> newMultiplyAfter = std::make_shared<opset1::Multiply>(
            convolution->copy_with_new_inputs({ dequantization.multiply->input_value(0), convolution->input_value(1) }),
            // workaround: constant is cloning because it's used twice and can not be fused below
            dequantization.multiply->input_value(1).get_node_shared_ptr()->clone_with_new_inputs({}));

        replace_node(convolution, newMultiplyAfter);
        convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();

        if (is_type<opset1::Convert>(convolution->get_input_node_ptr(0))) {
            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_ptr(0)->get_input_node_shared_ptr(0),
                convolution->get_input_node_shared_ptr(1) });
            replace_node(convolution, newConvolution);
            convolution = newConvolution;
        }
    }

    {
        decomposeFakeQuantizeForWeightsPath(convolution, supportAsymmetricQuantization);

        std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(convolution->input_value(1).get_node_shared_ptr());
        std::shared_ptr<opset1::Multiply> multiplyFromWeights = as_type_ptr<opset1::Multiply>(
            reshapeFromWeights == nullptr ?
            convolution->input_value(1).get_node_shared_ptr() :
            convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Subtract> subtractFromWeights = as_type_ptr<opset1::Subtract>(multiplyFromWeights->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Convert> convertFromWeights = as_type_ptr<opset1::Convert>(subtractFromWeights == nullptr ?
            multiplyFromWeights->get_input_node_shared_ptr(0) :
            subtractFromWeights->get_input_node_shared_ptr(0));

        {
            Shape newScaleShape = multiplyFromWeights->get_input_shape(1);
            // that's all we need: [C, 1, 1, 1] => [C, 1, 1]
            newScaleShape.pop_back();

            if (reshapeFromWeights != nullptr) {
                reshapeFromWeights = as_type_ptr<opset1::Reshape>(reshapeFromWeights->copy_with_new_inputs({
                    multiplyFromWeights->input_value(0),
                    reshapeFromWeights->input_value(1) }));
            }

            auto newMultiplyAfter = std::make_shared<opset1::Multiply>(
                convolution->copy_with_new_inputs({
                    convolution->input_value(0),
                    reshapeFromWeights != nullptr ?
                        reshapeFromWeights :
                        multiplyFromWeights->input_value(0)
                    }),
                fold_reshape<opset1::Reshape>(
                    multiplyFromWeights->input_value(1),
                    std::make_shared<opset1::Constant>(element::u64, Shape{ newScaleShape.size() }, newScaleShape),
                    false));
            replace_node(convolution, newMultiplyAfter);
            convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        }

        if (subtractFromWeights != nullptr) {
            optimizeSubtract(subtractFromWeights);
        }

        if (convertFromWeights != nullptr) {
            std::shared_ptr<Node> childNode = reshapeFromWeights == nullptr ? convolution : reshapeFromWeights;

            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_shared_ptr(0),
                childNode.get() == convolution.get() ?
                    convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0) :
                    childNode->copy_with_new_inputs({convertFromWeights->input_value(0), childNode->input_value(1)})});
            replace_node(convolution, newConvolution);
            convolution = newConvolution;
        }
    }

    std::shared_ptr<ngraph::opset1::Multiply> finalDequantization = optimizeMultipliesAfter(
        convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this());

    updateOutput(context, finalDequantization, convolution);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
