// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convolution_backprop_data.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConvolutionBackpropDataTransformation::ConvolutionBackpropDataTransformation(const Params& params) : WeightableLayerTransformation(params) {
}

void ConvolutionBackpropDataTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::ConvolutionBackpropData>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Multiply>() }));
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::ConvolutionBackpropData>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>() }));
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::ConvolutionBackpropData>(
                    { make_op_label<opset1::Multiply>(), make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::ConvolutionBackpropData>(
                    { make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>(), make_op_label<opset1::Constant>() }));
}

bool ConvolutionBackpropDataTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    if (deconvolutionSpecificChannelsRatio) {
        size_t inputChannels = layer->get_input_shape(0)[1];
        size_t outputChannels = layer->get_output_shape(0)[1];
        if (inputChannels % 4 != 0 || outputChannels % 16 != 0) {
            return false;
        }
    }
    return WeightableLayerTransformation::isQuantized(layer, false);
}

bool ConvolutionBackpropDataTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolutionBackpropData = m.get_match_root();

    if (!canBeTransformed(context, convolutionBackpropData)) {
        auto weightsInput = convolutionBackpropData->get_input_node_shared_ptr(1);
        std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(weightsInput);
        FakeQuantizeDequantization dequantization = reshapeFromWeights == nullptr ?
                         NetworkHelper::getDequantization(convolutionBackpropData, 1ul) :
                         NetworkHelper::getDequantization(reshapeFromWeights);
        if (dequantization.empty()) {
            const auto fqOnWeights = getFakeQuantizeOnWeights(convolutionBackpropData);
            std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(fqOnWeights);
            if (reshapeFromWeights != nullptr) {
                resultConstant = fold_reshape<opset1::Reshape>(
                        resultConstant,
                        reshapeFromWeights->input_value(1),
                        false);
            }
            if (as_type_ptr<opset1::Constant>(resultConstant)) {
                replace_node(weightsInput, resultConstant);
            }
        } else {
            NetworkHelper::foldDequantization(dequantization.multiply, 0, true);
        }
        return true;
    }

    convolutionBackpropData = NetworkHelper::separateInStandaloneBranch(convolutionBackpropData);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(convolutionBackpropData);
    {
        if (dequantization.subtract != nullptr) {
            std::shared_ptr<ngraph::Node> layer = dequantization.subtract;
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(layer);

            NetworkHelper::optimizeSubtract(dequantization.subtract);
        }
        std::shared_ptr<opset1::Constant> reducedConstant = as_type_ptr<opset1::Constant>(dequantization.multiplyConstant);
        std::shared_ptr<Node> newMultiplyAfterConst = std::make_shared<opset1::Constant>(
                reducedConstant->get_output_element_type(0),
                Shape{ 1 },
                reducedConstant->cast_vector<float>()[0]);
        auto inputs = convolutionBackpropData->input_values();
        inputs[0] = dequantization.multiply->input_value(0);
        const auto copyNode = convolutionBackpropData->copy_with_new_inputs(inputs);

        const auto relaxedConvolutionBackpropData = std::make_shared<op::TypeRelaxed<opset1::ConvolutionBackpropData>>(
            *as_type_ptr<opset1::ConvolutionBackpropData>(copyNode),
            std::vector<element::Type>{deqPrecision, deqPrecision},
            std::vector<element::Type>{deqPrecision});

        const auto newMultiplyAfter = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            std::vector<element::Type>{ deqPrecision, deqPrecision },
            std::vector<element::Type>{ dequantization.multiply->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(relaxedConvolutionBackpropData, deqPrecision).get(),
            ngraph::op::TemporaryReplaceOutputType(newMultiplyAfterConst, deqPrecision).get());

        replace_node(convolutionBackpropData, newMultiplyAfter);
        convolutionBackpropData = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        inputs[0] = convolutionBackpropData->get_input_node_ptr(0)->input_value(0);
        if (is_type<opset1::Convert>(convolutionBackpropData->get_input_node_ptr(0))) {
            auto newConvolution = convolutionBackpropData->copy_with_new_inputs(inputs);
            replace_node(convolutionBackpropData, newConvolution);
            convolutionBackpropData = newConvolution;
        }
    }

    {
        decomposeFakeQuantizeForWeightsPath(convolutionBackpropData, 1ul);

        dequantization = NetworkHelper::getDequantization(convolutionBackpropData, 1ul);

        if (is_type<opset1::FakeQuantize>(dequantization.data.get_node())) {
            const std::shared_ptr<opset1::FakeQuantize> fq = as_type_ptr<opset1::FakeQuantize>(dequantization.data.get_node_shared_ptr());
            std::shared_ptr<ngraph::Node> newFQ = NetworkHelper::fold_fake_quantize(fq, true);
            NetworkHelper::copyInfo(fq, newFQ);
            replace_node(fq, newFQ);
        }

        std::shared_ptr<opset1::Multiply> multiplyFromWeights = as_type_ptr<opset1::Multiply>(
                convolutionBackpropData->input_value(1).get_node_shared_ptr());
        std::shared_ptr<opset1::Subtract> subtractFromWeights = as_type_ptr<opset1::Subtract>(multiplyFromWeights->get_input_node_shared_ptr(0));

        {
            Shape newScaleShape = multiplyFromWeights->get_input_shape(1);
            auto inputs = convolutionBackpropData->input_values();
            inputs[1] = multiplyFromWeights->input_value(0);
            auto newMultiplyAfter = std::make_shared<DequantizationMultiply>(
                convolutionBackpropData->copy_with_new_inputs(inputs),
                foldConvert(
                    fold_reshape<opset1::Reshape>(
                        multiplyFromWeights->input_value(1),
                        std::make_shared<opset1::Constant>(element::u64, Shape{ newScaleShape.size() }, newScaleShape),
                        false),
                    convolutionBackpropData->get_output_element_type(0)));
            replace_node(convolutionBackpropData, newMultiplyAfter);
            convolutionBackpropData = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        }

        if (subtractFromWeights != nullptr) {
            // optimize zero point on weights
            auto optimizedSubtract = NetworkHelper::optimizeSubtract(subtractFromWeights);
            if (optimizedSubtract == nullptr) {
                subtractFromWeights = nullptr;
            } else {
                subtractFromWeights = as_type_ptr<opset1::Subtract>(optimizedSubtract);

                const Shape weightsShape = subtractFromWeights->input(0).get_shape();
                Shape zeroPointShape(weightsShape.size(), 1ul);
                zeroPointShape[1] = weightsShape[1];

                auto zeroPointConstant = fold<opset1::Broadcast>(
                        subtractFromWeights->get_input_node_shared_ptr(1),
                        std::make_shared<opset1::Constant>(element::i32, Shape{zeroPointShape.size()}, zeroPointShape));
                replace_node(subtractFromWeights->get_input_node_shared_ptr(1), zeroPointConstant);
            }
        }

        std::shared_ptr<opset1::Convert> convertFromWeights =
                as_type_ptr<opset1::Convert>(
                    subtractFromWeights == nullptr ?
                        multiplyFromWeights->get_input_node_shared_ptr(0) :
                        subtractFromWeights->get_input_node_shared_ptr(0));
        if (convertFromWeights != nullptr) {
            auto inputs = convolutionBackpropData->input_values();
            inputs[1] = convolutionBackpropData->get_input_node_ptr(1)->input_value(0);
            // remove Convert on weights
            auto newConvolution = convolutionBackpropData->clone_with_new_inputs(inputs);
            replace_node(convolutionBackpropData, newConvolution);
            convolutionBackpropData = newConvolution;
        }
    }
    std::shared_ptr<ngraph::opset1::Multiply> finalDequantization = NetworkHelper::optimizeMultipliesAfter(
            convolutionBackpropData->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    ngraph::copy_runtime_info({ convolutionBackpropData, finalDequantization }, finalDequantization);
    updateOutput(context, finalDequantization, convolutionBackpropData);

    auto onWeights = convolutionBackpropData->get_input_node_shared_ptr(1);
    if (is_type<opset1::Reshape>(onWeights)) {
        onWeights = onWeights->get_input_node_shared_ptr(0);
    }

    if (is_type<opset1::Subtract>(onWeights)) {
        auto& rt = onWeights->get_rt_info();
        rt["DISABLED_CONSTANT_FOLDING"] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
    }

    return true;
}

bool ConvolutionBackpropDataTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (deconvolutionSpecificChannelsRatio) {
        size_t inputChannels = op->get_input_shape(0)[1];
        size_t outputChannels = op->get_output_shape(0)[1];
        if (inputChannels % 4 != 0 || outputChannels % 16 != 0) {
            return false;
        }
    }

    return canConvolutionBeTransformed(context, op);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
