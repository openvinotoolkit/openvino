// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convolution.hpp"

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

ConvolutionTransformation::ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {
}

void ConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Convolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Multiply>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Convolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>() }));
}

bool ConvolutionTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    return WeightableLayerTransformation::isQuantized(layer, false);
}

bool ConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();

    if (!WeightableLayerTransformation::canBeTransformed(context, convolution)) {
        return false;
    }

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(convolution);
    if (!canSubtractBeHandled(convolution, dequantization)) {
        return false;
    }

    if ((!supportAsymmetricQuantization) && getDataPrecisionOnWeights(convolution).hasZeroPoint) {
        return false;
    }

    if (updatePrecisions && !dequantization.empty() && !dequantization.isLowPrecision()) {
        return false;
    }

    convolution = NetworkHelper::separateInStandaloneBranch(convolution);
    dequantization = NetworkHelper::getDequantization(convolution);

    {
        std::shared_ptr<opset1::Subtract> subtract;
        if (dequantization.subtract != nullptr) {
            std::shared_ptr<ngraph::Node> layer = dequantization.subtract;
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(layer);

            auto optimizedSubtract = NetworkHelper::optimizeSubtract(dequantization.subtract);
            if (optimizedSubtract == nullptr) {
                optimizedSubtract = dequantization.subtract;
            }
            subtract = as_type_ptr<opset1::Subtract>(optimizedSubtract);
        }

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

        const size_t groupsCount = NetworkHelper::getGroupsCount(convolution);
        std::shared_ptr<Node> newMultiplyAfterConst;
        if (groupsCount > 1ul) {
            std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));

            const std::vector<float> scales = multiplyConst->cast_vector<float>();
            if (scales.size() == 1ul) {
                newMultiplyAfterConst = dequantization.multiply->input_value(1).get_node_shared_ptr()->clone_with_new_inputs({});
            } else {
                const ngraph::Shape inputShape = convolution->get_input_shape(0);
                const size_t inputChannelsInGroup = inputShape[1] / groupsCount;
                const ngraph::Shape outputShape = convolution->get_output_shape(0);
                std::vector<float> outputScales(outputShape[1]);

                const size_t outputChannelsInGroup = outputShape[1] / groupsCount;
                for (size_t group = 0; group < groupsCount; ++group) {
                    const float scaleValue = scales[group * inputChannelsInGroup];

                    for (size_t i = 0; i < outputChannelsInGroup; ++i) {
                        size_t index = group * outputChannelsInGroup + i;
                        outputScales[index] = scaleValue;
                    }
                }

                auto newMulShape = Shape{ outputScales.size() };
                for (size_t i = 0; i < convolution->get_output_shape(0).size() - 2; ++i) {
                    newMulShape.push_back(1ul);
                }

                newMultiplyAfterConst = std::make_shared<opset1::Constant>(
                    dequantization.multiply->get_input_element_type(1),
                    newMulShape,
                    outputScales);
            }
        } else {
            std::shared_ptr<opset1::Constant> reducedConstant = as_type_ptr<opset1::Constant>(
                dequantization.multiply->input_value(1).get_node_shared_ptr());
            newMultiplyAfterConst = std::make_shared<opset1::Constant>(
                reducedConstant->get_output_element_type(0),
                Shape{ 1 },
                reducedConstant->cast_vector<float>()[0]);
        }

        const auto copyNode = convolution->copy_with_new_inputs({ dequantization.multiply->input_value(0), convolution->input_value(1) });
        auto conv = as_type_ptr<opset1::Convolution>(copyNode);
        std::shared_ptr<Node> relaxedNewConvolution;
        if (conv) {
            relaxedNewConvolution = std::make_shared<op::TypeRelaxed<opset1::Convolution>>(
                    *conv,
                    std::vector<element::Type>{deqPrecision, deqPrecision},
                    std::vector<element::Type>{deqPrecision});
        } else {
            relaxedNewConvolution = std::make_shared<op::TypeRelaxed<opset1::GroupConvolution>>(
                    *as_type_ptr<opset1::GroupConvolution>(copyNode),
                    std::vector<element::Type>{deqPrecision, deqPrecision},
                    std::vector<element::Type>{deqPrecision});
        }

        std::shared_ptr<ngraph::opset1::Multiply> newMultiplyAfter = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            std::vector<element::Type>{ deqPrecision, deqPrecision },
            std::vector<element::Type>{ dequantization.multiply->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(relaxedNewConvolution, deqPrecision).get(),
            ngraph::op::TemporaryReplaceOutputType(newMultiplyAfterConst, deqPrecision).get());

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
        decomposeFakeQuantizeForWeightsPath(convolution);

        std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(convolution->input_value(1).get_node_shared_ptr());

        const auto dequantization = reshapeFromWeights == nullptr ?
            NetworkHelper::getDequantization(convolution, 1ul) :
            NetworkHelper::getDequantization(reshapeFromWeights);
        assert(!dequantization.empty());
        if (is_type<opset1::FakeQuantize>(dequantization.data.get_node())) {
            const std::shared_ptr<opset1::FakeQuantize> fq = as_type_ptr<opset1::FakeQuantize>(dequantization.data.get_node_shared_ptr());
            std::shared_ptr<ngraph::Node> newFQ = NetworkHelper::fold_fake_quantize(fq, true);
            NetworkHelper::copyInfo(fq, newFQ);
            replace_node(fq, newFQ);
        }

        std::shared_ptr<opset1::Multiply> multiplyFromWeights = as_type_ptr<opset1::Multiply>(
            reshapeFromWeights == nullptr ?
            convolution->input_value(1).get_node_shared_ptr() :
            convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Subtract> subtractFromWeights = as_type_ptr<opset1::Subtract>(multiplyFromWeights->get_input_node_shared_ptr(0));

        {
            Shape newScaleShape = multiplyFromWeights->get_input_shape(1);
            if (!newScaleShape.empty()) {
                // that's all we need: [C, 1, 1, 1] => [C, 1, 1]
                newScaleShape.pop_back();
            }

            if (reshapeFromWeights != nullptr) {
                reshapeFromWeights = as_type_ptr<opset1::Reshape>(reshapeFromWeights->copy_with_new_inputs({
                    multiplyFromWeights->input_value(0),
                    reshapeFromWeights->input_value(1) }));
            }

            auto newMultiplyAfter = std::make_shared<DequantizationMultiply>(
                convolution->copy_with_new_inputs({
                    convolution->input_value(0),
                    reshapeFromWeights != nullptr ?
                        reshapeFromWeights :
                        multiplyFromWeights->input_value(0)
                    }),
                fold<opset1::Convert>(
                    fold_reshape<opset1::Reshape>(
                        multiplyFromWeights->input_value(1),
                        std::make_shared<opset1::Constant>(element::u64, Shape{ newScaleShape.size() }, newScaleShape),
                        false),
                    convolution->get_output_element_type(0)));
            replace_node(convolution, newMultiplyAfter);
            convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        }

        if (subtractFromWeights != nullptr) {
            // optimize zero point on weights
            auto optimizedSubtract = NetworkHelper::optimizeSubtract(subtractFromWeights);

            // TODO: handle optimizedSubtract == nullptr;
            if (optimizedSubtract == nullptr) {
                subtractFromWeights = nullptr;
            } else {
                subtractFromWeights = as_type_ptr<opset1::Subtract>(optimizedSubtract);

                const Shape weightsShape = subtractFromWeights->input(0).get_shape();
                Shape zeroPointShape(weightsShape.size(), 1ul);
                zeroPointShape[0] = weightsShape[0];

                auto zeroPointConstant = fold<opset1::Broadcast>(
                    subtractFromWeights->get_input_node_shared_ptr(1),
                    std::make_shared<opset1::Constant>(element::i32, Shape{ zeroPointShape.size() }, zeroPointShape));
                replace_node(subtractFromWeights->get_input_node_shared_ptr(1), zeroPointConstant);
            }
        }

        std::shared_ptr<opset1::Convert> convertFromWeights = as_type_ptr<opset1::Convert>(subtractFromWeights == nullptr ?
            multiplyFromWeights->get_input_node_shared_ptr(0) :
            subtractFromWeights->get_input_node_shared_ptr(0));
        if (convertFromWeights != nullptr) {
            // remove Convert on weights
            std::shared_ptr<Node> childNode = reshapeFromWeights == nullptr ? convolution : reshapeFromWeights;

            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_shared_ptr(0),
                childNode.get() == convolution.get() ?
                    convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0) :
                    childNode->copy_with_new_inputs({convertFromWeights->input_value(0), childNode->input_value(1)})});
            replace_node(convolution, newConvolution);
            convolution = newConvolution;
        }

        reshapeFromWeights = as_type_ptr<opset1::Reshape>(convolution->get_input_node_shared_ptr(1));
        if (reshapeFromWeights != nullptr) {
            // remove Reshape on weights
            const std::shared_ptr<Node> newWeights = fold_reshape<opset1::Reshape>(
                reshapeFromWeights->input_value(0),
                reshapeFromWeights->input_value(1),
                false);

            replace_node(reshapeFromWeights, newWeights);
        }
    }

    std::shared_ptr<ngraph::opset1::Multiply> finalDequantization = NetworkHelper::optimizeMultipliesAfter(
        convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    ngraph::copy_runtime_info({ convolution, finalDequantization }, finalDequantization);
    updateOutput(context, finalDequantization, convolution);

    auto onWeights = convolution->get_input_node_shared_ptr(1);
    if (is_type<opset1::Reshape>(onWeights)) {
        onWeights = onWeights->get_input_node_shared_ptr(0);
    }

    if (is_type<opset1::Subtract>(onWeights)) {
        auto& rt = onWeights->get_rt_info();
        rt["DISABLED_CONSTANT_FOLDING"] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
    }
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
