// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convolution.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include <transformations/rt_info/disable_constant_folding.hpp>
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConvolutionTransformation::ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {
    MATCHER_SCOPE(ConvolutionTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>({
        ngraph::pattern::wrap_type<opset1::Multiply>(),
        std::make_shared<pattern::op::Or>(OutputVector {
            pattern::wrap_type<opset1::Multiply>(),
            pattern::wrap_type<opset1::FakeQuantize>()
        })
    });


    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(ConvolutionTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ConvolutionTransformation::isQuantized(const std::shared_ptr<const Node>& layer,
    const std::vector<ngraph::element::Type>& defaultPrecisions) const {
    return ConvolutionTransformation::isQuantizedStatic(layer, defaultPrecisions);
}

bool ConvolutionTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer,
    const std::vector<ngraph::element::Type>& defaultPrecisions) {
    return WeightableLayerTransformation::isQuantizedStatic(layer, false, defaultPrecisions);
}

size_t ConvolutionTransformation::getInputChannels(const std::shared_ptr<ngraph::Node> conv) const {
    const auto channels = conv->get_input_partial_shape(1)[1];
    assert(channels.is_static());
    return channels.get_length();
}

bool ConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) {
    auto convolution = m.get_match_root();

    if (!canConvolutionBeTransformed(context, convolution, defaultPrecisions)) {
        const auto weightInput = convolution->get_input_node_shared_ptr(1);
        const auto reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(weightInput);
        FakeQuantizeDequantization dequantization = reshapeFromWeights == nullptr ?
                                                    NetworkHelper::getDequantization(convolution, defaultPrecisions, 1ul) :
                                                    NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions);
        if (dequantization.empty()) {
            const auto fqOnWeights = getFakeQuantizeOnWeights(convolution);
            std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(fqOnWeights);
            if (reshapeFromWeights != nullptr) {
                resultConstant = fold_reshape<opset1::Reshape>(
                        resultConstant,
                        reshapeFromWeights->input_value(1),
                        false);
            }
            if (ov::is_type<opset1::Constant>(resultConstant)) {
                replace_node(weightInput, resultConstant);
            }
        } else {
            NetworkHelper::foldDequantization(dequantization.multiply, 0, defaultPrecisions, true);
        }
        return true;
    }

    convolution = NetworkHelper::separateInStandaloneBranch(convolution, defaultPrecisions);

    const bool fqOnWeightsWasDecomposed = decomposeFakeQuantizeForWeightsPath(convolution);
    if (updatePrecisions && !fqOnWeightsWasDecomposed) {
        return false;
    }

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(convolution, defaultPrecisions);

    std::shared_ptr<Node> newMultiplyAfter;
    {
        std::shared_ptr<opset1::Subtract> subtract;
        if (dequantization.subtract != nullptr) {
            auto optimizedSubtract = NetworkHelper::optimizeSubtract(dequantization.subtract);
            if (optimizedSubtract == nullptr) {
                optimizedSubtract = dequantization.subtract;
            }
            subtract = ov::as_type_ptr<opset1::Subtract>(optimizedSubtract);
        }

        // workaround normalizes shape of Subtract to match CPU plugin expectations
        if (subtract && subtract->get_output_partial_shape(0) != subtract->get_input_partial_shape(1)) {
            size_t length = subtract->get_output_partial_shape(0).rank().get_length();

            // Insert explicit broadcast for channel dimension [1] and immediately fold it
            Shape broadcastShape(length, 1);
            broadcastShape[1] = getInputChannels(convolution);

            std::shared_ptr<Node> newShift = fold<opset1::Broadcast>(
                subtract->input_value(1),
                std::make_shared<opset1::Constant>(
                    element::i64,
                    Shape{ length },
                    broadcastShape));

            const auto newSubtract = ov::as_type_ptr<opset1::Subtract>(subtract->clone_with_new_inputs({
                subtract->input_value(0),
                newShift }));
            NetworkHelper::copyInfo(subtract, newSubtract);
            replace_node(subtract, newSubtract);

            newSubtract->set_output_type(0, subtract->get_output_element_type(0), newSubtract->get_output_partial_shape(0));
            subtract = newSubtract;
        }

        const size_t groupsCount = NetworkHelper::getGroupsCount(convolution);
        std::shared_ptr<Node> newMultiplyAfterConst;
        if (groupsCount > 1ul) {
            const std::vector<float> scales = dequantization.multiplyConstant->cast_vector<float>();
            if (scales.size() == 1ul) {
                newMultiplyAfterConst = dequantization.multiplyConstant->clone_with_new_inputs({});
            } else {
                const ngraph::PartialShape inputPShape = convolution->get_input_partial_shape(0);
                const size_t inputChannelsInGroup = inputPShape[1].get_length() / groupsCount;
                const ngraph::PartialShape outputPShape = convolution->get_output_partial_shape(0);
                std::vector<float> outputScales(outputPShape[1].get_length());

                const size_t outputChannelsInGroup = outputPShape[1].get_length() / groupsCount;
                for (size_t group = 0; group < groupsCount; ++group) {
                    const float scaleValue = scales[group * inputChannelsInGroup];

                    for (size_t i = 0; i < outputChannelsInGroup; ++i) {
                        size_t index = group * outputChannelsInGroup + i;
                        outputScales[index] = scaleValue;
                    }
                }

                const size_t outRank = outputPShape.rank().get_length();
                auto newMulShape = Shape{ outputScales.size() };
                for (size_t i = 0; i < outRank - 2; ++i) {
                    newMulShape.push_back(1ul);
                }

                newMultiplyAfterConst = std::make_shared<opset1::Constant>(
                    dequantization.multiplyConstant->get_element_type(),
                    newMulShape,
                    outputScales);
            }
        } else {
            newMultiplyAfterConst = std::make_shared<opset1::Constant>(
                dequantization.multiplyConstant->get_element_type(),
                Shape{ 1 },
                dequantization.multiplyConstant->cast_vector<float>()[0]);
        }

        const auto copyNode = convolution->clone_with_new_inputs({ dequantization.multiply->input_value(0), convolution->input_value(1) });
        auto conv = ov::as_type_ptr<opset1::Convolution>(copyNode);
        std::shared_ptr<Node> relaxedNewConvolution;
        if (conv) {
            relaxedNewConvolution = std::make_shared<op::TypeRelaxed<opset1::Convolution>>(
                    *conv,
                    std::vector<element::Type>{deqPrecision, deqPrecision},
                    std::vector<element::Type>{deqPrecision});
        } else {
            relaxedNewConvolution = std::make_shared<op::TypeRelaxed<opset1::GroupConvolution>>(
                    *ov::as_type_ptr<opset1::GroupConvolution>(copyNode),
                    std::vector<element::Type>{deqPrecision, deqPrecision},
                    std::vector<element::Type>{deqPrecision});
        }
        NetworkHelper::copyInfo(convolution, relaxedNewConvolution);

        newMultiplyAfter = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
            std::vector<element::Type>{ deqPrecision, deqPrecision },
            std::vector<element::Type>{ dequantization.multiply->get_output_element_type(0) },
            ngraph::op::TemporaryReplaceOutputType(relaxedNewConvolution, deqPrecision).get(),
            ngraph::op::TemporaryReplaceOutputType(newMultiplyAfterConst, deqPrecision).get());

        NetworkHelper::insertDequantizationAfter(convolution, newMultiplyAfter, relaxedNewConvolution);
        convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();

        if (ov::is_type<opset1::Convert>(convolution->get_input_node_ptr(0))) {
            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->get_input_node_ptr(0)->input_value(0),
                convolution->input_value(1)});
            replace_node(convolution, newConvolution);
            NetworkHelper::copyInfo(convolution, newConvolution);
            convolution = newConvolution;
        }
    }

    {
        if (!updatePrecisions && !fqOnWeightsWasDecomposed) {
            // TODO: LPT: issue #58685
            return false;
        }

        std::shared_ptr<opset1::Reshape> reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(convolution->get_input_node_shared_ptr(1));

        dequantization = reshapeFromWeights == nullptr ?
            NetworkHelper::getDequantization(convolution, defaultPrecisions, 1ul) :
            NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions);
        assert(!dequantization.empty());
        if (ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node())) {
            const std::shared_ptr<opset1::FakeQuantize> fq = ov::as_type_ptr<opset1::FakeQuantize>(dequantization.data.get_node_shared_ptr());
            std::shared_ptr<ngraph::Node> newFQ = NetworkHelper::fold_fake_quantize(fq, true);
            NetworkHelper::copyInfo(fq, newFQ);
            replace_node(fq, newFQ);
        }

        std::shared_ptr<opset1::Multiply> multiplyFromWeights = ov::as_type_ptr<opset1::Multiply>(
            reshapeFromWeights == nullptr ?
            convolution->get_input_node_shared_ptr(1) :
            convolution->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
        std::shared_ptr<opset1::Subtract> subtractFromWeights = ov::as_type_ptr<opset1::Subtract>(multiplyFromWeights->get_input_node_shared_ptr(0));

        {
            const auto newScalePShape = multiplyFromWeights->get_input_partial_shape(1);
            assert(newScalePShape.is_static());
            Shape newScaleShape = newScalePShape.to_shape();

            if (!newScaleShape.empty()) {
                // that's all we need: [C, 1, 1, 1] => [C, 1, 1]
                newScaleShape.pop_back();
            }

            if (reshapeFromWeights != nullptr) {
                reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(reshapeFromWeights->clone_with_new_inputs({
                    multiplyFromWeights->input_value(0),
                    reshapeFromWeights->input_value(1) }));
            }

            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->input_value(0),
                reshapeFromWeights != nullptr ?
                    reshapeFromWeights :
                    multiplyFromWeights->input_value(0)
            });
            NetworkHelper::copyInfo(convolution, newConvolution);

            newMultiplyAfter = std::make_shared<opset1::Multiply>(
                newConvolution,
                foldConvert(
                    fold_reshape<opset1::Reshape>(
                        multiplyFromWeights->input_value(1),
                        std::make_shared<opset1::Constant>(element::u64, Shape{ newScaleShape.size() }, newScaleShape),
                        false),
                    convolution->get_output_element_type(0)));
            NetworkHelper::insertDequantizationAfter(convolution, newMultiplyAfter, newConvolution);
            convolution = newMultiplyAfter->input_value(0).get_node_shared_ptr();
        }

        if (subtractFromWeights != nullptr) {
            // optimize zero point on weights
            auto optimizedSubtract = NetworkHelper::optimizeSubtract(subtractFromWeights);

            // TODO: handle optimizedSubtract == nullptr;
            if (optimizedSubtract == nullptr) {
                subtractFromWeights = nullptr;
            } else {
                subtractFromWeights = ov::as_type_ptr<opset1::Subtract>(optimizedSubtract);

                const auto weightsPShape = subtractFromWeights->get_input_partial_shape(0);
                assert(weightsPShape.is_static());

                const size_t weightsRankValue = weightsPShape.rank().get_length();
                Shape zeroPointShape(weightsRankValue, 1ul);
                zeroPointShape[0] = static_cast<size_t>(weightsPShape[0].get_length());

                auto zeroPointConstant = fold<opset1::Broadcast>(
                    subtractFromWeights->input_value(1),
                    std::make_shared<opset1::Constant>(element::i32, Shape{ zeroPointShape.size() }, zeroPointShape));
                NetworkHelper::copyInfo(subtractFromWeights->get_input_node_shared_ptr(1), zeroPointConstant);
                replace_node(subtractFromWeights->get_input_node_shared_ptr(1), zeroPointConstant);
            }
        }

        std::shared_ptr<opset1::Convert> convertFromWeights = ov::as_type_ptr<opset1::Convert>(subtractFromWeights == nullptr ?
            multiplyFromWeights->get_input_node_shared_ptr(0) :
            subtractFromWeights->get_input_node_shared_ptr(0));
        if (convertFromWeights != nullptr) {
            // remove Convert on weights
            std::shared_ptr<Node> childNode = reshapeFromWeights == nullptr ? convolution : reshapeFromWeights;

            auto newConvolution = convolution->clone_with_new_inputs({
                convolution->input_value(0),
                childNode.get() == convolution.get() ?
                    convolution->get_input_node_ptr(1)->input_value(0) :
                    childNode->clone_with_new_inputs({convertFromWeights->input_value(0), childNode->input_value(1)})});
            replace_node(convolution, newConvolution);
            NetworkHelper::copyInfo(convolution, newConvolution);
            convolution = newConvolution;
        }

        reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(convolution->get_input_node_shared_ptr(1));
        if (reshapeFromWeights != nullptr) {
            // remove Reshape on weights
            const std::shared_ptr<Node> newWeights = fold_reshape<opset1::Reshape>(
                reshapeFromWeights->input_value(0),
                reshapeFromWeights->input_value(1),
                false);

            replace_node(reshapeFromWeights, newWeights);
        }
    }

    const auto finalDequantization = NetworkHelper::optimizeMultipliesAfter(newMultiplyAfter);
    ngraph::copy_runtime_info({ convolution, finalDequantization }, finalDequantization);
    updateOutput(context, finalDequantization, convolution);

    // [C, 1, 1] -> [1, C, 1, 1]
    NetworkHelper::normalizeDequantizationShape(finalDequantization);

    auto onWeights = convolution->get_input_node_shared_ptr(1);
    if (ov::is_type<opset1::Reshape>(onWeights)) {
        onWeights = onWeights->get_input_node_shared_ptr(0);
    }

    if (ov::is_type<opset1::Subtract>(onWeights)) {
        ov::disable_constant_folding(onWeights);
    }
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
