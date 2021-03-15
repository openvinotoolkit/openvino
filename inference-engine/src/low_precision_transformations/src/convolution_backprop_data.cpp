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
    return WeightableLayerTransformation::isQuantized(layer, false);
}

bool ConvolutionBackpropDataTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolutionBackpropData = m.get_match_root();

    if (!canBeTransformed(context, convolutionBackpropData)) {
        return false;
    }

    convolutionBackpropData = NetworkHelper::separateInStandaloneBranch(convolutionBackpropData);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(convolutionBackpropData);
    const bool haveOutputShape = convolutionBackpropData->get_input_size() == 3;
    {
        std::shared_ptr<opset1::Subtract> subtract;
        if (dequantization.subtract != nullptr) {
            std::shared_ptr<ngraph::Node> layer = dequantization.subtract;
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(layer);

            auto optimizedSubtract = NetworkHelper::optimizeSubtract(dequantization.subtract);
            subtract = optimizedSubtract ? as_type_ptr<opset1::Subtract>(optimizedSubtract) : dequantization.subtract;
        }

        std::shared_ptr<opset1::Constant> reducedConstant = as_type_ptr<opset1::Constant>(
            dequantization.multiply->input_value(1).get_node_shared_ptr());
        std::shared_ptr<Node> newMultiplyAfterConst = std::make_shared<opset1::Constant>(
            reducedConstant->get_output_element_type(0),
            Shape{ 1 },
            reducedConstant->cast_vector<float>()[0]);

        const auto copyNode = haveOutputShape ?
            convolutionBackpropData->copy_with_new_inputs({
                dequantization.multiply->input_value(0),
                convolutionBackpropData->input_value(1),
                convolutionBackpropData->input_value(2) }) :
            convolutionBackpropData->copy_with_new_inputs({
                dequantization.multiply->input_value(0),
                convolutionBackpropData->input_value(1) });

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

        if (is_type<opset1::Convert>(convolutionBackpropData->get_input_node_ptr(0))) {
            auto newConvolution = haveOutputShape ?
                convolutionBackpropData->copy_with_new_inputs({
                    convolutionBackpropData->get_input_node_ptr(0)->get_input_node_shared_ptr(0),
                    convolutionBackpropData->input_value(1),
                    convolutionBackpropData->input_value(2) }) :
                convolutionBackpropData->copy_with_new_inputs({
                    convolutionBackpropData->get_input_node_ptr(0)->get_input_node_shared_ptr(0),
                    convolutionBackpropData->input_value(1) });
            replace_node(convolutionBackpropData, newConvolution);
            convolutionBackpropData = newConvolution;
        }
    }

    {
        decomposeFakeQuantizeForWeightsPath(convolutionBackpropData, 1);

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

            auto newMultiplyAfter = std::make_shared<DequantizationMultiply>(
                haveOutputShape ?
                    convolutionBackpropData->copy_with_new_inputs({
                          convolutionBackpropData->input_value(0),
                          multiplyFromWeights->input_value(0),
                          convolutionBackpropData->input_value(2)}) :
                    convolutionBackpropData->copy_with_new_inputs({
                          convolutionBackpropData->input_value(0),
                          multiplyFromWeights->input_value(0) }),
                fold<opset1::Convert>(
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
            // remove Convert on weights
            auto newConvolution =
                haveOutputShape ?
                    convolutionBackpropData->clone_with_new_inputs({
                        convolutionBackpropData->get_input_node_shared_ptr(0),
                        convolutionBackpropData->get_input_node_ptr(1)->get_input_node_shared_ptr(0),
                        convolutionBackpropData->get_input_node_shared_ptr(2)}) :
                    convolutionBackpropData->clone_with_new_inputs({
                        convolutionBackpropData->get_input_node_shared_ptr(0),
                        convolutionBackpropData->get_input_node_ptr(1)->get_input_node_shared_ptr(0)});
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
bool ConvolutionBackpropDataTransformation::canBeTransformed(const TransformationContext &context,
                                                 std::shared_ptr<Node> layer) const {
    if (!WeightableLayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer);
    if (!canSubtractBeHandled(layer, dequantization)) {
        return false;
    }

    if (updatePrecisions && !NetworkHelper::checkZeroPoint(dequantization.subtract)) {
        return false;
    }

    if (updatePrecisions && !dequantization.empty() && !dequantization.isLowPrecision()) {
        return false;
    }

    std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(layer->get_input_node_shared_ptr(1));
    dequantization = reshapeFromWeights == nullptr ?
                     NetworkHelper::getDequantization(layer, 1ul) :
                     NetworkHelper::getDequantization(reshapeFromWeights);

    const auto fqOnWeights = getFakeQuantizeOnWeights(layer);
    if (dequantization.empty()) {
        const auto dataPrecision = getDataPrecisionOnWeights(layer);
        if ((!supportAsymmetricQuantization) && dataPrecision.hasZeroPoint) {
            return false;
        }
        if (updatePrecisions && !NetworkHelper::checkZeroPoint(fqOnWeights, dataPrecision)) {
            const std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(fqOnWeights);
            if (as_type_ptr<opset1::Constant>(resultConstant)) {
                replace_node(fqOnWeights, resultConstant);
            }
            return false;
        }
    } else {
        if (updatePrecisions && !NetworkHelper::checkZeroPoint(dequantization.subtract)) {
            NetworkHelper::foldDequantization(dequantization.multiply, 0, true);
            return false;
        }
    }

    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
