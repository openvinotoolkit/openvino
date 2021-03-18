// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/concat.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/common/subgraph.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void ConcatTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::Concat>(pass, context);
}

bool ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());
    if (!canBeTransformed(context, concat)) {
        return false;
    }

    ngraph::pass::low_precision::Subgraph subgraph(layerTransformationsManager);
    std::unordered_set<std::string> handledLayers;
    if (!subgraph.fillSubgraphForConcat(concat, handledLayers)) {
        return false;
    }

    if (subgraph.quantizationLayers.empty() || isHandled(context, subgraph.quantizationLayers)) {
        return false;
    }

    // precisions can be different
    ngraph::Node& quantizationLayer = *subgraph.quantizationLayers[0];
    std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer.shared_from_this());
    DataPrecision dataPrecision = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false);
    if (dataPrecision.precision == ngraph::element::undefined) {
        return false;
    }

    std::unordered_map<std::string, ngraph::pass::low_precision::FakeQuantizeDequantization> dequantizations;
    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        const std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(subgraph.quantizationLayers[i]);
        if (fq == nullptr) {
            return false;
        }

        if (!NetworkHelper::isQuantizeSupported(fq)) {
            return false;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(fq);

        // per tensor scale is supported only
        if (quantizationDetails.inputHighValues.size() != 1ul) {
            return false;
        }

        const DataPrecision dataPrecision2 = getDataPrecision(subgraph.quantizationLayers[i]->shared_from_this(), quantizationDetails, false);
        if (dataPrecision2.precision == ngraph::element::undefined) {
            return false;
        }

        if (dataPrecision.precision != dataPrecision2.precision) {
            // quantization levels are the same, difference can be in sign
            // wider interval (precision) is preferable: use signed if least one interval is signed
            dataPrecision = dataPrecision.precision.is_signed() ? dataPrecision : dataPrecision2;
        }
    }

    if (dataPrecision.precision == ngraph::element::undefined) {
        return false;
    }

    std::vector<QuantizationDetails> quantizationLayersDetails;
    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        std::shared_ptr<opset1::FakeQuantize> fakeQuantize = as_type_ptr<opset1::FakeQuantize>(subgraph.quantizationLayers[i]);
        auto newFakeQuantize = NetworkHelper::fuseConvert(fakeQuantize);
        if (newFakeQuantize == nullptr) {
            subgraph.quantizationLayers[i] = fakeQuantize;
            quantizationLayersDetails.push_back(QuantizationDetails::getDetails(fakeQuantize));
            continue;
        }

        fakeQuantize = newFakeQuantize;
        newFakeQuantize = NetworkHelper::composeFakeQuantize(fakeQuantize);
        if (newFakeQuantize == nullptr) {
            subgraph.quantizationLayers[i] = fakeQuantize;
            quantizationLayersDetails.push_back(QuantizationDetails::getDetails(fakeQuantize));
            continue;
        }

        fakeQuantize = newFakeQuantize;
        subgraph.quantizationLayers[i] = fakeQuantize;
        quantizationLayersDetails.push_back(QuantizationDetails::getDetails(fakeQuantize));
    }

    FakeQuantizeDequantization dequantization;

    if ((quantizationLayersDetails[0].inputHighValues.size() == 1)) {
        float outputLowValue = quantizationLayersDetails[0].outputLowValues[0];
        float outputHighValue = quantizationLayersDetails[0].outputHighValues[0];

        for (size_t index = 0lu; index < subgraph.quantizationLayers.size(); index++) {
            const QuantizationDetails& quantizationDetails = quantizationLayersDetails[index];
            if (outputLowValue > quantizationDetails.outputLowValues[0]) {
                outputLowValue = quantizationDetails.outputLowValues[0];
            }
            if (outputHighValue < quantizationDetails.outputHighValues[0]) {
                outputHighValue = quantizationDetails.outputHighValues[0];
            }
        }

        if ((outputLowValue == 0.f) && (outputHighValue == 0.f)) {
            return false;
        }

        const float maxOutputInterval = outputHighValue - outputLowValue;
        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
            const size_t minLevels = getMinQuantizationLevels(
                dataPrecision,
                maxOutputInterval,
                quantizationLayersDetails,
                outputLowValue,
                outputHighValue);
            if (minLevels < this->minQuantizationLevels) {
                return false;
            }
        }

        // FQ -> SUB_quantization -> MUL_quantization -[INT8]-> SUB_dequantization -> MUL_dequantization ->
        const float quantizationMul = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;
        const float dequantizationMul = maxOutputInterval / (dataPrecision.max - dataPrecision.min);

        // FQ outputLowValue = dataPrecision.min * dequantizationMul - quantizationSub
        const float quantizationSub = outputLowValue - dataPrecision.min * dequantizationMul;
        const float dequantizationSub = std::round(-quantizationSub * quantizationMul);

        // 1. get data for dequantization. Dequantization data will be used several times later.
        dequantization = ngraph::pass::low_precision::NetworkHelper::makeDequantization(
            dequantizationMul,
            dequantizationSub,
            subgraph.quantizationLayers[0]->get_output_element_type(0),
            subgraph.quantizationLayers[0]->get_output_shape(0),
            updatePrecisions ? dataPrecision.precision : subgraph.quantizationLayers[0]->get_output_element_type(0),
            deqPrecision);

        for (size_t index = 0; index < subgraph.quantizationLayers.size(); index++) {
            std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantizeLayer = as_type_ptr<ngraph::opset1::FakeQuantize>(
                subgraph.quantizationLayers[index]->shared_from_this());

            const QuantizationDetails& quantizationDetails = quantizationLayersDetails[index];

            switch (quantizedTensorAlignmentOnActivations) {
                case QuantizedTensorAlignment::None: {
                    THROW_TRANSFORMATION_EXCEPTION << "not implemented: " << quantizedTensorAlignmentOnActivations;
                }
                case QuantizedTensorAlignment::UpdateLevel: {
                    const float updatedOutputLowValue = (quantizationDetails.outputLowValues[0] - quantizationSub) * quantizationMul;
                    const float updatedOutputHighValue = (quantizationDetails.outputHighValues[0] - quantizationSub) * quantizationMul;

                    // 2. update FakeQuantize - one time action
                    std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
                        fakeQuantizeLayer,
                        updatePrecisions ? dataPrecision.precision : fakeQuantizeLayer->get_output_element_type(0),
                        roundf(updatedOutputLowValue),
                        roundf(updatedOutputHighValue));

                    const size_t levels = static_cast<size_t>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
                    newFakeQuantizeLayer->set_levels(levels);

                    subgraph.quantizationLayers[index] = newFakeQuantizeLayer;
                    subgraph.layers[fakeQuantizeLayer->get_friendly_name()] = newFakeQuantizeLayer;
                    break;
                }
                default: {
                    THROW_TRANSFORMATION_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
                }
            }
        }
    } else {
        return false;
    }

    auto dequantizationValuesCallback = [&](
        std::shared_ptr<ngraph::Node> layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
        dequantizationsToConcatenate.push_back(dequantization);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            const std::shared_ptr<ngraph::Node>& node = it.second;
            if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(node) != nullptr) {
                ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(node->shared_from_this(), dataPrecision.precision);
            } else {
                // set precision to explicitly to have updated precision during transformation
                for (size_t i = 0; i < node->get_output_size(); ++i) {
                    node->set_output_type(i, dataPrecision.precision, node->get_output_partial_shape(i));
                }
            }
        }
    }

    for (const std::shared_ptr<ngraph::Node>& quantizationLayer : subgraph.quantizationLayers) {
        context.quantizedFakeQuantizeNames.insert(quantizationLayer->get_friendly_name());
    }
    return true;
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

bool ConcatTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    std::shared_ptr<opset1::Concat> concat = as_type_ptr<opset1::Concat>(layer);
    return concat && concat->get_axis() == 1ul;
}


void ConcatTransformation::addDequantizationLayers(
    TransformationContext& context,
    ngraph::pass::low_precision::Subgraph& subgraph,
    std::function<void(
        std::shared_ptr<ngraph::Node> layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate)> getLayerDequantizationCallback) const {
    std::unordered_map<std::string, ngraph::Node*> outputs;
    for (size_t i = 0; i < context.function->get_output_size(); ++i) {
        ngraph::Node* node = context.function->get_output_op(i).get();
        if (node->get_input_size() != 1ul) {
            THROW_IE_LPT_EXCEPTION(*node) << "unexpected inputs count for result node";
        }

        outputs.emplace(node->get_input_node_shared_ptr(0)->get_friendly_name(), node);
    }

    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>> notHandledSubgraphLayers = subgraph.layers;
    while (notHandledSubgraphLayers.size() != 0ul) {
        const auto layerIt = notHandledSubgraphLayers.begin();
        std::shared_ptr<ngraph::Node> layer = layerIt->second;
        notHandledSubgraphLayers.erase(layerIt);

        std::vector<FakeQuantizeDequantization> layerDequantizations;

        for (size_t i = 0; i < layer->get_output_size(); ++i) {
            const auto childInputs = layer->get_output_target_inputs(i);
            for (const auto childInput : childInputs) {
                ngraph::Node& child = *childInput.get_node();

                if (subgraph.layers.find(child.get_friendly_name()) == subgraph.layers.end()) {
                    if (layerDequantizations.size() == 0ul) {
                        // fill layerDequantizations collection
                        getLayerDequantizationCallback(layer, layer->get_friendly_name(), layerDequantizations);
                    }

                    std::shared_ptr<ngraph::Node> source = layer->shared_from_this();
                    {
                        std::vector<std::shared_ptr<ngraph::Node>> convertNodes;
                        std::vector<std::shared_ptr<ngraph::Node>> subtractNodes;
                        std::vector<std::shared_ptr<ngraph::Node>> multiplyNodes;

                        // forming nodes for concatenation
                        if (layerDequantizations.size() > 1ul) {
                            auto broadcastElementWiseConst = [](
                                // FakeQuantize constant shape must be broadcastable to the shape on data.
                                std::shared_ptr<ngraph::opset1::Constant> operation,
                                const ngraph::Shape targetShape) -> std::shared_ptr<Node> {
                                auto targetShapeConst = std::make_shared<ngraph::opset1::Constant>(
                                    element::i64, ngraph::Shape{ targetShape.size() },
                                    targetShape);

                                auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
                                    operation,
                                    targetShapeConst,
                                    ngraph::op::AutoBroadcastType::NUMPY);

                                return broadcast;
                            };

                            bool allDequantizationShiftAreZero = true;
                            bool allDequantizationMultiplyAreZero = true;
                            for (FakeQuantizeDequantization dequantization : layerDequantizations) {
                                if (dequantization.subtract != nullptr) {
                                    allDequantizationShiftAreZero = false;
                                }
                                if (dequantization.multiply != nullptr) {
                                    allDequantizationMultiplyAreZero = false;
                                }
                            }

                            for (size_t i = 0; i < layerDequantizations.size(); ++i) {
                                const auto& dequantization = layerDequantizations[i];

                                if (dequantization.convert != nullptr) {
                                    convertNodes.push_back(dequantization.convert);
                                }

                                const ngraph::element::Type precision = deqPrecision;
                                ngraph::Shape targetShape(layer->get_input_shape(i).size(), 1ul);
                                targetShape[1] = layer->get_input_shape(i)[1];

                                if (!allDequantizationShiftAreZero) {
                                    subtractNodes.push_back(dequantization.subtract == nullptr ?
                                        std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 0.f })) :
                                        broadcastElementWiseConst(
                                            as_type_ptr<ngraph::opset1::Constant>(dequantization.subtract->input_value(1).get_node_shared_ptr()),
                                            targetShape));
                                }

                                if (!allDequantizationMultiplyAreZero) {
                                    multiplyNodes.push_back(dequantization.multiply == nullptr ?
                                        std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 1.0f })) :
                                        broadcastElementWiseConst(
                                            as_type_ptr<ngraph::opset1::Constant>(dequantization.multiply->input_value(1).get_node_shared_ptr()),
                                            targetShape));
                                }
                            }
                        } else {
                            // TODO: check constant shapes here - has to be scalar
                            if (layerDequantizations[0].convert != nullptr) {
                                convertNodes.push_back(layerDequantizations[0].convert);
                            }

                            if (layerDequantizations[0].subtract != nullptr) {
                                subtractNodes.push_back(layerDequantizations[0].subtract->input_value(1).get_node_shared_ptr());
                            }

                            if (layerDequantizations[0].multiply != nullptr) {
                                multiplyNodes.push_back(layerDequantizations[0].multiply->input_value(1).get_node_shared_ptr());
                            }
                        }

                        // TODO: the second place (first is FQ decomposition) where dequantization operations are inserted
                        const std::shared_ptr<ngraph::Node> destination = child.shared_from_this();

                        if (!convertNodes.empty()) {
                            const size_t sourceOutputIdx = NetworkHelper::getChildInputIndex(source, destination);
                            std::shared_ptr<ngraph::Node> convert =
                                convertNodes[0]->clone_with_new_inputs({ destination->get_input_source_output(sourceOutputIdx) });
                            insert_new_node_between(source, destination, convert);
                            ngraph::copy_runtime_info({ layer, convert }, convert);
                            source = convert;
                        }

                        // concatenation axis is 1
                        if (!subtractNodes.empty()) {
                            const size_t sourceOutputIdx = NetworkHelper::getChildInputIndex(source, destination);
                            std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<DequantizationSubtract>(
                                destination->get_input_source_output(sourceOutputIdx),
                                NetworkHelper::toScalarIfPossible(subtractNodes.size() == 1ul ?
                                    subtractNodes[0] :
                                    ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(subtractNodes, 1)));
                            insert_new_node_between(source, destination, subtract);
                            ngraph::copy_runtime_info({ layer, subtract }, subtract);
                            source = subtract;
                        }

                        if (!multiplyNodes.empty()) {
                            const size_t sourceOutputIdx = NetworkHelper::getChildInputIndex(source, destination);
                            std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
                                DequantizationMultiply(
                                    destination->get_input_source_output(sourceOutputIdx),
                                    NetworkHelper::toScalarIfPossible(multiplyNodes.size() == 1ul ?
                                        multiplyNodes[0] :
                                        ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(multiplyNodes, 1))),
                                    layerDequantizations[0].multiply->get_output_element_type(0));
                            insert_new_node_between(source, destination, multiply);
                            ngraph::copy_runtime_info({ layer, multiply }, multiply);
                            source = multiply;
                        }
                    }

                    // first input is used
                    const ngraph::element::Type precision = layerDequantizations[0].data.get_element_type();
                    layer->set_output_type(0, precision, layer->get_output_partial_shape(0));

                    const auto it = outputs.find(layer->get_friendly_name());
                    if (it != outputs.end() && is_type<ngraph::opset1::Result>(child.shared_from_this())) {
                        const std::string originalName = layer->get_friendly_name();
                        const std::string newName = layer->get_friendly_name() + LayerTransformation::originalLayerPostfix;
                        layer->set_friendly_name(newName);
                        source->set_friendly_name(originalName);
                        subgraph.layers[layer->get_friendly_name()] = layer;
                    }
                }
            }
        }
    }
}

bool ConcatTransformation::isHandled(const TransformationContext& context, const std::vector<std::shared_ptr<ngraph::Node>>& quantizationOperations) {
    for (const std::shared_ptr<ngraph::Node>& quantizationLayer : quantizationOperations) {
        if (context.quantizedFakeQuantizeNames.find(quantizationLayer->get_friendly_name()) != context.quantizedFakeQuantizeNames.end()) {
            return true;
        }
    }

    return false;
}

size_t ConcatTransformation::getMinQuantizationLevels(
    const DataPrecision& dataPrecision,
    const float maxOutputInterval,
    const std::vector<QuantizationDetails>& quantizationLayersDetails,
    const float outputLowValue,
    const float outputHighValue) const {
    size_t minLevels = std::numeric_limits<std::size_t>::max();
    for (const QuantizationDetails quantizationDetails : quantizationLayersDetails) {
        // if there is negative part then calculation is based on `outputLowValue` if not then on `outputHighValue` only
        const float updatedOutputLowValue = outputLowValue != 0.f ?
            (quantizationDetails.outputLowValues[0] / outputLowValue) * dataPrecision.min :
            (quantizationDetails.outputLowValues[0] / outputHighValue) * dataPrecision.max;

        // if there is positive part then calculation is based on `outputHighValue` if not then on `outputLowValue` only
        const float updatedOutputHighValue = outputHighValue != 0.f ?
            (quantizationDetails.outputHighValues[0] / outputHighValue) * dataPrecision.max :
            (quantizationDetails.outputHighValues[0] / outputLowValue) * dataPrecision.min;

        const size_t levels = static_cast<size_t>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
        if (minLevels > levels) {
            minLevels = levels;
        }
    }
    return minLevels;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
