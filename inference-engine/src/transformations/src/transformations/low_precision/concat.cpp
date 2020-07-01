// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/concat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/common/fake_quantize_dequantization.hpp"
#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/common/subgraph.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void ConcatTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    // TODO: new way
    //addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::Concat>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::Multiply>() }));

    // TODO: current way

    // TODO: unlimited FQ amount
    // addSingleNodePattern<opset1::FakeQuantize>(pass, context);

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Concat>({ make_op_label<ngraph::opset1::FakeQuantize>(), make_op_label<ngraph::opset1::FakeQuantize>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Concat>({
            make_op_label<ngraph::opset1::FakeQuantize>(),
            make_op_label<ngraph::opset1::FakeQuantize>(),
            make_op_label<ngraph::opset1::FakeQuantize>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Concat>({
            make_op_label<ngraph::opset1::FakeQuantize>(),
            make_op_label<ngraph::opset1::FakeQuantize>(),
            make_op_label<ngraph::opset1::FakeQuantize>(),
            make_op_label<ngraph::opset1::FakeQuantize>() }));
}

void ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());

    ngraph::pass::low_precision::Subgraph subgraph;
    std::unordered_set<std::string> handledLayers;
    if (!subgraph.fillSubgraphForConcat(*concat, handledLayers)) {
        return;
    }

    // TODO: check if FQ has been handled already
    // for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
    //    if (context.quantizedFakeQuantizeNames.find(quantizationLayer->name) != context.quantizedFakeQuantizeNames.end()) {
    //        return;
    //    }
    // }

    // TODO: update later
    // TODO: check if precisions are different and return
    ngraph::Node& quantizationLayer = *subgraph.quantizationLayers[0];
    std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer.shared_from_this());
    DataPrecision dataPrecision = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false, false);
    if (dataPrecision.precision == ngraph::element::undefined) {
        return;
    }

    // TODO: use raw pointer instead names
    std::unordered_map<std::string, ngraph::pass::low_precision::FakeQuantizeDequantization> dequantizations;
    std::vector<QuantizationDetails> quantizationLayersDetails;

    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        ngraph::Node* fakeQuantizeLayer = subgraph.quantizationLayers[i];
        std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fakeQuantizeLayer->shared_from_this());

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(fq);
        quantizationLayersDetails.push_back(quantizationDetails);

        const DataPrecision dataPrecision2 = getDataPrecision(subgraph.quantizationLayers[i]->shared_from_this(), quantizationDetails, false, false);
        if (dataPrecision2.precision == ngraph::element::undefined) {
            return;
        }

        if (dataPrecision.precision != dataPrecision2.precision) {
            // quantization levels are the same, difference can be in sign
            // wider interval (precision) is preferable: use signed if least one interval is signed
            dataPrecision = dataPrecision.precision.is_signed() ? dataPrecision : dataPrecision2;
        }
    }

    if (dataPrecision.precision == ngraph::element::undefined) {
        return;
    }

    // per tensor scale is supported only
    if (quantizationLayersDetails.empty() || (quantizationLayersDetails[0].inputHighValues.size() != 1ul)) {
        return;
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
            return;
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
                return;
            }
        }


        const float dequantizationScale = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
        const float max = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.max);
        const float min = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.min);
        const float dequantizationShift = outputLowValue - min;

        const float quantizationScale = 1.f / dequantizationScale;
        const float quantizationShift = - dequantizationShift * quantizationScale;

        // 1. get data for dequantization. Dequantization data will be used several times later.
        dequantization = ngraph::pass::low_precision::createDequantization(
            dequantizationScale,
            dequantizationShift,
            subgraph.quantizationLayers[0]->get_output_element_type(0),
            subgraph.quantizationLayers[0]->get_output_shape(0),
            dataPrecision.precision,
            dataPrecision.min,
            dataPrecision.max);

        for (int index = 0; index < subgraph.quantizationLayers.size(); index++) {
            std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantizeLayer = as_type_ptr<ngraph::opset1::FakeQuantize>(subgraph.quantizationLayers[index]->shared_from_this());
            const QuantizationDetails& quantizationDetails = quantizationLayersDetails[index];

            switch (quantizedTensorAlignmentOnActivations) {
                case QuantizedTensorAlignment::None:
                case QuantizedTensorAlignment::UpdateIntervals: {
                    THROW_TRANSFORMATION_EXCEPTION << "not implemented: " << quantizedTensorAlignmentOnActivations;
                }
                case QuantizedTensorAlignment::UpdateLevel: {
                    // TODO: reuse ngraph::pass::low_precision::updateFakeQuantize(fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);

                    const float updatedOutputLowValue = quantizationDetails.outputLowValues[0] * quantizationScale + quantizationShift;
                    const float updatedOutputHighValue = quantizationDetails.outputHighValues[0] * quantizationScale + quantizationShift;

                    // replace_node(
                    //    fakeQuantizeLayer->get_input_node_shared_ptr(3),
                    //    std::make_shared<ngraph::opset1::Constant>(
                    //        fakeQuantizeLayer->get_input_element_type(3),
                    //        Shape({}),
                    //        updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue));

                    // replace_node(
                    //    fakeQuantizeLayer->get_input_node_shared_ptr(4),
                    //    std::make_shared<ngraph::opset1::Constant>(
                    //        fakeQuantizeLayer->get_input_element_type(4),
                    //        Shape({}),
                    //        updatePrecisions ? roundf(updatedOutputHighValue) : updatedOutputHighValue));

                    // 2. update FakeQuantize - one time action
                    std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::updateFakeQuantize(
                        fakeQuantizeLayer,
                        dataPrecision.precision,
                        updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue,
                        updatePrecisions ? roundf(updatedOutputHighValue) : updatedOutputHighValue);

                    const size_t levels = static_cast<size_t>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
                    newFakeQuantizeLayer->set_levels(levels);

                    subgraph.quantizationLayers[index] = newFakeQuantizeLayer.get();

                    // TODO: debug only
                    // std::cout << "ConcatTransformation::transform" << newFakeQuantizeLayer->get_friendly_name() << ": " << levels << ": " <<
                    //    roundf(updatedOutputLowValue) << "-" << roundf(updatedOutputHighValue) << std::endl;
                    break;
                }
                default: {
                    THROW_TRANSFORMATION_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
                }
            }
        }
    } else {
        return;
    }

    // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    auto dequantizationValuesCallback = [&](
        ngraph::Node& layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
        dequantizationsToConcatenate.push_back(dequantization);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    // std::cout << "ConcatTransformation::transform" << concat->get_friendly_name() << std::endl;
    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            ngraph::Node* node = it.second;
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(node->shared_from_this(), dataPrecision.precision);
            // std::cout << "\t" << node->get_friendly_name() << ": " << dataPrecision.precision << std::endl;
        }
    }

    // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });
    // std::cout << "ConcatTransformation::transform: done: " << concat->get_friendly_name() << std::endl;
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

void ConcatTransformation::addDequantizationLayers(
    TransformationContext& context,
    ngraph::pass::low_precision::Subgraph& subgraph,
    std::function<void(
        ngraph::Node& layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate)> getLayerDequantizationCallback) const {
    std::unordered_map<std::string, ngraph::Node*> outputs;
    for (size_t i = 0; i < context.network->get_output_size(); ++i) {
        ngraph::Node* node = context.network->get_output_op(i).get();
        if (node->get_input_size() != 1ul) {
            THROW_IE_LPT_EXCEPTION(*node) << "unexpected inputs count for result node";
        }

        outputs.emplace(node->get_input_node_shared_ptr(0)->get_friendly_name(), node);
    }

    std::unordered_map<std::string, ngraph::Node*> notHandledSubgraphLayers = subgraph.layers;
    while (notHandledSubgraphLayers.size() != 0ul) {
        const auto layerIt = notHandledSubgraphLayers.begin();
        ngraph::Node* layer = layerIt->second;
        notHandledSubgraphLayers.erase(layerIt);

        std::vector<FakeQuantizeDequantization> layerDequantizations;

        for (int i = 0; i < layer->get_output_size(); ++i) {
            const auto childInputs = layer->get_output_target_inputs(i);
            for (const auto childInput : childInputs) {
                ngraph::Node& child = *childInput.get_node();
                if (subgraph.layers.find(child.get_friendly_name()) == subgraph.layers.end()) {
                    // child operation is out of Concat subgraph: we need to add dequantization operations
                    // std::cout << "\tadd dequantization operations: " << layer->get_friendly_name() << " -> " << child.get_friendly_name() << std::endl;
                    if (layerDequantizations.size() == 0ul) {
                        getLayerDequantizationCallback(*layer, layer->get_friendly_name(), layerDequantizations);
                    }

                    std::shared_ptr<ngraph::Node> source = layer->shared_from_this();

                    // TODO: remove to separate method: addDequantizationBetween
                    {
                        std::vector<std::shared_ptr<ngraph::Node>> convertNodes;
                        std::vector<std::shared_ptr<ngraph::Node>> subtractNodes;
                        std::vector<std::shared_ptr<ngraph::Node>> multiplyNodes;

                        if (layerDequantizations.size() > 1ul) {
                            auto broadcastElementWiseConst = [](
                                std::shared_ptr<ngraph::opset1::Constant> operation,
                                const ngraph::Shape targetShape) -> std::shared_ptr<Node> {
                                auto unsqueeze = ngraph::pass::low_precision::fold<ngraph::opset1::Unsqueeze>(
                                    operation->shared_from_this(),
                                    std::make_shared<ngraph::opset1::Constant>(element::i64, ngraph::Shape{ 4 }, std::vector<size_t>{ 0, 1, 2, 3 }));

                                auto targetShapeConst = std::make_shared<ngraph::opset1::Constant>(
                                    element::i64, ngraph::Shape{ targetShape.size() },
                                    targetShape);

                                auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
                                    unsqueeze,
                                    targetShapeConst,
                                    ngraph::op::AutoBroadcastType::NUMPY);

                                return broadcast;
                            };

                            for (FakeQuantizeDequantization dequantization : layerDequantizations) {
                                convertNodes.push_back(dequantization.convert);

                                const ngraph::element::Type precision = dequantization.data->get_output_element_type(0);
                                ngraph::Shape targetShape = dequantization.data->get_output_shape(0);

                                // TODO: shape is hardcoded;
                                if (targetShape.size() != 4ul) {
                                    THROW_TRANSFORMATION_EXCEPTION << "not supported yet";
                                }

                                targetShape[0] = 1ul;
                                targetShape[2] = 1ul;
                                targetShape[3] = 1ul;

                                subtractNodes.push_back(dequantization.subtract == nullptr ?
                                    // TODO: question vector precision is hardcoded - is it OK?
                                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<size_t>({ 0 })) :
                                    broadcastElementWiseConst(
                                        as_type_ptr<ngraph::opset1::Constant>(dequantization.subtract->input_value(1).get_node_shared_ptr()),
                                        targetShape));

                                multiplyNodes.push_back(dequantization.multiply == nullptr ?
                                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 1.0 })) :
                                    broadcastElementWiseConst(
                                        as_type_ptr<ngraph::opset1::Constant>(dequantization.multiply->input_value(1).get_node_shared_ptr()),
                                        targetShape));
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

                        // TODO: to debug only
                        // std::cout << std::endl;

                        // TODO: the second place (first is FQ decomposition) where dequantization operations are inserted
                        const std::shared_ptr<ngraph::Node> destination = child.shared_from_this();

                        if (!convertNodes.empty()) {
                            std::shared_ptr<ngraph::Node> convert = convertNodes[0]->clone_with_new_inputs({ source });
                            insert_new_node_between(source, destination, convert);
                            source = convert;
                        }

                        // TODO: concatenation axis is hardcoded

                        if (!subtractNodes.empty()) {
                            // TODO: debug only
                            // ngraph::Shape sourceShape = source->get_output_shape(0);
                            // ngraph::element::Type sourceConstType = source->get_output_element_type(0);
                            // ngraph::Shape subtractConstShape = subtractNodes[0]->get_output_shape(0);
                            // ngraph::element::Type subtractConstType = subtractNodes[0]->get_output_element_type(0);

                            std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
                                source,
                                subtractNodes.size() == 1ul ?
                                    subtractNodes[0] :
                                    ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(subtractNodes, 1));
                            insert_new_node_between(source, destination, subtract);
                            source = subtract;

                            // std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ context.network };
                            // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);
                        }

                        if (!multiplyNodes.empty()) {
                            std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
                                source,
                                multiplyNodes.size() == 1ul ?
                                    multiplyNodes[0] :
                                    ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(multiplyNodes, 1));
                            insert_new_node_between(source, destination, multiply);
                            source = multiply;
                        }
                    }

                    // layer->set_output_type(0, layerDequantizations[0].precisionBeforeDequantization, layer->get_output_partial_shape(0));
                    // TODO: why first input is used?
                    const ngraph::element::Type precision = layerDequantizations[0].data->get_output_element_type(0);
                    layer->set_output_type(0, precision, layer->get_output_partial_shape(0));

                    const auto it = outputs.find(layer->get_friendly_name());
                    if (it != outputs.end()) {
                        const std::string originalName = layer->get_friendly_name();
                        const std::string newName = layer->get_friendly_name() + "_original"; // LayerTransformation::lastLayerPostfix;
                        layer->set_friendly_name(newName);
                        source->set_friendly_name(originalName);
                        subgraph.layers[layer->get_friendly_name()] = layer;
                    }

                    // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });
                }
            }
        }
    }
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

        const int levels = static_cast<int>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
        if (minLevels > levels) {
            minLevels = levels;
        }
    }
    return minLevels;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
