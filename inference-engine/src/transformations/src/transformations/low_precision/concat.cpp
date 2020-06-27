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
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Concat>({ make_op_label<ngraph::opset1::FakeQuantize>(), make_op_label<ngraph::opset1::FakeQuantize>() }));
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

    // TODO: uncomment
    // if (!isMultiChannel(subgraph.concatLayers)) {
    //    ConcatTransformation::transform(context, concat);
    //    return;
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

    for (ngraph::Node* fakeQuantizeLayer : subgraph.quantizationLayers) {
        std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fakeQuantizeLayer->shared_from_this());

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(fq);

        // TODO: uncomment
        // const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(fq);
        // const size_t channelsCount = 3ul;

        // std::vector<float> dequantizationScales(channelsCount);
        // std::vector<float> dequantizationShifts(channelsCount);
        // for (size_t i = 0ul; i < channelsCount; ++i) {
        //    dequantizationScales[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
        //        (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) / (dataPrecision.max - dataPrecision.min) :
        //        1.0;

        //    dequantizationShifts[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
        //        (quantizationDetails.getOutputHighValue(i) - (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) *
        //        (dataPrecision.max / (dataPrecision.max - dataPrecision.min))) :
        //        0.f;
        // }
        // checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);

        // 1. get data for dequantization. Dequantization data will be used several times.
        FakeQuantizeDequantization fakeQuantizeDequantization = ngraph::pass::low_precision::getFakeQuantizeDequantization(
            fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);
        dequantizations[fakeQuantizeLayer->get_friendly_name()] = fakeQuantizeDequantization;

        // 2. update FakeQuantize - one time action
        ngraph::pass::low_precision::updateFakeQuantize(fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);
    }

    //if (updatePrecisions) {
    //    for (const auto it : subgraph.layers) {
    //        const CNNLayer* layer = it.second;
    //        CNNNetworkHelper::setOutDataPrecision(*layer, dataPrecision.precision);
    //    }
    //}

    auto dequantizationValuesCallback = [&](
        ngraph::Node& layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
        if (layer.get_friendly_name() != originalLayerName) {
            const auto update = [](
                const std::string& originalLayerName,
                const std::string& newLayerName,
                std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationLayers) {
                auto it = dequantizationLayers.find(originalLayerName);
                if (it != dequantizationLayers.end()) {
                    dequantizationLayers.emplace(newLayerName, it->second);
                    dequantizationLayers.erase(it);
                }
            };
            update(originalLayerName, layer.get_friendly_name(), dequantizations);
        }

        fillDequantization(
            layer,
            dequantizations,
            dequantizationsToConcatenate);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    //for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
    //    context.quantizedFakeQuantizeNames.insert(quantizationLayer->name);
    //}
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

// TODO: move to base class
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
                    if (layerDequantizations.size() == 0ul) {
                        getLayerDequantizationCallback(*layer, layer->get_friendly_name(), layerDequantizations);
                    }

                    std::shared_ptr<ngraph::Node> source = layer->shared_from_this();

                    // TODO: remove to separate method: addDequantizationBetween
                    {
                        std::vector<std::shared_ptr<ngraph::Node>> subtractNodes;
                        std::vector< std::shared_ptr<ngraph::Node>> multiplyNodes;
                        for (FakeQuantizeDequantization dequantization : layerDequantizations) {
                            // TODO: check: Convert has to exist

                            // TODO: refactor: create lambda
                            {
                                auto unsqueeze = ngraph::pass::low_precision::fold<ngraph::opset1::Unsqueeze>(
                                    dequantization.subtract->get_input_node_ptr(1)->shared_from_this(),
                                    std::make_shared<ngraph::opset1::Constant>(element::i64, ngraph::Shape{ 3 }, std::vector<size_t>{ 1, 2, 3 }));

                                auto targetShape = std::make_shared<ngraph::opset1::Constant>(
                                    element::i64, ngraph::Shape{ 4 },
                                    std::vector<size_t>{ 1, 3, 1, 1 });

                                auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
                                    unsqueeze,
                                    targetShape,
                                    ngraph::op::AutoBroadcastType::NUMPY);

                                subtractNodes.push_back(broadcast);
                            }

                            {
                                auto unsqueeze = ngraph::pass::low_precision::fold<ngraph::opset1::Unsqueeze>(
                                    dequantization.multiply->get_input_node_ptr(1)->shared_from_this(),
                                    std::make_shared<ngraph::opset1::Constant>(element::i64, ngraph::Shape{ 3 }, std::vector<size_t>{ 1, 2, 3 }));

                                auto targetShape = std::make_shared<ngraph::opset1::Constant>(
                                    element::i64, ngraph::Shape{ 4 },
                                    std::vector<size_t>{ 1, 3, 1, 1 });

                                auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
                                    unsqueeze,
                                    targetShape,
                                    ngraph::op::AutoBroadcastType::NUMPY);

                                multiplyNodes.push_back(broadcast);
                            }
                        }

                        const std::shared_ptr<ngraph::Node> destination = child.shared_from_this();

                        FakeQuantizeDequantization resultDequantization = layerDequantizations[0];
                        if (resultDequantization.convert != nullptr) {
                            std::shared_ptr<ngraph::Node> convert = resultDequantization.convert->clone_with_new_inputs({ source });
                            insert_new_node_between(source, destination, convert);
                            source = convert;
                        }

                        if (resultDequantization.subtract != nullptr) {
                            std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
                                source,
                                ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(subtractNodes, 1));
                            insert_new_node_between(source, destination, subtract);
                            source = subtract;
                        }

                        if (resultDequantization.multiply != nullptr) {
                            std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
                                source,
                                ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(multiplyNodes, 1));
                            insert_new_node_between(source, destination, multiply);
                            source = multiply;
                        }
                    }

                    // layer->set_output_type(0, layerDequantizations[0].precisionBeforeDequantization, layer->get_output_partial_shape(0));
                    // TODO: why first input is used?
                    const ngraph::element::Type precision = layerDequantizations[0].dataNode->get_input_element_type(0);
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

// fill dequantizationsToMerge collection for layer with using dequantizationByFakeQuantize
void ConcatTransformation::fillDequantization(
    ngraph::Node& layer,
    const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
    std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
    std::vector<ngraph::opset1::FakeQuantize*> fakeQuantizes;
    ngraph::opset1::FakeQuantize* currentFakeQuantize = ngraph::as_type<ngraph::opset1::FakeQuantize>(&layer);
    if (currentFakeQuantize != nullptr) {
        fakeQuantizes.push_back(currentFakeQuantize);
    } else {
        fillQuantization(layer, fakeQuantizes);
    }

    for (const ngraph::opset1::FakeQuantize* fakeQuantize : fakeQuantizes) {
        const auto it = dequantizationByFakeQuantize.find(fakeQuantize->get_friendly_name());
        if (it == dequantizationByFakeQuantize.end()) {
            THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization scale values are not found";
        }
        const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
        dequantizationsToConcatenate.push_back(fakeQuantizeDequantization);
    }
}

void ConcatTransformation::fillQuantization(const ngraph::Node& layer, std::vector<ngraph::opset1::FakeQuantize*>& fakeQuantizes) {
    for (int i = 0; i < layer.get_input_size(); ++i) {
        ngraph::Node* parent = layer.get_input_node_ptr(i);
        ngraph::opset1::FakeQuantize* fakeQuantize = ngraph::as_type<ngraph::opset1::FakeQuantize>(parent);
        if (fakeQuantize != nullptr) {
            fakeQuantizes.push_back(fakeQuantize);
        } else {
            fillQuantization(*parent, fakeQuantizes);
        }
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
