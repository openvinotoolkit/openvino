// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/concat.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/common/subgraph.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ConcatTransformation, "ConcatTransformation", 0);

ConcatTransformation::ConcatTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = ngraph::pattern::wrap_type<opset1::Concat>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ConcatTransformation");
    this->register_matcher(m, callback);
}

bool ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());
    if (!canBeTransformed(context, concat)) {
        return false;
    }

    std::vector<FakeQuantizeDequantization> layerDequantizations;
    layerDequantizations.reserve(concat->get_input_size());
    for (size_t parentIndex = 0ul; parentIndex < concat->get_input_size(); parentIndex++) {
        FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, parentIndex);
        if (dequantization.empty()) {
            return false;
        }
        layerDequantizations.push_back(dequantization);
    }

    bool allDequantizationShiftAreZero = true;
    bool allDequantizationMultiplyAreZero = true;
    for (const auto& dequantization : layerDequantizations) {
        if (dequantization.subtract != nullptr) {
            allDequantizationShiftAreZero = false;
        }

        if (dequantization.multiply != nullptr) {
            allDequantizationMultiplyAreZero = false;
        }

        if (!allDequantizationShiftAreZero && !allDequantizationMultiplyAreZero) {
            break;
        }
    }

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

    OutputVector dataNodes;
    NodeVector convertNodes;
    NodeVector subtractNodes;
    NodeVector multiplyNodes;
    for (size_t i = 0; i < layerDequantizations.size(); ++i) {
        const auto& dequantization = layerDequantizations[i];

        dataNodes.push_back(dequantization.data);

        if (dequantization.convert != nullptr) {
            convertNodes.push_back(dequantization.convert);
        }

        Shape targetShape(concat->get_input_shape(i).size(), 1ul);
        targetShape[1] = concat->get_input_shape(i)[1];

        if (!allDequantizationShiftAreZero) {
            subtractNodes.push_back(dequantization.subtract == nullptr ?
                std::make_shared<ngraph::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 0.f })) :
                broadcastElementWiseConst(dequantization.subtractConstant, targetShape));
        }

        if (!allDequantizationMultiplyAreZero) {
            multiplyNodes.push_back(dequantization.multiply == nullptr ?
                std::make_shared<ngraph::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 1.0f })) :
                broadcastElementWiseConst(dequantization.multiplyConstant, targetShape));
        }
    }

    const auto newConcat = concat->clone_with_new_inputs(dataNodes);

    std::shared_ptr<ngraph::Node> lastDequantization = newConcat;
    if (!convertNodes.empty()) {
        const auto convert = convertNodes[0]->clone_with_new_inputs({ newConcat });

        //ngraph::copy_runtime_info({ layer, convert }, convert);
        NetworkHelper::copyInfo({ concat, convert }, convert);
        lastDequantization = convert;
    }

    // concatenation axis is 1
    if (!subtractNodes.empty()) {
        const auto subtract = std::make_shared<DequantizationSubtract>(
            lastDequantization,
            NetworkHelper::toScalarIfPossible(subtractNodes.size() == 1ul ?
                subtractNodes[0] :
                ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(subtractNodes, 1)));

        //ngraph::copy_runtime_info({ layer, subtract }, subtract);
        NetworkHelper::copyInfo({ concat, subtract }, subtract);
        lastDequantization = subtract;
    }

    if (!multiplyNodes.empty()) {
        const auto multiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            DequantizationMultiply(
                lastDequantization,
                NetworkHelper::toScalarIfPossible(multiplyNodes.size() == 1ul ?
                    multiplyNodes[0] :
                    ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(multiplyNodes, 1))),
            layerDequantizations[0].multiply->get_output_element_type(0));

        //ngraph::copy_runtime_info({ layer, multiply }, multiply);
        NetworkHelper::copyInfo({ concat, multiply }, multiply);
        lastDequantization = multiply;
    }

    replace_node(concat, lastDequantization);
    NetworkHelper::copyInfo(concat, newConcat);
    updateOutput(context, lastDequantization, newConcat);
    return true;
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

bool ConcatTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    std::shared_ptr<opset1::Concat> concat = as_type_ptr<opset1::Concat>(layer);
    if (concat == nullptr) {
        return false;
    }

    const auto axis = concat->get_axis();
    const size_t normalizedAxis = normalize_axis(concat->get_friendly_name(), axis, concat->get_output_partial_shape(0).rank());
    return normalizedAxis == 1ul;
}

void ConcatTransformation::fillDequantizationNodes(
    const std::vector<FakeQuantizeDequantization>& layerDequantizations,
    const std::shared_ptr<Node> layer,
    NodeVector& convertNodes,
    NodeVector& subtractNodes,
    NodeVector& multiplyNodes) const {
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
        for (const auto& dequantization : layerDequantizations) {
            if (dequantization.subtract != nullptr) {
                allDequantizationShiftAreZero = false;
            }
            if (dequantization.multiply != nullptr) {
                allDequantizationMultiplyAreZero = false;
            }
        }

        for (size_t i = 0; i < layerDequantizations.size(); ++i) {
            const auto& dequantization = layerDequantizations[i];
            const ngraph::element::Type precision = deqPrecision;
            ngraph::Shape targetShape(layer->get_input_shape(i).size(), 1ul);
            targetShape[1] = layer->get_input_shape(i)[1];

            if (dequantization.convert != nullptr) {
                convertNodes.push_back(dequantization.convert);
            }

            if (!allDequantizationShiftAreZero) {
                subtractNodes.push_back(dequantization.subtract == nullptr ?
                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 0.f })) :
                    broadcastElementWiseConst(dequantization.subtractConstant, targetShape));
            }

            if (!allDequantizationMultiplyAreZero) {
                multiplyNodes.push_back(dequantization.multiply == nullptr ?
                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 1.0f })) :
                    broadcastElementWiseConst(dequantization.multiplyConstant, targetShape));
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
}

std::shared_ptr<Node> ConcatTransformation::concatenateDeqNodes(NodeVector& nodes) const {
    return nodes.size() == 1ul ? nodes[0] : fold<ngraph::opset1::Concat>(nodes, 1);
}

void ConcatTransformation::addDequantizationLayers(
    TransformationContext& context,
    ngraph::pass::low_precision::Subgraph& subgraph,
    std::function<void(
        std::shared_ptr<ngraph::Node> layer,
        std::shared_ptr<ngraph::Node> child,
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
                    std::shared_ptr<ngraph::Node> source = layer;
                    const std::shared_ptr<ngraph::Node> destination = child.shared_from_this();

                    if (layerDequantizations.size() == 0ul) {
                        // fill layerDequantizations collection
                        getLayerDequantizationCallback(source, destination, source->get_friendly_name(), layerDequantizations);
                    }

                    {
                        NodeVector convertNodes;
                        NodeVector subtractNodes;
                        NodeVector multiplyNodes;

                        // forming nodes for concatenation
                        fillDequantizationNodes(layerDequantizations, layer, convertNodes, subtractNodes, multiplyNodes);

                        // TODO: the second place (first is FQ decomposition) where dequantization operations are inserted
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
                                NetworkHelper::toScalarIfPossible(concatenateDeqNodes(subtractNodes)));

                            insert_new_node_between(source, destination, subtract);
                            ngraph::copy_runtime_info({ layer, subtract }, subtract);
                            source = subtract;
                        }

                        if (!multiplyNodes.empty()) {
                            const size_t sourceOutputIdx = NetworkHelper::getChildInputIndex(source, destination);
                            std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
                                DequantizationMultiply(
                                    destination->get_input_source_output(sourceOutputIdx),
                                    NetworkHelper::toScalarIfPossible(concatenateDeqNodes(multiplyNodes))),
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

                        // Split & VariadicSplit have other naming rules
                        if (is_type<opset1::Split>(layer) || is_type<opset1::VariadicSplit>(layer)) {
                            source->set_friendly_name(originalName + "." + std::to_string(i));
                        } else {
                            source->set_friendly_name(originalName);
                        }
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
    return 0ul;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
