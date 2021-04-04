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

ConcatTransformation::ConcatTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = ngraph::pattern::wrap_type<opset1::Concat>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || m_transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ConcatTransformation");
    this->register_matcher(m, callback);
}

void ConcatTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::Concat>(pass, context);
}

bool ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());
    if (!canBeTransformed(context, concat)) {
        return false;
    }

    FakeQuantizeDequantization resultDequantization;

    // TODO: not completed
    std::vector<FakeQuantizeDequantization> layerDequantizations;
    layerDequantizations.reserve(concat->get_input_size());
    for (size_t parentIndex = 0ul; parentIndex < concat->get_input_size(); parentIndex++) {
        FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, parentIndex);
        if (dequantization.empty()) {
            return false;
        }
        layerDequantizations.push_back(dequantization);
    }


    std::shared_ptr<ngraph::Node> lastDequantization;

    std::shared_ptr<ngraph::Node> source = concat;
    std::shared_ptr<ngraph::Node> layer = concat;
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

                if (!allDequantizationShiftAreZero && !allDequantizationMultiplyAreZero) {
                    break;
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

        // TODO: not completed
        const std::shared_ptr<ngraph::Node> destination = concat->outputs()[0].get_target_inputs().begin()->get_node()->shared_from_this();

        if (!convertNodes.empty()) {
            const size_t sourceOutputIdx = NetworkHelper::getChildInputIndex(source, destination);
            std::shared_ptr<ngraph::Node> convert =
                convertNodes[0]->clone_with_new_inputs({ destination->get_input_source_output(sourceOutputIdx) });
            insert_new_node_between(source, destination, convert);
            //ngraph::copy_runtime_info({ layer, convert }, convert);
            NetworkHelper::copyInfo({ layer, convert }, convert);
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
            //ngraph::copy_runtime_info({ layer, subtract }, subtract);
            NetworkHelper::copyInfo({ layer, subtract }, subtract);
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
            //ngraph::copy_runtime_info({ layer, multiply }, multiply);
            NetworkHelper::copyInfo({ layer, multiply }, multiply);
            source = multiply;
            lastDequantization = multiply;
        }
    }

    // TODO: debug only
    const auto precision1 = layerDequantizations[0].data.get_element_type();
    const auto precision2 = layerDequantizations[1].data.get_element_type();

    auto newConcat = std::make_shared<opset1::Concat>(OutputVector{ layerDequantizations[0].data, layerDequantizations[1].data }, 1ul);
    replace_node(concat, newConcat);
    NetworkHelper::copyInfo(concat, newConcat);

    updateOutput(context, lastDequantization, newConcat);

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
