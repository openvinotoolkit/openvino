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
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ConcatTransformation");
    this->register_matcher(m, callback);
}

bool ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
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

    bool someDqInLowPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return value.isLowPrecision(); });

    bool someDqInFpPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return !value.isLowPrecision(); });

    bool DqWithDifferentPrecision = someDqInLowPrecision && someDqInFpPrecision;

    OutputVector dataNodes;
    NodeVector convertNodes;
    NodeVector subtractNodes;
    NodeVector multiplyNodes;
    for (size_t i = 0; i < layerDequantizations.size(); ++i) {
        const auto& dequantization = layerDequantizations[i];

        if (DqWithDifferentPrecision && dequantization.isLowPrecision()) {
            dataNodes.push_back(dequantization.convert);
        } else {
            dataNodes.push_back(dequantization.data);
        }

        if (dequantization.convert != nullptr) {
            convertNodes.push_back(dequantization.convert);
        }

        Shape targetShape(concat->get_input_partial_shape(i).rank().get_length(), 1ul);
        targetShape[1] = concat->get_input_partial_shape(i)[1].get_length();

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
    const auto outPShape = concat->get_output_partial_shape(0);
    const size_t normalizedAxis = ngraph::normalize_axis(concat->get_friendly_name(), axis, outPShape.rank());

    // TODO: LPT: to support current flow: #58269
    if (normalizedAxis != 1ul) {
        return false;
    }

    if (outPShape.rank().is_dynamic() || outPShape[normalizedAxis].is_dynamic()) {
        return false;
    }

    const bool perTensorQuantizationIsRequired = normalizedAxis != 1ul;

    element::Type precision;
    for (size_t i = 0ul; i < concat->get_input_size(); i++) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, i);
        if (dequantization.empty() || (updatePrecisions && !dequantization.isLowPrecision())) {
            return false;
        }

        if (precision == element::undefined) {
            precision = dequantization.data.get_element_type();
        } else if (precision != dequantization.data.get_element_type()) {
            return false;
        }

        if (perTensorQuantizationIsRequired &&
            (((dequantization.subtractConstant != nullptr) && !NetworkHelper::isScalarLike(dequantization.subtractConstant)) ||
            ((dequantization.multiplyConstant != nullptr) && !NetworkHelper::isScalarLike(dequantization.multiplyConstant)))) {
            return false;
        }
    }
    return true;
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
            ngraph::Shape targetShape(layer->get_input_partial_shape(i).rank().get_length(), 1ul);
            targetShape[1] = layer->get_input_partial_shape(i)[1].get_length();

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

bool ConcatTransformation::isHandled(const TransformationContext& context, const std::vector<std::shared_ptr<ngraph::Node>>& quantizationOperations) {
    for (const std::shared_ptr<ngraph::Node>& quantizationLayer : quantizationOperations) {
        if (context.quantizedFakeQuantizeNames.find(quantizationLayer->get_friendly_name()) != context.quantizedFakeQuantizeNames.end()) {
            return true;
        }
    }

    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
