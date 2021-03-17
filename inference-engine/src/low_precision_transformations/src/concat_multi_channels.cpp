// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/concat_multi_channels.hpp"

#include <queue>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/common/subgraph.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

bool ConcatMultiChannelsTransformation::isMultiChannel(const std::vector<std::shared_ptr<ngraph::opset1::Concat>>& concatLayers) const noexcept {
    for (const std::shared_ptr<ngraph::opset1::Concat>& concat : concatLayers) {
        const std::vector<std::shared_ptr<ngraph::Node>> children = getChildrenRecursivelyExceptPrecisionPreserved(concat);
        for (const std::shared_ptr<ngraph::Node>& child : children) {
            if (is_type<ngraph::opset1::Convolution>(child.get())) {
                return false;
            }
        }
    }
    return true;
}

void ConcatMultiChannelsTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::Concat>(pass, context);
}

bool ConcatMultiChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
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

    if (!isMultiChannel(subgraph.concatLayers)) {
        ConcatTransformation::transform(context, m);
        return false;
    }

    DataPrecision dataPrecision;
    {
        for (auto quantizationLayer : subgraph.quantizationLayers) {
            std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer->shared_from_this());
            const DataPrecision tmp = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false);

            if (dataPrecision.precision == ngraph::element::undefined) {
                dataPrecision = tmp;
                continue;
            }

            if ((tmp.precision != dataPrecision.precision) && (tmp.precision == ngraph::element::u8)) {
                dataPrecision = tmp;
            }
        }
    }

    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        const std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(subgraph.quantizationLayers[i]);
        if (fq == nullptr) {
            return false;
        }

        if (!NetworkHelper::isQuantizeSupported(fq)) {
            return false;
        }
    }

    std::unordered_map<std::string, ngraph::pass::low_precision::FakeQuantizeDequantization> dequantizations;

    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        const std::shared_ptr<ngraph::Node>& fakeQuantizeLayer = subgraph.quantizationLayers[i];

        std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fakeQuantizeLayer->shared_from_this());
        assert(fq);

        auto newFakeQuantize = NetworkHelper::fuseConvert(fq);
        if (newFakeQuantize != nullptr) {
            fq = newFakeQuantize;
        }

        newFakeQuantize = NetworkHelper::composeFakeQuantize(fq);
        if (newFakeQuantize != nullptr) {
            fq = newFakeQuantize;
        }

        const DataPrecision currentDataPrecision = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false);
        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);

        // 1. get data for dequantization. Dequantization data will be used several times later.
        const FakeQuantizeDequantization fakeQuantizeDequantization = ngraph::pass::low_precision::NetworkHelper::createDequantizationFromFakeQuantize(
            fq,
            dataPrecision.precision,
            dataPrecision.min,
            dataPrecision.max,
            dataPrecision.precision == currentDataPrecision.precision ? currentDataPrecision.hasZeroPoint : true,
            updatePrecisions,
            deqPrecision);
        dequantizations[fakeQuantizeLayer->get_friendly_name()] = fakeQuantizeDequantization;

        // 2. update FakeQuantize - one time action
        const std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
            fq,
            updatePrecisions ? dataPrecision.precision : fakeQuantizeLayer->get_output_element_type(0),
            roundf(dataPrecision.min),
            roundf(dataPrecision.max));

        subgraph.quantizationLayers[i] = newFakeQuantizeLayer;
        subgraph.layers[fakeQuantizeLayer->get_friendly_name()] = newFakeQuantizeLayer;
    }

    auto dequantizationValuesCallback = [&](
        std::shared_ptr<ngraph::Node> layer,
        const std::string originalLayerName,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
        if (layer->get_friendly_name() != originalLayerName) {
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
            update(originalLayerName, layer->get_friendly_name(), dequantizations);
        }

        fillDequantization(
            layer,
            dequantizations,
            dequantizationsToConcatenate);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            const std::shared_ptr<ngraph::Node> node = it.second;
            if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(node)) {
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

bool ConcatMultiChannelsTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

// fill dequantizationsToMerge collection for layer with using dequantizationByFakeQuantize
void ConcatMultiChannelsTransformation::fillDequantization(
    std::shared_ptr<ngraph::Node> layer,
    std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
    std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) const {
    std::shared_ptr<ngraph::opset1::FakeQuantize> currentFakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(layer);
    if (currentFakeQuantize) {
        const auto it = dequantizationByFakeQuantize.find(currentFakeQuantize->get_friendly_name());
        if (it == dequantizationByFakeQuantize.end()) {
            THROW_IE_LPT_EXCEPTION(*currentFakeQuantize) << "dequantization scale values are not found";
        }
        const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
        dequantizationsToConcatenate.push_back(broadcastDequantiationConstant(fakeQuantizeDequantization));
    } else {
        fillQuantization(layer, dequantizationByFakeQuantize, dequantizationsToConcatenate);
    }
}

void ConcatMultiChannelsTransformation::fillQuantization(
    const std::shared_ptr<ngraph::Node> layer,
    const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
    std::vector<FakeQuantizeDequantization>& dequantization) const {
    for (size_t i = 0; i < layer->get_input_size(); ++i) {
        std::shared_ptr<ngraph::Node> parent = layer->get_input_node_shared_ptr(i);

        std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
        if (fakeQuantize) {
            const auto it = dequantizationByFakeQuantize.find(fakeQuantize->get_friendly_name());
            if (it == dequantizationByFakeQuantize.end()) {
                THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization scale values are not found";
            }

            const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
            dequantization.push_back(broadcastDequantiationConstant(fakeQuantizeDequantization));
        } else {
            std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(parent);
            if (concat) {
                std::vector<FakeQuantizeDequantization> dequantizationToConcatenate;
                fillQuantization(concat, dequantizationByFakeQuantize, dequantizationToConcatenate);

                // add concatenated dequantization operations to dequantization collection
                dequantization.push_back(getConcatenatedDequantization(concat, dequantizationToConcatenate));
            } else {
                std::shared_ptr<ngraph::opset1::StridedSlice> stridedSlice = ngraph::as_type_ptr<ngraph::opset1::StridedSlice>(parent);
                if (stridedSlice) {
                    std::vector<FakeQuantizeDequantization> dequantizationToPropagate;
                    fillQuantization(stridedSlice, dequantizationByFakeQuantize, dequantizationToPropagate);

                    const size_t sourceOutputIdx = NetworkHelper::getParentOutputIndex(parent, layer);
                    // add folded dequantization operations to dequantization colection
                    dequantization.push_back(getFoldedDequantization(stridedSlice, dequantizationToPropagate[0], sourceOutputIdx));
                } else {
                    fillQuantization(parent, dequantizationByFakeQuantize, dequantization);
                }
            }
        }
    }
}

// broadcast of dequantization constants by channels
FakeQuantizeDequantization ConcatMultiChannelsTransformation::broadcastDequantiationConstant(const FakeQuantizeDequantization& deq) {
    ngraph::Shape targetShape(deq.data.get_shape().size(), 1ul);
    targetShape[1] = deq.data.get_shape()[1];

    FakeQuantizeDequantization result;
    result.data = deq.data;
    result.convert = deq.convert;

    const auto targetShapeConst = std::make_shared<ngraph::opset1::Constant>(
        element::i64, ngraph::Shape{ targetShape.size() },
        targetShape);

    if (deq.subtract) {
        auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
            deq.subtractConstant,
            targetShapeConst,
            ngraph::op::AutoBroadcastType::NUMPY);

        result.subtract = deq.subtract;
        result.subtractConstant = as_type_ptr<ngraph::opset1::Constant>(broadcast);
    }

    if (deq.multiply) {
        auto broadcast = ngraph::pass::low_precision::fold<ngraph::opset1::Broadcast>(
            deq.multiplyConstant,
            targetShapeConst,
            ngraph::op::AutoBroadcastType::NUMPY);

        result.multiply = deq.multiply;
        result.multiplyConstant = as_type_ptr<ngraph::opset1::Constant>(broadcast);
    }

    return result;
}

FakeQuantizeDequantization ConcatMultiChannelsTransformation::getConcatenatedDequantization(
    const std::shared_ptr<ngraph::opset1::Concat> concat,
    const std::vector<FakeQuantizeDequantization>& dequantization) const {
    bool allDequantizationShiftAreZero = true;
    bool allDequantizationMultiplyAreZero = true;
    for (const auto& deq : dequantization) {
        if (deq.subtract != nullptr) {
            allDequantizationShiftAreZero = false;
        }
        if (deq.multiply != nullptr) {
            allDequantizationMultiplyAreZero = false;
        }
    }

    NodeVector convertNodes;
    NodeVector subNodes;
    NodeVector mulNodes;
    //preparing to concatenate dequantization nodes
    for (const auto& deq : dequantization) {
        ngraph::Shape targetShape(deq.data.get_shape().size(), 1ul);
        targetShape[1] = deq.data.get_shape()[1];

        if (deq.convert != nullptr) {
            convertNodes.push_back(deq.convert);
        }
        if (!allDequantizationShiftAreZero) {
            subNodes.push_back(deq.subtract == nullptr ?
                std::make_shared<ngraph::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 0.f })) :
                deq.subtractConstant);
        }
        if (!allDequantizationMultiplyAreZero) {
            mulNodes.push_back(deq.multiply == nullptr ?
                std::make_shared<ngraph::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 1.0f })) :
                deq.multiplyConstant);
        }
    }

    std::shared_ptr<Node> parent = concat;
    std::shared_ptr<DequantizationConvert> convert;
    if (!convertNodes.empty()) {
        convert = as_type_ptr<DequantizationConvert>(dequantization[0].convert->clone_with_new_inputs({ parent }));
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    std::shared_ptr<ngraph::opset1::Constant> subConst;
    if (!subNodes.empty()) {
        subConst = as_type_ptr<ngraph::opset1::Constant>(
            subNodes.size() == 1ul ? subNodes[0] : fold<ngraph::opset1::Concat>(subNodes, 1ul));

        subtract = std::make_shared<DequantizationSubtract>(parent, subConst);
        parent = subtract;
    }

    std::shared_ptr<DequantizationMultiply> multiply;
    std::shared_ptr<ngraph::opset1::Constant> mulConst;
    if (!mulNodes.empty()) {
        mulConst = as_type_ptr<ngraph::opset1::Constant>(
            mulNodes.size() == 1ul ? mulNodes[0] : fold<ngraph::opset1::Concat>(mulNodes, 1ul));

        multiply = std::make_shared<DequantizationMultiply>(parent, mulConst);
    }

    return FakeQuantizeDequantization(concat, convert, subtract, nullptr, subConst, multiply, mulConst);
}

FakeQuantizeDequantization ConcatMultiChannelsTransformation::getFoldedDequantization(
    const std::shared_ptr<ngraph::Node> operation,
    const FakeQuantizeDequantization& dequantization,
    const size_t sourceOutputIdx) {
    OutputVector inputs = operation->input_values();
    OutputVector outputs(operation->get_output_size());

    std::shared_ptr<Node> parent = operation;
    std::shared_ptr<DequantizationConvert> convert;
    if (dequantization.convert) {
        convert = as_type_ptr<DequantizationConvert>(dequantization.convert->clone_with_new_inputs({ parent }));
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    std::shared_ptr<ngraph::opset1::Constant> subConst;
    if (dequantization.subtract) {
        inputs[0] = dequantization.subtractConstant;
        const auto op = operation->clone_with_new_inputs(inputs);

        // constant folding of subtract constant
        op->constant_fold(outputs, inputs);

        subConst = as_type_ptr<ngraph::opset1::Constant>(outputs[sourceOutputIdx].get_node_shared_ptr());
        subtract = std::make_shared<DequantizationSubtract>(parent, subConst);
        parent = subtract;
    }

    std::shared_ptr<DequantizationMultiply> multiply;
    std::shared_ptr<ngraph::opset1::Constant> mulConst;
    if (dequantization.multiply) {
        inputs[0] = dequantization.multiplyConstant;
        const auto op = operation->clone_with_new_inputs(inputs);

        // constant folding of multiply constant
        op->constant_fold(outputs, inputs);

        mulConst = as_type_ptr<ngraph::opset1::Constant>(outputs[sourceOutputIdx].get_node_shared_ptr());
        multiply = std::make_shared<DequantizationMultiply>(parent, mulConst);
    }

    return FakeQuantizeDequantization(operation->output(sourceOutputIdx), convert, subtract, nullptr, subConst, multiply, mulConst);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
