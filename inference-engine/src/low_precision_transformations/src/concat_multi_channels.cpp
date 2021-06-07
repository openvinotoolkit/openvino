// Copyright (C) 2018-2021 Intel Corporation
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
            if ((is_type<ngraph::opset1::Convolution>(child.get()) ||
                is_type<ngraph::opset1::ConvolutionBackpropData>(child.get())) &&
                this->layerTransformationsManager->isQuantized(child)) {
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
        std::vector<element::Type> concatChildrenPrecisions = precisionsOnActivations;
        for (auto quantizationLayer : subgraph.quantizationLayers) {
            std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer->shared_from_this());
            if (!NetworkHelper::isQuantizeSupported(fq)) {
                return false;
            }

            // define concatenation operation consumers precisions
            std::vector<element::Type> fqChildrenPrecisions = precisionsOnActivations;
            fillAvailablePrecisions(quantizationLayer, fqChildrenPrecisions);
            concatChildrenPrecisions = NetworkHelper::precisionIntersection(concatChildrenPrecisions, fqChildrenPrecisions);
            if (concatChildrenPrecisions.empty()) {
                return false;
            }

            // define FakeQuantize precisions without zero point
            const DataPrecision tmp = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false);
            if (dataPrecision.precision == ngraph::element::undefined) {
                dataPrecision = tmp;
                continue;
            }

            if ((tmp.precision != dataPrecision.precision) && (tmp.precision == ngraph::element::u8)) {
                dataPrecision = tmp;
            }
        }

        if (std::find(concatChildrenPrecisions.begin(), concatChildrenPrecisions.end(), dataPrecision.precision) == concatChildrenPrecisions.end()) {
            dataPrecision = DataPrecision(concatChildrenPrecisions[0]);
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
        std::shared_ptr<ngraph::Node> child,
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

        if (!is_type<ngraph::opset1::Concat>(layer)) {
            // for intermediate layers we should get Dq operations to be inserted between layer and child
            assert(dequantizationsToConcatenate.size() == 1ul);
            const size_t sourceOutputIdx = NetworkHelper::getParentOutputIndex(layer, child);
            if (layer->get_input_shape(0)[1] != layer->get_output_shape(sourceOutputIdx)[1]) {
                dequantizationsToConcatenate[0] = getFoldedDequantization(layer, dequantizationsToConcatenate[0], sourceOutputIdx);
            }
        }
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

void ConcatMultiChannelsTransformation::fillDequantization(
    const std::shared_ptr<ngraph::Node> layer,
    const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
    std::vector<FakeQuantizeDequantization>& dequantization) const {
    const auto fillDqByFakeQuantize = [&](const std::shared_ptr<ngraph::Node>& fq) {
        const auto it = dequantizationByFakeQuantize.find(fq->get_friendly_name());
        if (it == dequantizationByFakeQuantize.end()) {
            THROW_IE_LPT_EXCEPTION(*fq) << "dequantization scale values are not found";
        }

        const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
        dequantization.push_back(fakeQuantizeDequantization);
    };

    if (is_type<ngraph::opset1::FakeQuantize>(layer)) {
        fillDqByFakeQuantize(layer);
    } else {
        for (size_t i = 0; i < layer->get_input_size(); ++i) {
            std::shared_ptr<ngraph::Node> parent = layer->get_input_node_shared_ptr(i);
            if (as_type_ptr<ngraph::opset1::Constant>(parent)) {
                continue;
            }

            const auto fakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
            if (fakeQuantize) {
                fillDqByFakeQuantize(fakeQuantize);
            } else {
                const auto concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(parent);
                if (concat) {
                    std::vector<FakeQuantizeDequantization> dequantizationToConcatenate;
                    fillDequantization(concat, dequantizationByFakeQuantize, dequantizationToConcatenate);

                    // add concatenated dequantization operations to dequantization collection
                    dequantization.push_back(getConcatenatedDequantization(concat, dequantizationToConcatenate));
                } else {
                    const size_t sourceOutputIdx = NetworkHelper::getParentOutputIndex(parent, layer);
                    if (parent->get_input_shape(0)[1] != parent->get_output_shape(sourceOutputIdx)[1]) {
                        std::vector<FakeQuantizeDequantization> dequantizationToPropagate;
                        fillDequantization(parent, dequantizationByFakeQuantize, dequantizationToPropagate);

                        // add folded dequantization operations to dequantization colection
                        dequantization.push_back(getFoldedDequantization(parent, dequantizationToPropagate[0], sourceOutputIdx));
                    } else {
                        fillDequantization(parent, dequantizationByFakeQuantize, dequantization);
                    }
                }
            }
        }
    }
}

FakeQuantizeDequantization ConcatMultiChannelsTransformation::getConcatenatedDequantization(
    const std::shared_ptr<ngraph::opset1::Concat> concat,
    const std::vector<FakeQuantizeDequantization>& dequantization) const {
    NodeVector convertNodes;
    NodeVector subtractNodes;
    NodeVector multiplyNodes;

    // forming nodes for concatenation
    fillDequantizationNodes(dequantization, concat, convertNodes, subtractNodes, multiplyNodes);

    std::shared_ptr<Node> parent = concat;
    std::shared_ptr<DequantizationConvert> convert;
    if (!convertNodes.empty()) {
        convert = as_type_ptr<DequantizationConvert>(dequantization[0].convert->clone_with_new_inputs({ parent }));
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    std::shared_ptr<ngraph::opset1::Constant> subConst;
    if (!subtractNodes.empty()) {
        subConst = as_type_ptr<ngraph::opset1::Constant>(concatenateDeqNodes(subtractNodes));
        subtract = std::make_shared<DequantizationSubtract>(parent, subConst);
        parent = subtract;
    }

    std::shared_ptr<DequantizationMultiply> multiply;
    std::shared_ptr<ngraph::opset1::Constant> mulConst;
    if (!multiplyNodes.empty()) {
        mulConst = as_type_ptr<ngraph::opset1::Constant>(concatenateDeqNodes(multiplyNodes));
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
    Output<Node> data = operation->output(sourceOutputIdx);

    std::shared_ptr<Node> parent = operation;
    std::shared_ptr<DequantizationConvert> convert;
    if (dequantization.convert) {
        convert = as_type_ptr<DequantizationConvert>(dequantization.convert->clone_with_new_inputs({ data }));
        parent = convert;
    }

    std::shared_ptr<DequantizationSubtract> subtract;
    std::shared_ptr<ngraph::opset1::Constant> subConst;
    if (dequantization.subtract) {
        subConst = NetworkHelper::foldDequantizationConstant(dequantization.subtractConstant, operation, sourceOutputIdx);
        subtract = std::make_shared<DequantizationSubtract>(parent, subConst);
        parent = subtract;
    }

    std::shared_ptr<DequantizationMultiply> multiply;
    std::shared_ptr<ngraph::opset1::Constant> mulConst;
    if (dequantization.multiply) {
        mulConst = NetworkHelper::foldDequantizationConstant(dequantization.multiplyConstant, operation, sourceOutputIdx);
        multiply = std::make_shared<DequantizationMultiply>(parent, mulConst);
    }

    return FakeQuantizeDequantization(data, convert, subtract, nullptr, subConst, multiply, mulConst);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
