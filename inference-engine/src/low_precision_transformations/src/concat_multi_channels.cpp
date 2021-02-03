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
            updatePrecisions);
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
    std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) {
    std::vector<std::shared_ptr<ngraph::opset1::FakeQuantize>> fakeQuantizes;
    std::shared_ptr<ngraph::opset1::FakeQuantize> currentFakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(layer);
    if (currentFakeQuantize != nullptr) {
        fakeQuantizes.push_back(currentFakeQuantize);
    } else {
        fillQuantization(layer, fakeQuantizes);
        if (fakeQuantizes.size() == layer->get_input_size()) {
            updateDequantizationShapesIfNecessary(layer, fakeQuantizes, dequantizationByFakeQuantize);
        }
    }

    for (const auto& fakeQuantize : fakeQuantizes) {
        const auto it = dequantizationByFakeQuantize.find(fakeQuantize->get_friendly_name());
        if (it == dequantizationByFakeQuantize.end()) {
            THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization scale values are not found";
        }
        const FakeQuantizeDequantization& fakeQuantizeDequantization = it->second;
        dequantizationsToConcatenate.push_back(fakeQuantizeDequantization);
    }
}

void ConcatMultiChannelsTransformation::updateDequantizationShapesIfNecessary(
    std::shared_ptr<ngraph::Node> layer,
    std::vector<std::shared_ptr<ngraph::opset1::FakeQuantize>>& fakeQuantizes,
    std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize) {
    for (size_t i = 0; i < fakeQuantizes.size(); ++i) {
        ngraph::Shape inputShape = layer->get_input_shape(i);
        ngraph::Shape dequantizationShape = fakeQuantizes[i]->get_shape();
        if (inputShape[1] != dequantizationShape[1]) {
            FakeQuantizeDequantization replacedDequantization = dequantizationByFakeQuantize[fakeQuantizes[i]->get_friendly_name()];

            const float scale = as_type_ptr<ngraph::opset1::Constant>(replacedDequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
            const float shift = replacedDequantization.subtract ? replacedDequantization.subtractConstant->cast_vector<float>()[0] : 0.f;
            const auto precisionBefore = replacedDequantization.data.get_element_type();
            const auto precisionAfter = replacedDequantization.multiply->get_element_type();

            auto newDequantization = ngraph::pass::low_precision::NetworkHelper::makeDequantization(
                scale, shift, precisionBefore, inputShape, precisionAfter, 0.f, 5.f);
            dequantizationByFakeQuantize[fakeQuantizes[i]->get_friendly_name()] = newDequantization;
        }
    }
}

void ConcatMultiChannelsTransformation::fillQuantization(
    const std::shared_ptr<ngraph::Node> layer,
    std::vector<std::shared_ptr<ngraph::opset1::FakeQuantize>>& fakeQuantizes) {
    for (size_t i = 0; i < layer->get_input_size(); ++i) {
        std::shared_ptr<ngraph::Node> parent = layer->get_input_node_shared_ptr(i);
        std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
        if (fakeQuantize != nullptr) {
            fakeQuantizes.push_back(fakeQuantize);
        } else {
            fillQuantization(parent, fakeQuantizes);
        }
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
