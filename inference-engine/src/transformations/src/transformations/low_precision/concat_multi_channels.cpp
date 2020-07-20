// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/concat_multi_channels.hpp"

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

namespace ngraph {
namespace pass {
namespace low_precision {

bool isMultiChannel(const std::vector<ngraph::opset1::Concat*>& concatLayers) {
    for (ngraph::opset1::Concat* concat : concatLayers) {
        std::shared_ptr<ngraph::Node> concatPtr = concat->shared_from_this();
        const std::vector<std::shared_ptr<ngraph::Node>> children = NetworkHelper::getChildrenRecursivelyExceptTypes(
            concatPtr,
            { "Pooling", "Resample" });

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

void ConcatMultiChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());

    ngraph::pass::low_precision::Subgraph subgraph(layerTransformationsManager);
    std::unordered_set<std::string> handledLayers;
    if (!subgraph.fillSubgraphForConcat(*concat, handledLayers)) {
        return;
    }

    if (subgraph.quantizationLayers.empty() || isHandled(context, subgraph.quantizationLayers)) {
        return;
    }

    if (!isMultiChannel(subgraph.concatLayers)) {
        ConcatTransformation::transform(context, m);
        return;
    }

    // precisions can be different
    ngraph::Node& quantizationLayer = *subgraph.quantizationLayers[0];
    std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(quantizationLayer.shared_from_this());
    DataPrecision dataPrecision = getDataPrecision(fq, QuantizationDetails::getDetails(fq), false, false);
    if (dataPrecision.precision == ngraph::element::undefined) {
        return;
    }

    // TODO: use raw pointer instead names
    std::unordered_map<std::string, ngraph::pass::low_precision::FakeQuantizeDequantization> dequantizations;

    for (size_t i = 0; i < subgraph.quantizationLayers.size(); ++i) {
        ngraph::Node* fakeQuantizeLayer = subgraph.quantizationLayers[i];
        const ngraph::Shape shape = fakeQuantizeLayer->get_output_shape(0);
        if (shape.size() < 4ul) {
            return;
        }

        std::shared_ptr<ngraph::opset1::FakeQuantize> fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fakeQuantizeLayer->shared_from_this());
        if (fq == nullptr) {
            return;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(fq);

        // 1. get data for dequantization. Dequantization data will be used several times later.
        FakeQuantizeDequantization fakeQuantizeDequantization = ngraph::pass::low_precision::NetworkHelper::createDequantizationFromFakeQuantize(
            fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);
        dequantizations[fakeQuantizeLayer->get_friendly_name()] = fakeQuantizeDequantization;

        // 2. update FakeQuantize - one time action
        std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
            fq,
            updatePrecisions ? dataPrecision.precision : fakeQuantizeLayer->get_output_element_type(0),
            roundf(dataPrecision.min),
            roundf(dataPrecision.max));

        subgraph.quantizationLayers[i] = newFakeQuantizeLayer.get();
        subgraph.layers[fakeQuantizeLayer->get_friendly_name()] = newFakeQuantizeLayer.get();
    }

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

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            ngraph::Node* node = it.second;
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(node->shared_from_this(), dataPrecision.precision);
        }
    }

    for (const ngraph::Node* quantizationLayer : subgraph.quantizationLayers) {
        context.quantizedFakeQuantizeNames.insert(quantizationLayer->get_friendly_name());
    }
}

bool ConcatMultiChannelsTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

// fill dequantizationsToMerge collection for layer with using dequantizationByFakeQuantize
void ConcatMultiChannelsTransformation::fillDequantization(
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

void ConcatMultiChannelsTransformation::fillQuantization(const ngraph::Node& layer, std::vector<ngraph::opset1::FakeQuantize*>& fakeQuantizes) {
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
