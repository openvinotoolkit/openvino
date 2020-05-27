// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "ie_layers.h"
#include "cpp/ie_cnn_network.h"
#include <ngraph/ngraph.hpp>

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {

Precision CNNNetwork::getPrecision() const {
    if (actual == nullptr) THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
    return actual->getPrecision();
}

CNNLayerPtr CNNNetwork::getLayerByName(const char* layerName) const {
    CNNLayerPtr layer;
    CALL_STATUS_FNC(getLayerByName, layerName, layer);
    return layer;
}

details::CNNNetworkIterator CNNNetwork::begin() const {
    return details::CNNNetworkIterator(actual);
}

details::CNNNetworkIterator CNNNetwork::end() const {
    return details::CNNNetworkIterator();
}

size_t CNNNetwork::size() const {
    return std::distance(std::begin(*this), std::end(*this));
}

StatusCode ICNNNetwork::AddExtension(const IShapeInferExtensionPtr& extension, ResponseDesc* resp) noexcept {
    (void)extension;
    (void)resp;
    return NOT_IMPLEMENTED;
};

void CNNNetwork::AddExtension(InferenceEngine::IShapeInferExtensionPtr extension) {
    CALL_STATUS_FNC(AddExtension, extension);
}

CNNLayer::CNNLayer(const LayerParams& prms)
    : node(nullptr), name(prms.name), type(prms.type), precision(prms.precision), userValue({0}) {}

CNNLayer::CNNLayer(const CNNLayer& other)
    : node(other.node), name(other.name), type(other.type), precision(other.precision),
    outData(other.outData), insData(other.insData), _fusedWith(other._fusedWith),
    userValue(other.userValue), affinity(other.affinity),
    params(other.params), blobs(other.blobs) {}

LayerParams::LayerParams() {}

LayerParams::LayerParams(const std::string & name, const std::string & type, Precision precision)
    : name(name), type(type), precision(precision) {}

LayerParams::LayerParams(const LayerParams & other)
    : name(other.name), type(other.type), precision(other.precision) {}

LayerParams & LayerParams::operator= (const LayerParams & other) {
    if (&other != this) {
        name = other.name;
        type = other.type;
        precision = other.precision;
    }
    return *this;
}

WeightableLayer::WeightableLayer(const LayerParams& prms) : CNNLayer(prms) {}

}  // namespace InferenceEngine
