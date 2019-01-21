// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_layer_fragment.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;
using namespace details;

Builder::LayerFragment::LayerFragment(const std::string& type, const std::string& name): layer(type, name), refLayer(layer) {}

Builder::LayerFragment::LayerFragment(Layer& genLayer): layer("", ""), refLayer(genLayer) {}

Builder::LayerFragment &Builder::LayerFragment::operator=(const Builder::LayerFragment &rval) {
    layer = rval.layer;
    refLayer = rval.refLayer;
    if (!layer.getType().empty() && !layer.getName().empty())
        refLayer = layer;
    return *this;
}

Builder::LayerFragment::LayerFragment(const Builder::LayerFragment & rval): LayerFragment("", "") {
    *this = rval;
}

Builder::LayerFragment::operator Builder::Layer() const {
    getLayer().validate();
    return getLayer();
}

const std::string& Builder::LayerFragment::getType() const {
    return getLayer().getType();
}
const std::string& Builder::LayerFragment::getName() const {
    return getLayer().getName();
}

Builder::Layer& Builder::LayerFragment::getLayer() const {
    return refLayer;
}

const std::vector<size_t> Builder::LayerFragment::uInts2size_t(const std::vector<unsigned int>& vector) const {
    std::vector<size_t> newVector;
    newVector.reserve(vector.size());
    for (const auto& it : vector) {
        newVector.push_back(it);
    }
    return newVector;
}
