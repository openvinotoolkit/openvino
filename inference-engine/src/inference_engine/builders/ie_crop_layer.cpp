// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_crop_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::CropLayer::CropLayer(const std::string& name): LayerFragment("Crop", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(2);
}

Builder::CropLayer::CropLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Crop"))
        THROW_IE_EXCEPTION << "Cannot create CropLayer decorator for layer " << getLayer().getType();
}

Builder::CropLayer& Builder::CropLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::CropLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::CropLayer& Builder::CropLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::CropLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::CropLayer& Builder::CropLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::CropLayer::getAxis() const {
    return uInts2size_t(getLayer().getParameters()["axis"].asUInts());
}

Builder::CropLayer& Builder::CropLayer::setAxis(const std::vector<size_t>& axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}

const std::vector<size_t> Builder::CropLayer::getOffset() const {
    return uInts2size_t(getLayer().getParameters()["offset"].asUInts());
}

Builder::CropLayer& Builder::CropLayer::setOffset(const std::vector<size_t>& offsets) {
    getLayer().getParameters()["offset"] = offsets;
    return *this;
}

void Builder::CropLayer::validate(const Layer& layer) {
    if (layer.getInputPorts().size() != 2)
        THROW_IE_EXCEPTION << "Incorrect parameters for layer " << layer.getName() << " should have 2 inputs!";
}

REG_VALIDATOR_FOR(Crop, Builder::CropLayer::validate);