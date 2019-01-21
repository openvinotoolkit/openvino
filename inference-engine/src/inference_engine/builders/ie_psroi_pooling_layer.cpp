// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_psroi_pooling_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PSROIPoolingLayer::PSROIPoolingLayer(const std::string& name): LayerFragment("PSROIPooling", name) {
    getLayer().getOutputPorts().resize(1);
}

Builder::PSROIPoolingLayer::PSROIPoolingLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "PSROIPooling"))
        THROW_IE_EXCEPTION << "Cannot create PSROIPoolingLayer decorator for layer " << getLayer().getType();
}

Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const std::vector<Port>& Builder::PSROIPoolingLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setInputPorts(const std::vector<Port>& ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "PSROIPoolingLayer should have 2 inputs!";
    getLayer().getInputPorts() = ports;
    return *this;
}
const Port& Builder::PSROIPoolingLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}
float Builder::PSROIPoolingLayer::getSpatialScale() const {
    return getLayer().getParameters()["spatial_scale"].asFloat();
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setSpatialScale(float spatialScale) {
    getLayer().getParameters()["spatial_scale"] = spatialScale;
    return *this;
}
size_t Builder::PSROIPoolingLayer::getOutputDim() const {
    return getLayer().getParameters()["output_dim"].asUInt();
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setOutputDim(size_t outDim) {
    getLayer().getParameters()["output_dim"] = outDim;
    return *this;
}
size_t Builder::PSROIPoolingLayer::getGroupSize() const {
    return getLayer().getParameters()["group_size"].asUInt();
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setGroupSize(size_t size) {
    getLayer().getParameters()["group_size"] = size;
    return *this;
}
