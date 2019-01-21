// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_roi_pooling_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ROIPoolingLayer::ROIPoolingLayer(const std::string& name): LayerFragment("ROIPooling", name) {
    getLayer().getOutputPorts().resize(1);
    setPooled({0, 0});
}

Builder::ROIPoolingLayer::ROIPoolingLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ROIPooling"))
        THROW_IE_EXCEPTION << "Cannot create ROIPoolingLayer decorator for layer " << getLayer().getType();
}

Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const std::vector<Port>& Builder::ROIPoolingLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setInputPorts(const std::vector<Port>& ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "ROIPoolingLayer should have 2 inputs!";
    getLayer().getInputPorts() = ports;
    return *this;
}
const Port& Builder::ROIPoolingLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}
float Builder::ROIPoolingLayer::getSpatialScale() const {
    return getLayer().getParameters()["spatial_scale"].asFloat();
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setSpatialScale(float spatialScale) {
    getLayer().getParameters()["spatial_scale"] = spatialScale;
    return *this;
}
const std::vector<int> Builder::ROIPoolingLayer::getPooled() const {
    return {getLayer().getParameters()["pooled_h"].asInt(0), getLayer().getParameters()["pooled_w"].asInt(0)};
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setPooled(const std::vector<int>& pooled) {
    if (pooled.size() != 2)
        THROW_IE_EXCEPTION << "ROIPoolingLayer supports only pooled for height and width dimensions";
    getLayer().getParameters()["pooled_h"] = pooled[0];
    getLayer().getParameters()["pooled_w"] = pooled[1];
    return *this;
}
