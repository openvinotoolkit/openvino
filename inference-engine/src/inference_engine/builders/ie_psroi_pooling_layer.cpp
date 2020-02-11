// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_psroi_pooling_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PSROIPoolingLayer::PSROIPoolingLayer(const std::string& name): LayerDecorator("PSROIPooling", name) {
    getLayer()->getOutputPorts().resize(1);
}

Builder::PSROIPoolingLayer::PSROIPoolingLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("PSROIPooling");
}

Builder::PSROIPoolingLayer::PSROIPoolingLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("PSROIPooling");
}

Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const std::vector<Port>& Builder::PSROIPoolingLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setInputPorts(const std::vector<Port>& ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "PSROIPoolingLayer should have 2 inputs!";
    getLayer()->getInputPorts() = ports;
    return *this;
}
const Port& Builder::PSROIPoolingLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}
float Builder::PSROIPoolingLayer::getSpatialScale() const {
    return getLayer()->getParameters().at("spatial_scale");
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setSpatialScale(float spatialScale) {
    getLayer()->getParameters()["spatial_scale"] = spatialScale;
    return *this;
}
size_t Builder::PSROIPoolingLayer::getOutputDim() const {
    return getLayer()->getParameters().at("output_dim");
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setOutputDim(size_t outDim) {
    getLayer()->getParameters()["output_dim"] = outDim;
    return *this;
}
size_t Builder::PSROIPoolingLayer::getGroupSize() const {
    return getLayer()->getParameters().at("group_size");
}
Builder::PSROIPoolingLayer& Builder::PSROIPoolingLayer::setGroupSize(size_t size) {
    getLayer()->getParameters()["group_size"] = size;
    return *this;
}

REG_CONVERTER_FOR(PSROIPooling, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["group_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("group_size", 0));
    layer.getParameters()["output_dim"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("output_dim", 0));
    layer.getParameters()["spatial_scale"] = cnnLayer->GetParamAsFloat("spatial_scale", 0);
});
