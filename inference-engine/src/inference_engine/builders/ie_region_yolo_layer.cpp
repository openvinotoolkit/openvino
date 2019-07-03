// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_region_yolo_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::RegionYoloLayer::RegionYoloLayer(const std::string& name): LayerDecorator("RegionYolo", name) {
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
}

Builder::RegionYoloLayer::RegionYoloLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("RegionYolo");
}

Builder::RegionYoloLayer::RegionYoloLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("RegionYolo");
}

Builder::RegionYoloLayer& Builder::RegionYoloLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const Port& Builder::RegionYoloLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::RegionYoloLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

int Builder::RegionYoloLayer::getCoords() const {
    return getLayer()->getParameters().at("coords");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setCoords(int coords) {
    getLayer()->getParameters()["coords"] = coords;
    return *this;
}
int Builder::RegionYoloLayer::getClasses() const {
    return getLayer()->getParameters().at("classes");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setClasses(int classes) {
    getLayer()->getParameters()["classes"] = classes;
    return *this;
}
int Builder::RegionYoloLayer::getNum() const {
    return getLayer()->getParameters().at("num");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setNum(int num) {
    getLayer()->getParameters()["num"] = num;
    return *this;
}
bool Builder::RegionYoloLayer::getDoSoftMax() const {
    return getLayer()->getParameters().at("do_softmax");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setDoSoftMax(bool flag) {
    getLayer()->getParameters()["do_softmax"] = flag ? 1 : 0;
    return *this;
}
float Builder::RegionYoloLayer::getAnchors() const {
    return getLayer()->getParameters().at("anchors");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setAnchors(float anchors) {
    getLayer()->getParameters()["anchors"] = anchors;
    return *this;
}
int Builder::RegionYoloLayer::getMask() const {
    return getLayer()->getParameters().at("mask");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setMask(int mask) {
    getLayer()->getParameters()["mask"] = mask;
    return *this;
}
size_t Builder::RegionYoloLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setAxis(size_t axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}
size_t Builder::RegionYoloLayer::getEndAxis() const {
    return getLayer()->getParameters().at("end_axis");
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setEndAxis(size_t axis) {
    getLayer()->getParameters()["end_axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(RegionYoloLayer, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["end_axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("end_axis", 0));
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 0));
    layer.getParameters()["num"] = cnnLayer->GetParamAsInt("num", 0);
    layer.getParameters()["mask"] = cnnLayer->GetParamAsInt("mask", 0);
    layer.getParameters()["coords"] = cnnLayer->GetParamAsInt("coords", 0);
    layer.getParameters()["classes"] = cnnLayer->GetParamAsInt("classes", 0);
    layer.getParameters()["anchors"] = cnnLayer->GetParamAsFloat("anchors", 0);
    layer.getParameters()["do_softmax"] = cnnLayer->GetParamAsBool("do_softmax", false);
});