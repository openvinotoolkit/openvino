// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_region_yolo_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::RegionYoloLayer::RegionYoloLayer(const std::string& name): LayerFragment("RegionYolo", name) {
    getLayer().getInputPorts().resize(1);
    getLayer().getOutputPorts().resize(1);
}

Builder::RegionYoloLayer::RegionYoloLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "RegionYolo"))
        THROW_IE_EXCEPTION << "Cannot create RegionYoloLayer decorator for layer " << getLayer().getType();
}

Builder::RegionYoloLayer& Builder::RegionYoloLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const Port& Builder::RegionYoloLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setInputPort(const Port& port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::RegionYoloLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

int Builder::RegionYoloLayer::getCoords() const {
    return getLayer().getParameters()["coords"].asInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setCoords(int coords) {
    getLayer().getParameters()["coords"] = coords;
    return *this;
}
int Builder::RegionYoloLayer::getClasses() const {
    return getLayer().getParameters()["classes"].asInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setClasses(int classes) {
    getLayer().getParameters()["classes"] = classes;
    return *this;
}
int Builder::RegionYoloLayer::getNum() const {
    return getLayer().getParameters()["num"].asInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setNum(int num) {
    getLayer().getParameters()["num"] = num;
    return *this;
}
bool Builder::RegionYoloLayer::getDoSoftMax() const {
    return getLayer().getParameters()["do_softmax"].asBool();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setDoSoftMax(bool flag) {
    getLayer().getParameters()["do_softmax"] = flag ? 1 : 0;
    return *this;
}
float Builder::RegionYoloLayer::getAnchors() const {
    return getLayer().getParameters()["anchors"].asFloat();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setAnchors(float anchors) {
    getLayer().getParameters()["anchors"] = anchors;
    return *this;
}
int Builder::RegionYoloLayer::getMask() const {
    return getLayer().getParameters()["mask"].asInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setMask(int mask) {
    getLayer().getParameters()["mask"] = mask;
    return *this;
}
size_t Builder::RegionYoloLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asUInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setAxis(size_t axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}
size_t Builder::RegionYoloLayer::getEndAxis() const {
    return getLayer().getParameters()["end_axis"].asUInt();
}
Builder::RegionYoloLayer& Builder::RegionYoloLayer::setEndAxis(size_t axis) {
    getLayer().getParameters()["end_axis"] = axis;
    return *this;
}
