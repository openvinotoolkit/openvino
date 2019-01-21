// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_detection_output_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::DetectionOutputLayer::DetectionOutputLayer(const std::string& name): LayerFragment("DetectionOutput", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(2);
}

Builder::DetectionOutputLayer::DetectionOutputLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "DetectionOutput"))
        THROW_IE_EXCEPTION << "Cannot create DetectionOutputLayer decorator for layer " << getLayer().getType();
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::DetectionOutputLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for DetectionOutput layer.";
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::DetectionOutputLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::DetectionOutputLayer::getNumClasses() const {
    return getLayer().getParameters()["num_classes"].asUInt();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNumClasses(size_t num) {
    getLayer().getParameters()["num_classes"] = num;
    return *this;
}
int Builder::DetectionOutputLayer::getBackgroudLabelId() const {
    return getLayer().getParameters()["background_label_id"].asInt(-1);
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setBackgroudLabelId(int labelId) {
    getLayer().getParameters()["background_label_id"] = labelId;
    return *this;
}
int Builder::DetectionOutputLayer::getTopK() const {
    return getLayer().getParameters()["top_k"].asInt();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setTopK(int topK) {
    getLayer().getParameters()["top_k"] = topK;
    return *this;
}
int Builder::DetectionOutputLayer::getKeepTopK() const {
    return getLayer().getParameters()["keep_top_k"].asInt();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setKeepTopK(int topK) {
    getLayer().getParameters()["keep_top_k"] = topK;
    return *this;
}
int Builder::DetectionOutputLayer::getNumOrientClasses() const {
    return getLayer().getParameters()["num_orient_classes"].asInt();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNumOrientClasses(int numClasses) {
    getLayer().getParameters()["num_orient_classes"] = numClasses;
    return *this;
}
std::string Builder::DetectionOutputLayer::getCodeType() const {
    return getLayer().getParameters()["code_type"];
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setCodeType(std::string type) {
    getLayer().getParameters()["code_type"] = type;
    return *this;
}
int Builder::DetectionOutputLayer::getInterpolateOrientation() const {
    return getLayer().getParameters()["interpolate_orientation"].asInt();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setInterpolateOrientation(int orient) {
    getLayer().getParameters()["interpolate_orientation"] = orient;
    return *this;
}
float Builder::DetectionOutputLayer::getNMSThreshold() const {
    return getLayer().getParameters()["nms_threshold"].asFloat();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNMSThreshold(float threshold) {
    getLayer().getParameters()["nms_threshold"] = threshold;
    return *this;
}
float Builder::DetectionOutputLayer::getConfidenceThreshold() const {
    return getLayer().getParameters()["confidence_threshold"].asFloat();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setConfidenceThreshold(float threshold) {
    getLayer().getParameters()["confidence_threshold"] = threshold;
    return *this;
}
bool Builder::DetectionOutputLayer::getShareLocation() const {
    return getLayer().getParameters()["share_location"].asBool();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setShareLocation(bool flag) {
    getLayer().getParameters()["share_location"] = flag;
    return *this;
}
bool Builder::DetectionOutputLayer::getVariantEncodedInTarget() const {
    return getLayer().getParameters()["variance_encoded_in_target"].asBool();
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setVariantEncodedInTarget(bool flag) {
    getLayer().getParameters()["variance_encoded_in_target"] = flag;
    return *this;
}
