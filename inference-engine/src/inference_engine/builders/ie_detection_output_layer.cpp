// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_detection_output_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <cfloat>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::DetectionOutputLayer::DetectionOutputLayer(const std::string& name): LayerDecorator("DetectionOutput", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(2);
    setBackgroudLabelId(-1);
}

Builder::DetectionOutputLayer::DetectionOutputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("DetectionOutput");
}

Builder::DetectionOutputLayer::DetectionOutputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("DetectionOutput");
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::DetectionOutputLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for DetectionOutput getLayer().";
    getLayer()->getInputPorts() = ports;
    return *this;
}

const Port& Builder::DetectionOutputLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::DetectionOutputLayer::getNumClasses() const {
    return getLayer()->getParameters().at("num_classes");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNumClasses(size_t num) {
    getLayer()->getParameters()["num_classes"] = num;
    return *this;
}
int Builder::DetectionOutputLayer::getBackgroudLabelId() const {
    return getLayer()->getParameters().at("background_label_id");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setBackgroudLabelId(int labelId) {
    getLayer()->getParameters()["background_label_id"] = labelId;
    return *this;
}
int Builder::DetectionOutputLayer::getTopK() const {
    return getLayer()->getParameters().at("top_k");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setTopK(int topK) {
    getLayer()->getParameters()["top_k"] = topK;
    return *this;
}
int Builder::DetectionOutputLayer::getKeepTopK() const {
    return getLayer()->getParameters().at("keep_top_k");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setKeepTopK(int topK) {
    getLayer()->getParameters()["keep_top_k"] = topK;
    return *this;
}
int Builder::DetectionOutputLayer::getNumOrientClasses() const {
    return getLayer()->getParameters().at("num_orient_classes");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNumOrientClasses(int numClasses) {
    getLayer()->getParameters()["num_orient_classes"] = numClasses;
    return *this;
}
std::string Builder::DetectionOutputLayer::getCodeType() const {
    return getLayer()->getParameters().at("code_type");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setCodeType(std::string type) {
    getLayer()->getParameters()["code_type"] = type;
    return *this;
}
int Builder::DetectionOutputLayer::getInterpolateOrientation() const {
    return getLayer()->getParameters().at("interpolate_orientation");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setInterpolateOrientation(int orient) {
    getLayer()->getParameters()["interpolate_orientation"] = orient;
    return *this;
}
float Builder::DetectionOutputLayer::getNMSThreshold() const {
    return getLayer()->getParameters().at("nms_threshold");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setNMSThreshold(float threshold) {
    getLayer()->getParameters()["nms_threshold"] = threshold;
    return *this;
}
float Builder::DetectionOutputLayer::getConfidenceThreshold() const {
    return getLayer()->getParameters().at("confidence_threshold");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setConfidenceThreshold(float threshold) {
    getLayer()->getParameters()["confidence_threshold"] = threshold;
    return *this;
}
bool Builder::DetectionOutputLayer::getShareLocation() const {
    return getLayer()->getParameters().at("share_location");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setShareLocation(bool flag) {
    getLayer()->getParameters()["share_location"] = flag;
    return *this;
}
bool Builder::DetectionOutputLayer::getVariantEncodedInTarget() const {
    return getLayer()->getParameters().at("variance_encoded_in_target");
}
Builder::DetectionOutputLayer& Builder::DetectionOutputLayer::setVariantEncodedInTarget(bool flag) {
    getLayer()->getParameters()["variance_encoded_in_target"] = flag;
    return *this;
}

REG_VALIDATOR_FOR(DetectionOutput, [](const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::DetectionOutputLayer layer(input_layer);
    if (layer.getNumClasses() == 0) {
        THROW_IE_EXCEPTION << "NumClasses parameter is wrong in layer " << layer.getName() <<
                           ". It should be > 0.";
    }
    if (layer.getCodeType() != "caffe.PriorBoxParameter.CENTER_SIZE" &&
        layer.getCodeType() != "caffe.PriorBoxParameter.CORNER") {
        THROW_IE_EXCEPTION << "CodeType parameter is wrong in layer " << layer.getName() <<
                           ". It should be equal to 'caffe.PriorBoxParameter.CORNER' or 'caffe.PriorBoxParameter.CENTER_SIZE'";
    }
    if (layer.getBackgroudLabelId() < -1) {
        THROW_IE_EXCEPTION << "BackgroundLabelId parameter is wrong in layer " << layer.getName() <<
                           ". It should be >= 0 if this one is an Id of existing label else it should be equal to -1";
    }
    if (layer.getNMSThreshold() < 0) {
        THROW_IE_EXCEPTION << "NMSThreshold parameter is wrong in layer " << layer.getName() <<
                           ". It should be >= 0.";
    }
    if (layer.getConfidenceThreshold() < 0) {
        THROW_IE_EXCEPTION << "ConfidenceThreshold parameter is wrong in layer " << layer.getName() <<
                           ". It should be >= 0.";
    }
});

REG_CONVERTER_FOR(DetectionOutput, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["num_classes"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("num_classes"));
    layer.getParameters()["background_label_id"] = cnnLayer->GetParamAsInt("background_label_id", 0);
    layer.getParameters()["top_k"] = cnnLayer->GetParamAsInt("top_k", -1);
    layer.getParameters()["keep_top_k"] = cnnLayer->GetParamAsInt("keep_top_k", -1);
    layer.getParameters()["num_orient_classes"] = cnnLayer->GetParamAsInt("num_orient_classes", 0);
    layer.getParameters()["code_type"] = cnnLayer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CORNER");
    layer.getParameters()["interpolate_orientation"] = cnnLayer->GetParamAsInt("interpolate_orientation", 1);
    layer.getParameters()["nms_threshold"] = cnnLayer->GetParamAsFloat("nms_threshold");
    layer.getParameters()["confidence_threshold"] = cnnLayer->GetParamAsFloat("confidence_threshold", -FLT_MAX);
    layer.getParameters()["share_location"] = cnnLayer->GetParamAsBool("share_location", true);
    layer.getParameters()["variance_encoded_in_target"] = cnnLayer->GetParamAsBool("variance_encoded_in_target", false);
});
