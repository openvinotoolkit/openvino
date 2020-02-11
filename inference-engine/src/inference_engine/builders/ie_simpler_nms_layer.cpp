// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_simpler_nms_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::SimplerNMSLayer::SimplerNMSLayer(const std::string& name): LayerDecorator("SimplerNMS", name) {
    getLayer()->getOutputPorts().resize(1);
}

Builder::SimplerNMSLayer::SimplerNMSLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("SimplerNMS");
}

Builder::SimplerNMSLayer::SimplerNMSLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("SimplerNMS");
}

Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const std::vector<Port>& Builder::SimplerNMSLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}
const Port& Builder::SimplerNMSLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::SimplerNMSLayer::getPreNMSTopN() const {
    return getLayer()->getParameters().at("pre_nms_topn");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setPreNMSTopN(size_t topN) {
    getLayer()->getParameters()["pre_nms_topn"] = topN;
    return *this;
}
size_t Builder::SimplerNMSLayer::getPostNMSTopN() const {
    return getLayer()->getParameters().at("post_nms_topn");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setPostNMSTopN(size_t topN) {
    getLayer()->getParameters()["post_nms_topn"] = topN;
    return *this;
}
size_t Builder::SimplerNMSLayer::getFeatStride() const {
    return getLayer()->getParameters().at("feat_stride");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setFeatStride(size_t featStride) {
    getLayer()->getParameters()["feat_stride"] = featStride;
    return *this;
}
size_t Builder::SimplerNMSLayer::getMinBoxSize() const {
    return getLayer()->getParameters().at("min_bbox_size");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setMinBoxSize(size_t minSize) {
    getLayer()->getParameters()["min_bbox_size"] = minSize;
    return *this;
}
size_t Builder::SimplerNMSLayer::getScale() const {
    return getLayer()->getParameters().at("scale");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setScale(size_t scale) {
    getLayer()->getParameters()["scale"] = scale;
    return *this;
}

float Builder::SimplerNMSLayer::getCLSThreshold() const {
    return getLayer()->getParameters().at("cls_threshold");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setCLSThreshold(float threshold) {
    getLayer()->getParameters()["cls_threshold"] = threshold;
    return *this;
}
float Builder::SimplerNMSLayer::getIOUThreshold() const {
    return getLayer()->getParameters().at("iou_threshold");
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setIOUThreshold(float threshold) {
    getLayer()->getParameters()["iou_threshold"] = threshold;
    return *this;
}

REG_CONVERTER_FOR(SimplerNMS, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["iou_threshold"] = cnnLayer->GetParamAsFloat("iou_threshold");
    layer.getParameters()["cls_threshold"] = cnnLayer->GetParamAsFloat("cls_threshold");
    layer.getParameters()["scale"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("scale"));
    layer.getParameters()["min_bbox_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("min_bbox_size"));
    layer.getParameters()["feat_stride"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("feat_stride"));
    layer.getParameters()["pre_nms_topn"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("pre_nms_topn"));
    layer.getParameters()["post_nms_topn"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("post_nms_topn"));
});