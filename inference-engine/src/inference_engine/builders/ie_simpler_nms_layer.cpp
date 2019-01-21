// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_simpler_nms_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::SimplerNMSLayer::SimplerNMSLayer(const std::string& name): LayerFragment("SimplerNMS", name) {
    getLayer().getOutputPorts().resize(1);
}

Builder::SimplerNMSLayer::SimplerNMSLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "SimplerNMS"))
        THROW_IE_EXCEPTION << "Cannot create SimplerNMSLayer decorator for layer " << getLayer().getType();
}

Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const std::vector<Port>& Builder::SimplerNMSLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer().getInputPorts() = ports;
    return *this;
}
const Port& Builder::SimplerNMSLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::SimplerNMSLayer::getPreNMSTopN() const {
    return getLayer().getParameters()["pre_nms_topn"].asUInt();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setPreNMSTopN(size_t topN) {
    getLayer().getParameters()["pre_nms_topn"] = topN;
    return *this;
}
size_t Builder::SimplerNMSLayer::getPostNMSTopN() const {
    return getLayer().getParameters()["post_nms_topn"].asUInt();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setPostNMSTopN(size_t topN) {
    getLayer().getParameters()["post_nms_topn"] = topN;
    return *this;
}
size_t Builder::SimplerNMSLayer::getFeatStride() const {
    return getLayer().getParameters()["feat_stride"].asUInt();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setFeatStride(size_t featStride) {
    getLayer().getParameters()["feat_stride"] = featStride;
    return *this;
}
size_t Builder::SimplerNMSLayer::getMinBoxSize() const {
    return getLayer().getParameters()["min_bbox_size"].asUInt();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setMinBoxSize(size_t minSize) {
    getLayer().getParameters()["min_bbox_size"] = minSize;
    return *this;
}
size_t Builder::SimplerNMSLayer::getScale() const {
    return getLayer().getParameters()["scale"].asUInt();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setScale(size_t scale) {
    getLayer().getParameters()["scale"] = scale;
    return *this;
}

float Builder::SimplerNMSLayer::getCLSThreshold() const {
    return getLayer().getParameters()["cls_threshold"].asFloat();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setCLSThreshold(float threshold) {
    getLayer().getParameters()["cls_threshold"] = threshold;
    return *this;
}
float Builder::SimplerNMSLayer::getIOUThreshold() const {
    return getLayer().getParameters()["iou_threshold"].asFloat();
}
Builder::SimplerNMSLayer& Builder::SimplerNMSLayer::setIOUThreshold(float threshold) {
    getLayer().getParameters()["iou_threshold"] = threshold;
    return *this;
}
