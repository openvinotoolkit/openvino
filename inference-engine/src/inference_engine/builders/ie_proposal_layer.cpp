// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_proposal_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ProposalLayer::ProposalLayer(const std::string& name): LayerFragment("Proposal", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(3);
}

Builder::ProposalLayer::ProposalLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Proposal"))
        THROW_IE_EXCEPTION << "Cannot create ProposalLayer decorator for layer " << getLayer().getType();
}

Builder::ProposalLayer& Builder::ProposalLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::ProposalLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::ProposalLayer& Builder::ProposalLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for Proposal layer.";
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::ProposalLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ProposalLayer& Builder::ProposalLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::ProposalLayer::getPostNMSTopN() const {
    return getLayer().getParameters()["post_nms_topn"].asUInt();
}
Builder::ProposalLayer& Builder::ProposalLayer::setPostNMSTopN(size_t topN) {
    getLayer().getParameters()["post_nms_topn"] = topN;
    return *this;
}
size_t Builder::ProposalLayer::getPreNMSTopN() const {
    return getLayer().getParameters()["pre_nms_topn"].asUInt();
}
Builder::ProposalLayer& Builder::ProposalLayer::setPreNMSTopN(size_t topN) {
    getLayer().getParameters()["pre_nms_topn"] = topN;
    return *this;
}
float Builder::ProposalLayer::getNMSThresh() const {
    return getLayer().getParameters()["nms_thresh"].asFloat();
}
Builder::ProposalLayer& Builder::ProposalLayer::setNMSThresh(float thresh) {
    getLayer().getParameters()["nms_thresh"] = thresh;
    return *this;
}
size_t Builder::ProposalLayer::getBaseSize() const {
    return getLayer().getParameters()["base_size"].asUInt();
}
Builder::ProposalLayer& Builder::ProposalLayer::setBaseSize(size_t baseSize) {
    getLayer().getParameters()["base_size"] = baseSize;
    return *this;
}
size_t Builder::ProposalLayer::getMinSize() const {
    return getLayer().getParameters()["min_size"].asUInt();
}
Builder::ProposalLayer& Builder::ProposalLayer::setMinSize(size_t minSize) {
    getLayer().getParameters()["min_size"] = minSize;
    return *this;
}
size_t Builder::ProposalLayer::getFeatStride() const {
    return getLayer().getParameters()["feat_stride"].asUInt();
}
Builder::ProposalLayer& Builder::ProposalLayer::setFeatStride(size_t featStride) {
    getLayer().getParameters()["feat_stride"] = featStride;
    return *this;
}
const std::vector<float> Builder::ProposalLayer::getScale() const {
    return getLayer().getParameters()["scale"].asFloats();
}
Builder::ProposalLayer& Builder::ProposalLayer::setScale(const std::vector<float>& scales) {
    getLayer().getParameters()["scale"] = scales;
    return *this;
}
const std::vector<float> Builder::ProposalLayer::getRatio() const {
    return getLayer().getParameters()["ratio"].asFloats();
}
Builder::ProposalLayer& Builder::ProposalLayer::setRatio(const std::vector<float>& ratios) {
    getLayer().getParameters()["ratio"] = ratios;
    return *this;
}
