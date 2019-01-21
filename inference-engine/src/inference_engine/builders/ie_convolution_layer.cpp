// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_convolution_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ConvolutionLayer::ConvolutionLayer(const std::string& name): LayerFragment("Convolution", name) {
    getLayer().getInputPorts().resize(1);
    getLayer().getOutputPorts().resize(1);
}

Builder::ConvolutionLayer::ConvolutionLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Convolution"))
        THROW_IE_EXCEPTION << "Cannot create ConvolutionLayer decorator for layer " << getLayer().getType();
}

Builder::ConvolutionLayer::operator Builder::Layer() const {
    Layer genLayer(getLayer());

    std::vector<size_t> l_kernel = getKernel();
    std::vector<size_t> l_dilation = getDilation();
    std::vector<size_t> l_paddingBegin = getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = getPaddingsEnd();
    std::vector<size_t> l_strides = getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty())
        l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty())
        l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_dilation.empty() && !l_kernel.empty())
        l_dilation.resize(l_kernel.size(), 1);
    if (l_strides.empty() && !l_kernel.empty())
        l_strides.resize(l_kernel.size(), 1);

    if (!getOutDepth() || l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() ||
            l_kernel.size() != l_dilation.size() || l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << genLayer.getType() << " node " << genLayer.getName() << " contains incorrect parameters!";

    genLayer.getParameters()["kernel"] = l_kernel;
    genLayer.getParameters()["strides"] = l_strides;
    genLayer.getParameters()["pads_begin"] = l_paddingBegin;
    genLayer.getParameters()["pads_end"] = l_paddingEnd;
    genLayer.getParameters()["dilations"] = l_dilation;
    return genLayer;
}

Builder::ConvolutionLayer &Builder::ConvolutionLayer::setName(const std::string &name) {
    getLayer().getName() = name;
    return *this;
}

Builder::ConvolutionLayer& Builder::ConvolutionLayer::setWeights(const Blob::CPtr& weights) {
    getLayer().addConstantData("weights", weights);
    return *this;
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setBiases(const Blob::CPtr& biases) {
    getLayer().addConstantData("biases", biases);
    return *this;
}

const Port& Builder::ConvolutionLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::ConvolutionLayer& Builder::ConvolutionLayer::setInputPort(const Port& port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::ConvolutionLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ConvolutionLayer& Builder::ConvolutionLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getKernel() const {
    return uInts2size_t(getLayer().getParameters()["kernel"].asUInts({}));
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setKernel(const std::vector<size_t>& kernel) {
    getLayer().getParameters()["kernel"] = kernel;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getStrides() const {
    return uInts2size_t(getLayer().getParameters()["strides"].asUInts({}));
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setStrides(const std::vector<size_t>& strides) {
    getLayer().getParameters()["strides"] = strides;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getDilation() const {
    return uInts2size_t(getLayer().getParameters()["dilations"].asUInts({}));
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setDilation(const std::vector<size_t>& dilation) {
    getLayer().getParameters()["dilations"] = dilation;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getPaddingsBegin() const {
    return uInts2size_t(getLayer().getParameters()["pads_begin"].asUInts({}));
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setPaddingsBegin(const std::vector<size_t>& paddings) {
    getLayer().getParameters()["pads_begin"] = paddings;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getPaddingsEnd() const {
    return uInts2size_t(getLayer().getParameters()["pads_end"].asUInts({}));
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setPaddingsEnd(const std::vector<size_t>& paddings) {
    getLayer().getParameters()["pads_end"] = paddings;
    return *this;
}

size_t Builder::ConvolutionLayer::getGroup() const {
    return getLayer().getParameters()["group"].asUInt(1);
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setGroup(size_t group) {
    getLayer().getParameters()["group"] = group;
    return *this;
}

size_t Builder::ConvolutionLayer::getOutDepth() const {
    return getLayer().getParameters()["output"].asUInt(0);
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setOutDepth(size_t outDepth) {
    getLayer().getParameters()["output"] = outDepth;
    return *this;
}

void Builder::ConvolutionLayer::validate(const Layer& layer) {
    Layer convLayer = layer;
    Builder::ConvolutionLayer convBuilder(convLayer);
    std::vector<size_t> l_kernel = convBuilder.getKernel();

    // WA for old IRs
    if (l_kernel.empty() && layer.getParameters().find("kernel-x") != layer.getParameters().end() &&
            layer.getParameters().find("kernel-y") != layer.getParameters().end())
        return;

    std::vector<size_t> l_dilation = convBuilder.getDilation();
    std::vector<size_t> l_paddingBegin = convBuilder.getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = convBuilder.getPaddingsEnd();
    std::vector<size_t> l_strides = convBuilder.getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty())
        l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty())
        l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_dilation.empty() && !l_kernel.empty())
        l_dilation.resize(l_kernel.size(), 1);
    if (l_strides.empty() && !l_kernel.empty())
        l_strides.resize(l_kernel.size(), 1);

    if (!convBuilder.getOutDepth() || l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() ||
            l_kernel.size() != l_dilation.size() || l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << layer.getType() << " node " << layer.getName() << " contains incorrect parameters!";
}

REG_VALIDATOR_FOR(Convolution, Builder::ConvolutionLayer::validate);
