// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_convolution_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>
#include <limits>

using namespace InferenceEngine;

Builder::ConvolutionLayer::ConvolutionLayer(const std::string& name): LayerDecorator("Convolution", name) {
    getLayer()->getInputPorts().resize(3);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getOutputPorts().resize(1);
    setGroup(1);
    setKernel({});
    setOutDepth(0);
    setStrides({});
    setDilation({});
    setPaddingsEnd({});
    setPaddingsBegin({});
}

Builder::ConvolutionLayer::ConvolutionLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Convolution");
}

Builder::ConvolutionLayer::ConvolutionLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Convolution");
}

Builder::ConvolutionLayer &Builder::ConvolutionLayer::setName(const std::string &name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ConvolutionLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::ConvolutionLayer& Builder::ConvolutionLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::ConvolutionLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ConvolutionLayer& Builder::ConvolutionLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getKernel() const {
    return getLayer()->getParameters().at("kernel");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setKernel(const std::vector<size_t>& kernel) {
    getLayer()->getParameters()["kernel"] = kernel;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getStrides() const {
    return getLayer()->getParameters().at("strides");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setStrides(const std::vector<size_t>& strides) {
    getLayer()->getParameters()["strides"] = strides;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getDilation() const {
    return getLayer()->getParameters().at("dilations");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setDilation(const std::vector<size_t>& dilation) {
    getLayer()->getParameters()["dilations"] = dilation;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getPaddingsBegin() const {
    return getLayer()->getParameters().at("pads_begin");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setPaddingsBegin(const std::vector<size_t>& paddings) {
    getLayer()->getParameters()["pads_begin"] = paddings;
    return *this;
}

const std::vector<size_t> Builder::ConvolutionLayer::getPaddingsEnd() const {
    return getLayer()->getParameters().at("pads_end");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setPaddingsEnd(const std::vector<size_t>& paddings) {
    getLayer()->getParameters()["pads_end"] = paddings;
    return *this;
}

size_t Builder::ConvolutionLayer::getGroup() const {
    return getLayer()->getParameters().at("group");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setGroup(size_t group) {
    getLayer()->getParameters()["group"] = group;
    return *this;
}

size_t Builder::ConvolutionLayer::getOutDepth() const {
    return getLayer()->getParameters().at("output");
}
Builder::ConvolutionLayer& Builder::ConvolutionLayer::setOutDepth(size_t outDepth) {
    getLayer()->getParameters()["output"] = outDepth;
    return *this;
}

REG_VALIDATOR_FOR(Convolution, [] (const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    // WA for old IRs
    if (layer->getParameters().find("kernel") == layer->getParameters().end() &&
        layer->getParameters().find("kernel-x") != layer->getParameters().end() &&
        layer->getParameters().find("kernel-y") != layer->getParameters().end())
        return;

    Builder::ConvolutionLayer convBuilder(layer);
    std::vector<size_t> l_kernel = convBuilder.getKernel();
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

    if (l_kernel.empty()) {
        THROW_IE_EXCEPTION << "Kernel is empty!";
    }

    if (l_paddingBegin.size() != l_paddingEnd.size()) {
        THROW_IE_EXCEPTION << "Padding_begin dimension is not equal to padding_end dimension";
    }

    if (!l_paddingBegin.empty() && l_kernel.size() != l_paddingBegin.size()) {
        THROW_IE_EXCEPTION << "Padding dimension is not equal to kernel dimension";
    }

    if (l_kernel.size() != l_strides.size()) {
        THROW_IE_EXCEPTION << "Stride dimension is not equal to kernel dimension";
    }

    if (!l_dilation.empty() && l_kernel.size() != l_dilation.size()) {
        THROW_IE_EXCEPTION << "Dilation dimension is not equal to kernel dimension";
    }

    if (convBuilder.getOutDepth() == 0) {
        THROW_IE_EXCEPTION << "OutDepth parameter should be more than 0";
    }

    for (size_t kernel_dim : l_kernel) {
        if (kernel_dim == 0) {
            THROW_IE_EXCEPTION << "Kernel dimensions should be more than 0";
        }
    }

    for (size_t i_stride : l_strides) {
        if (i_stride == 0) {
            THROW_IE_EXCEPTION << "Strides should be more than 0";
        }
    }

    for (size_t dil : l_dilation) {
        if (dil == 0)
            THROW_IE_EXCEPTION << "Dilation should be more than 0";
    }

    if (!convBuilder.getGroup())
        THROW_IE_EXCEPTION << "Group should be more than 0";

    if (convBuilder.getInputPort().shape().empty())
        return;

    const size_t IC = convBuilder.getInputPort().shape()[1];
    if (IC % convBuilder.getGroup())
        THROW_IE_EXCEPTION << "Number of input channels (" << IC <<
                           ") is not divided by group number (" << convBuilder.getGroup() << ")";

    size_t weight_size = convBuilder.getOutDepth() * IC / convBuilder.getGroup();
    for (size_t kernel_dim : l_kernel) {
        if (static_cast<double>(weight_size) * kernel_dim > std::numeric_limits<size_t>::max()) {
            THROW_IE_EXCEPTION << "Weight size exceeds the size_t max";
        }
        weight_size *= kernel_dim;
    }

    if (partial)
        return;

    const auto weights = layer->getInputPorts()[1].getData()->getData();
    if (weights->size() != weight_size) {
        THROW_IE_EXCEPTION << "Weight size is not correct!";
    }

    const auto biases = layer->getInputPorts()[2].getData()->getData();
    if (biases && biases->cbuffer() && biases->size() != convBuilder.getOutDepth())
        THROW_IE_EXCEPTION << "Biases size is incorrect!";
});

REG_CONVERTER_FOR(Convolution, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    // WA for old IRs
    if (cnnLayer->params.find("kernel") == cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-x") != cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-y") != cnnLayer->params.end())
        return;

    std::vector<unsigned int> tmp = cnnLayer->GetParamAsUInts("kernel");
    std::vector<size_t> cur(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["kernel"] = cur;

    tmp = cnnLayer->GetParamAsUInts("strides");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["strides"] = cur;

    tmp = cnnLayer->GetParamAsUInts("dilations");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["dilations"] = cur;

    tmp = cnnLayer->GetParamAsUInts("pads_begin");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["pads_begin"] = cur;

    tmp = cnnLayer->GetParamAsUInts("pads_end");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["pads_end"] = cur;

    layer.getParameters()["group"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("group"));
    layer.getParameters()["output"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("output"));
});
