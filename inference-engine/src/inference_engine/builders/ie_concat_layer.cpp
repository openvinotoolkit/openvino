// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_concat_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ConcatLayer::ConcatLayer(const std::string& name): LayerDecorator("Concat", name) {
    getLayer()->getOutputPorts().resize(1);
    setAxis(1);
}

Builder::ConcatLayer::ConcatLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Concat");
}

Builder::ConcatLayer::ConcatLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Concat");
}

Builder::ConcatLayer& Builder::ConcatLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ConcatLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ConcatLayer& Builder::ConcatLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::ConcatLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::ConcatLayer& Builder::ConcatLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}

size_t Builder::ConcatLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::ConcatLayer& Builder::ConcatLayer::setAxis(size_t axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_VALIDATOR_FOR(Concat, [] (const InferenceEngine::Builder::Layer::CPtr &input_layer, bool partial) {
    if (partial) {
        return;
    }
    Builder::ConcatLayer layer(input_layer);
    if (layer.getInputPorts().size() < 1) {
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input ports. "
                           << "It takes at least two Blobs";
    }
    for (size_t i = 1; i < layer.getInputPorts().size(); ++i) {
        if (layer.getInputPorts()[i - 1].shape().size() != layer.getInputPorts()[i].shape().size()) {
            THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input ports. "
                               << "It should have equal number of dimensions";
        }
    }
    if (layer.getInputPorts()[0].shape().size() != layer.getOutputPort().shape().size()) {
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports "
                           << "It should have equal number of dimensions";
    }
    if (layer.getAxis() >= layer.getOutputPort().shape().size()) {
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << "contains incorrect axis. "
                           << "It should be >= 0 and < number of port's dimensions.";
    }
    for (size_t i = 0; i < layer.getOutputPort().shape().size(); ++i) {
        if (i == layer.getAxis()) {
            size_t sumInputDimensions = 0;
            for (const Port& port : layer.getInputPorts()) {
                sumInputDimensions += port.shape()[i];
            }
            if (sumInputDimensions != layer.getOutputPort().shape()[i]) {
                THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports "
                                   << "Sum of input port's dimensions in the given axis should be equal to output ports dimension in the same axis.";
            }
        } else {
            for (const Port& port : layer.getInputPorts()) {
                if (port.shape()[i] != layer.getOutputPort().shape()[i]) {
                    THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports. "
                                       << "It should have equal dimensions in axis different from given";
                }
            }
        }
    }
});

REG_CONVERTER_FOR(Concat, [] (const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});

