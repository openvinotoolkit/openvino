// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_rnn_sequence_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::RNNSequenceLayer::RNNSequenceLayer(const std::string& name): LayerDecorator("RNNSequence", name) {
    getLayer()->getOutputPorts().resize(2);
    getLayer()->getInputPorts().resize(5);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getInputPorts()[3].setParameter("type", "optional");
}

Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("RNNSequence");
}

Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("RNNSequence");
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::RNNSequenceLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}

const std::vector<Port>& Builder::RNNSequenceLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {
    getLayer()->getOutputPorts() = ports;
    return *this;
}
int Builder::RNNSequenceLayer::getHiddenSize() const {
    return getLayer()->getParameters().at("hidden_size");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setHiddenSize(int size) {
    getLayer()->getParameters()["hidden_size"] = size;
    return *this;
}
bool Builder::RNNSequenceLayer::getSequenceDim() const {
    return getLayer()->getParameters().at("sequence_dim");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setSqquenceDim(bool flag) {
    getLayer()->getParameters()["sequence_dim"] = flag;
    return *this;
}
const std::vector<std::string>& Builder::RNNSequenceLayer::getActivations() const {
    return getLayer()->getParameters().at("activations");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivations(const std::vector<std::string>& activations) {
    getLayer()->getParameters()["activations"] = activations;
    return *this;
}
const std::vector<float>& Builder::RNNSequenceLayer::getActivationsAlpha() const {
    return getLayer()->getParameters().at("activations_alpha");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {
    getLayer()->getParameters()["activations_alpha"] = activations;
    return *this;
}
const std::vector<float>& Builder::RNNSequenceLayer::getActivationsBeta() const {
    return getLayer()->getParameters().at("activations_beta");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {
    getLayer()->getParameters()["activations_beta"] = activations;
    return *this;
}
REG_CONVERTER_FOR(RNNSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["hidden_size"] = cnnLayer->GetParamAsInt("hidden_size");
    layer.getParameters()["sequence_dim"] = cnnLayer->GetParamsAsBool("sequence_dim", true);
    std::vector<std::string> activations;
    std::istringstream stream(cnnLayer->GetParamAsString("activations"));
    std::string str;
    while (getline(stream, str, ',')) {
         activations.push_back(str);
    }
    layer.getParameters()["activations"] = activations;
    layer.getParameters()["activations_alpha"] = cnnLayer->GetParamAsFloats("activations_alpha");
    layer.getParameters()["activations_beta"] = cnnLayer->GetParamAsFloats("activations_beta");
});


