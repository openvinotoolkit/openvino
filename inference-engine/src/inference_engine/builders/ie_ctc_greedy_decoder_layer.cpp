// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_ctc_greedy_decoder_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::CTCGreedyDecoderLayer::CTCGreedyDecoderLayer(const std::string& name): LayerDecorator("CTCGreedyDecoder", name) {
    getLayer()->getOutputPorts().resize(1);
}

Builder::CTCGreedyDecoderLayer::CTCGreedyDecoderLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("CTCGreedyDecoder");
}

Builder::CTCGreedyDecoderLayer::CTCGreedyDecoderLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("CTCGreedyDecoder");
}

Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const std::vector<Port>& Builder::CTCGreedyDecoderLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}
const Port& Builder::CTCGreedyDecoderLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}
bool Builder::CTCGreedyDecoderLayer::getCTCMergeRepeated() const {
    return getLayer()->getParameters().at("ctc_merge_repeated");
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setCTCMergeRepeated(bool flag) {
    getLayer()->getParameters()["ctc_merge_repeated"] = flag;
    return *this;
}

REG_VALIDATOR_FOR(CTCGreedyDecoder, [](const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::CTCGreedyDecoderLayer layer(input_layer);

    if (layer.getInputPorts().empty() || layer.getInputPorts().size() > 2) {
        THROW_IE_EXCEPTION << "Input ports are wrong in layer " << layer.getName() <<
                           ". There are should be 1 or 2 input ports";
    }
});

REG_CONVERTER_FOR(CTCGreedyDecoder, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["ctc_merge_repeated"] = cnnLayer->GetParamAsBool("ctc_merge_repeated", false);
});
