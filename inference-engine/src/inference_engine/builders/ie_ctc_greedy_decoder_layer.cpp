// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_ctc_greedy_decoder_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::CTCGreedyDecoderLayer::CTCGreedyDecoderLayer(const std::string& name): LayerFragment("CTCGreedyDecoder", name) {
    getLayer().getOutputPorts().resize(1);
}

Builder::CTCGreedyDecoderLayer::CTCGreedyDecoderLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "CTCGreedyDecoder"))
        THROW_IE_EXCEPTION << "Cannot create CTCGreedyDecoderLayer decorator for layer " << getLayer().getType();
}

Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const std::vector<Port>& Builder::CTCGreedyDecoderLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer().getInputPorts() = ports;
    return *this;
}
const Port& Builder::CTCGreedyDecoderLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}
bool Builder::CTCGreedyDecoderLayer::getCTCMergeRepeated() const {
    return getLayer().getParameters()["ctc_merge_repeated"].asBool();
}
Builder::CTCGreedyDecoderLayer& Builder::CTCGreedyDecoderLayer::setCTCMergeRepeated(bool flag) {
    getLayer().getParameters()["ctc_merge_repeated"] = flag;
    return *this;
}

