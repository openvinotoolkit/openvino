// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_const_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ConstLayer::ConstLayer(const std::string& name): LayerDecorator("Const", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getParameters()["custom"] = Blob::CPtr();
}

Builder::ConstLayer::ConstLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Const");
}

Builder::ConstLayer::ConstLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Const");
}

Builder::ConstLayer& Builder::ConstLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ConstLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ConstLayer& Builder::ConstLayer::setPort(const Port &port) {
    const auto & data = getLayer()->getOutputPorts()[0].getData();
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getOutputPorts()[0].setData(data);
    return *this;
}

Builder::ConstLayer& Builder::ConstLayer::setData(const Blob::CPtr& data) {
    getLayer()->getParameters()["custom"] = data;
    getLayer()->getOutputPorts()[0].getData()->setData(std::const_pointer_cast<Blob>(data));
    return *this;
}

const Blob::CPtr& Builder::ConstLayer::getData() const {
    if (getLayer()->getParameters().at("custom").as<Blob::CPtr>().get() !=
            getLayer()->getOutputPorts()[0].getData()->getData().get())
        THROW_IE_EXCEPTION << "Constant data output port has incorrect data!";
    return getLayer()->getParameters().at("custom").as<Blob::CPtr>();
}

REG_VALIDATOR_FOR(Const, [] (const InferenceEngine::Builder::Layer::CPtr& layer, bool partial)  {
    Builder::ConstLayer constBuilder(layer);
    const auto& data = constBuilder.getData();
    if (!data || data->cbuffer() == nullptr)
        THROW_IE_EXCEPTION << "Cannot create Const layer! Data is required!";
});
