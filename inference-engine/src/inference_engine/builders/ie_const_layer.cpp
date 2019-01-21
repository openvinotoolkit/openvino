// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_const_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ConstLayer::ConstLayer(const std::string& name): LayerFragment("Const", name) {
    getLayer().getOutputPorts().resize(1);
}

Builder::ConstLayer::ConstLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Const"))
        THROW_IE_EXCEPTION << "Cannot create ConstLayer decorator for layer " << getLayer().getType();
}

Builder::ConstLayer& Builder::ConstLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ConstLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ConstLayer& Builder::ConstLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

Builder::ConstLayer& Builder::ConstLayer::setData(const Blob::CPtr& data) {
    getLayer().addConstantData("custom", data);
    return *this;
}

