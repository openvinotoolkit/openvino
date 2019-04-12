// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_crop_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::CropLayer::CropLayer(const std::string& name): LayerDecorator("Crop", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(2);
}

Builder::CropLayer::CropLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Crop");
}

Builder::CropLayer::CropLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Crop");
}

Builder::CropLayer& Builder::CropLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::CropLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::CropLayer& Builder::CropLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}

const Port& Builder::CropLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::CropLayer& Builder::CropLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::CropLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::CropLayer& Builder::CropLayer::setAxis(const std::vector<size_t>& axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

const std::vector<size_t> Builder::CropLayer::getOffset() const {
    return getLayer()->getParameters().at("offset");
}

Builder::CropLayer& Builder::CropLayer::setOffset(const std::vector<size_t>& offsets) {
    getLayer()->getParameters()["offset"] = offsets;
    return *this;
}

REG_VALIDATOR_FOR(Crop, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    if (input_layer->getInputPorts().size() != 2) {
        THROW_IE_EXCEPTION << "Incorrect parameters for getLayer() " << input_layer->getName()
                           << " should have 2 input ports.";
    }
    if (input_layer->getOutputPorts().size() != 1) {
        THROW_IE_EXCEPTION << "Incorrect parameters for getLayer() " << input_layer->getName()
                           << " should have 1 output port";
    }
    Builder::CropLayer layer(input_layer);
    if (layer.getAxis().size() != layer.getOffset().size()) {
        THROW_IE_EXCEPTION <<  "Incorrect parameters for getLayer() " << input_layer->getName()
                           << ". Axis size must be equal to the size of Offset";
    }
    for (size_t i = 0; i < layer.getAxis().size(); ++i) {
        const size_t index = layer.getAxis()[i];
        if (index >= layer.getInputPorts()[0].shape().size()) {
            THROW_IE_EXCEPTION << "Incorrect parameters for getLayer() " << input_layer->getName()
                               << ". Each element of Axis should be less than input shape length";
        }
        if (layer.getOutputPort().shape()[index] != layer.getInputPorts()[1].shape()[index]) {
            THROW_IE_EXCEPTION <<  "Incorrect parameters for getLayer() " << input_layer->getName()
                               << ". The second input shapes should have the same value as the output shapes in the indexes contained in Axis";
        }
        if (layer.getInputPorts()[0].shape()[index] < layer.getOutputPort().shape()[index] + layer.getOffset()[i]) {
            THROW_IE_EXCEPTION <<  "Incorrect parameters for getLayer() " << input_layer->getName()
                               << ". The sum of offset and output shape in the " << i + 1 << " dimension is bigger then input shape size";
        }
    }
});

REG_CONVERTER_FOR(Crop, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::vector<unsigned int> tmp = cnnLayer->GetParamAsUInts("axis");
    layer.getParameters()["axis"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["axis"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }

    tmp = cnnLayer->GetParamAsUInts("offset");
    layer.getParameters()["offset"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["offset"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }
});
