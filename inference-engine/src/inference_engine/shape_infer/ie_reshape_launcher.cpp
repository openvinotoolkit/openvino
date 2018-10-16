// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_layers.h>
#include <ie_layer_validators.hpp>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "shape_infer/ie_reshape_launcher.hpp"
#include "shape_infer/ie_reshape_io_controllers.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

void DefaultInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    if (!impl) THROW_IE_EXCEPTION << errorBase + " shape infer implementation is null";
}

InputController* DefaultInitializer::createInputController(const CNNLayer* layer) {
    std::vector<DataPtr> data;
    for (auto const& insData : layer->insData) {
        data.push_back(insData.lock());
    }
    return new InputController(data, layer->name, false);
}

OutputController* DefaultInitializer::createOutputController(const CNNLayer* layer) {
    return new OutputController(layer->outData, layer->name, false);
}

ReshapeLauncher::ReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                 const DefaultInitializer::Ptr& initializer) : _layer(layer), _impl(impl) {
    initializer->check(layer, impl);
    _iController = initializer->createInputController(layer);
    _oController = initializer->createOutputController(layer);
}

ReshapeLauncher::~ReshapeLauncher() {
    delete _iController;
    delete _oController;
    _iController = nullptr;
    _oController = nullptr;
}

void ReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    _iController->setShapeByName(shape, dataName);
}

void ReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    ResponseDesc resp;
    std::vector<SizeVector> outShapes;
    auto sts = _impl->inferShapes(_iController->getShapes(true), _layer->params, _layer->blobs, outShapes, &resp);
    _oController->setShapes(outShapes);
    if (sts != OK)
        THROW_IE_EXCEPTION << resp.msg;
    _oController->propagateShapes(launchers);
}

void ReshapeLauncher::applyChanges(CNNLayer* layer) {
    checkLayer(layer);
    _iController->applyChanges();
    _oController->applyChanges();
}

void ReshapeLauncher::reset() {
    _iController->reset();
    _oController->reset();
}

std::string ReshapeLauncher::getLayerName() const {
    return _layer->name;
}

std::string ReshapeLauncher::getLayerType() const {
    return _layer->type;
}

void ReshapeLauncher::checkLayer(CNNLayer* layer) {
    if ((nullptr == _layer || layer == nullptr)) {
        THROW_IE_EXCEPTION << "Can't apply changes for empty layer";
    }
    auto oldParams = _layer->params;
    auto newParams = layer->params;
    if ((!oldParams.empty() && !newParams.empty() && !std::equal(oldParams.begin(), oldParams.end(), newParams.begin()))
        || (_layer->name != layer->name) || (_layer->type != layer->type) || oldParams.size() != newParams.size()) {
        THROW_IE_EXCEPTION << "Can't apply changes for layer with another params";
    }
}

void ReshapeLauncher::setIRShapeByName(const std::string& dataName) {
    SizeVector foundShape = _iController->getIRShapeByName(dataName);
    _iController->setShapeByName(foundShape, dataName);
}

void ReshapeLauncher::setShapeInferImpl(const IShapeInferImpl::Ptr& impl) {
    _impl = impl;
}

const CNNLayer* ReshapeLauncher::getLayer() const {
    return _layer;
}

InputController* FakeInitializer::createInputController(const CNNLayer* layer) {
    std::vector<DataPtr> outData;
    for (auto const& insData : layer->insData) {
        outData.push_back(insData.lock());
    }
    return new InputController(outData, layer->name, true);
}

void FakeInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
}

OutputController* FakeInitializer::createOutputController(const CNNLayer* layer) {
    return new OutputController(layer->outData, layer->name, true);
}

FakeReshapeLauncher::FakeReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
        : ReshapeLauncher(layer, impl, std::make_shared<FakeInitializer>()) {
}

void FakeReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    auto iShapesIR = _iController->getIRShapes();
    auto oShapesIR = _oController->getIRShapes();
    auto iShapes = _iController->getShapes(true);

    for (int i = 0; i < iShapes.size(); i++) {
        auto newInShape = iShapes[i];
        auto irInShape = iShapesIR[i];
        bool equal = std::equal(newInShape.begin(), newInShape.end(), irInShape.begin());
        if (!equal) {
            return THROW_IE_EXCEPTION
                    << "Failed to infer shapes for layer with type: " << _layer->type
                    << ". Use @IShapeInferExtension class to register shape infer function for this layer";
        }
    }

    _oController->setShapes(oShapesIR);
    _oController->propagateShapes(launchers);
}

void OutputOnlyInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    if (!layer->insData.empty())
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: "
                           << "layer type (`" + layer->type + "`) is supposed to not have inputs, but actually it has";
}

InputController* OutputOnlyInitializer::createInputController(const CNNLayer* layer) {
    return nullptr;
}

OutputController* OutputOnlyInitializer::createOutputController(const CNNLayer* layer) {
    return new OutputController(layer->outData, layer->name, true);
}

OutputOnlyReshapeLauncher::OutputOnlyReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                                     const OutputOnlyInitializer::Ptr& initializer)
        : ReshapeLauncher(layer, impl, initializer) {}

void OutputOnlyReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    _oController->setShapeByName(shape, dataName);
}

void OutputOnlyReshapeLauncher::setIRShapeByName(const std::string& dataName) {
    SizeVector foundShape = _oController->getIRShapeByName(dataName);
    _oController->setShapeByName(foundShape, dataName);
}

void OutputOnlyReshapeLauncher::applyChanges(CNNLayer* layer) {
    checkLayer(layer);
    _oController->applyChanges();
}

void OutputOnlyReshapeLauncher::reset() {
    _oController->reset();
}

void InputInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    OutputOnlyInitializer::check(layer, impl);
    std::string errorBase = "Failed to init reshape launcher: layer type (`" + layer->type + "`) is not";
    if (details::equal(layer->type, "memory")) {
        if (!layer->GetParamAsInt("index"))
            THROW_IE_EXCEPTION << errorBase << " `Memory`(as input)";
    } else if (!::details::equal(layer->type, "input")) {
        THROW_IE_EXCEPTION << errorBase << " `Input`";
    }
}

InputReshapeLauncher::InputReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                           const DefaultInitializer::Ptr& initializer)
        : OutputOnlyReshapeLauncher(layer, impl, initializer) {}

void InputReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    auto oShapes = _oController->getShapes(false);
    auto oIRShapes = _oController->getIRShapes();
    for (size_t i = 0; i < oShapes.size(); i++) {
        if (oShapes[i].empty()) {
            _oController->setShapeByIndex(oIRShapes[i], i);
        }
    }
    _oController->propagateShapes(launchers);
}

void ConstInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    OutputOnlyInitializer::check(layer, impl);
    if (!::details::equal(layer->type, "const"))
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: layer type (`" + layer->type + "`) is not `Const`";
}

ConstReshapeLauncher::ConstReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
        : OutputOnlyReshapeLauncher(layer, impl, std::make_shared<ConstInitializer>()) {}

void ConstReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    auto oShapesIR = _oController->getIRShapes();
    auto oShapes = _oController->getShapes(false);

    if (oShapes.empty()) {
        _oController->setShapes(oShapesIR);
    }
    if (oShapes != oShapesIR) {
        THROW_IE_EXCEPTION << "Failed to set different shapes for Const layer,"
                           << " original shapes:" << details::dumpVec(oShapesIR)
                           << " new shapes:" << details::dumpVec(oShapes);
    }
    _oController->propagateShapes(launchers);
}

void OutMemoryInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    int index = layer->GetParamAsInt("index");
    if (!::details::equal(layer->type, "memory") && index)
        THROW_IE_EXCEPTION
                << "Failed to init reshape launcher: layer type (`" + layer->type + "`) is not `Memory` as output";
    if (!layer->outData.empty())
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: "
                           << "layer type (`" + layer->type +
                              "`) is supposed to not have outputs, but actually it has";
}

OutputController* OutMemoryInitializer::createOutputController(const CNNLayer* layer) {
    return nullptr;
}

OutMemoryReshapeLauncher::OutMemoryReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
        : ReshapeLauncher(layer, impl, std::make_shared<OutMemoryInitializer>()) {
}

void OutMemoryReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
}

void OutMemoryReshapeLauncher::applyChanges(CNNLayer* layer) {
    checkLayer(layer);
    _iController->applyChanges();
}

void OutMemoryReshapeLauncher::reset() {
    _iController->reset();
}
