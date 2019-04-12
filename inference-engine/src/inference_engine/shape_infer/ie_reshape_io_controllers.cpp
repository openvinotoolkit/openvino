// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>
#include <string>
#include <vector>
#include <ie_layers.h>
#include <ie_layer_validators.hpp>
#include <blob_factory.hpp>
#include "shape_infer/ie_reshape_io_controllers.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

void DefaultChecker::run(const std::vector<DataPtr>& dataVec, const std::string& layerName) {
    std::string errorBase = "Failed to init controller for reshaping layer `" + layerName + "`";
    if (dataVec.empty()) THROW_IE_EXCEPTION << errorBase + ": vector of data is empty";
    for (const auto& data : dataVec) {
        if (!data) THROW_IE_EXCEPTION << errorBase + ": pointer to the data is null";
    }
}

InputController::InputController(const std::vector<DataPtr>& dataVec, const std::string& layerName,
                                 const DefaultChecker::Ptr& checker) : _dataVec(dataVec), _layerName(layerName) {
    checker->run(_dataVec, layerName);
    for (const auto& data : _dataVec) {
        if (data) {
            _dataNames.push_back(data->name);
            SizeVector dims = data->getTensorDesc().getDims();
            _irShapes.push_back(dims);
            // TODO probably need to create blobs with dimensions, not on getBlobs stage
            _inferedData.push_back(nullptr);
        }
    }
    _shapes = _irShapes;
}

void InputController::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    long pos = getPositionByName(dataName);
    _shapes[pos] = shape;
}

SizeVector InputController::getShapeByName(const std::string& dataName) {
    long pos = getPositionByName(dataName);
    return _shapes[pos];
}

std::vector<SizeVector> InputController::getShapes(bool check) {
    if (check) checkCorrespondence();
    return _shapes;
}

void InputController::applyChanges() {
    checkCorrespondence();
    for (int i = 0; i < _dataVec.size(); i++) {
        auto data = _dataVec[i];
        if (data) data->setDims(_shapes[i]);
    }
}

void InputController::checkCorrespondence() {
    if (_shapes.size() != _dataVec.size()) {
        THROW_IE_EXCEPTION << "ReshapeLauncher: Number of data(" << _dataVec.size()
                           << ") doesn't match with number of shapes(" << _shapes.size() << ") for layer '"
                           << _layerName << "'!";
    }
    // TODO: iterate and check for emptiness and size matching
}

void InputController::reset() {
    _shapes = _irShapes;
}

std::vector<SizeVector> InputController::getIRShapes() {
    return _irShapes;
}

SizeVector InputController::getIRShapeByName(const std::string& dataName) {
    long pos = getPositionByName(dataName);
    return _irShapes[pos];
}

long InputController::getPositionByName(const std::string& dataName) {
    auto pos = std::distance(_dataNames.begin(), std::find(_dataNames.begin(), _dataNames.end(), dataName));
    if (pos < 0 || pos >= _dataNames.size()) {
        THROW_IE_EXCEPTION << "Failed to find shape that corresponds Data name=" << dataName;
    }
    return pos;
}

void InputController::setShapeByIndex(const SizeVector& shape, size_t index) {
    size_t numShapes = _shapes.size();
    if (index >= numShapes) {
        THROW_IE_EXCEPTION << "Failed to set shape for index(" << index << ") that is more than number of shapes: "
                           << numShapes;
    }
    _shapes[index] = shape;
}

bool InputController::isDataAvailable() {
    if (_inferedData.empty()) return false;
    for (const auto& data : _inferedData) {
        if (!data) return false;
        else if (data->cbuffer() == nullptr) return false;
    }
    return true;
}

std::vector<Blob::CPtr> InputController::getBlobs(bool check) {
    if (check) checkCorrespondence();
    for (int i = 0; i < _dataVec.size(); i++) {
        if (_inferedData[i] == nullptr || _inferedData[i]->cbuffer() == nullptr) {
            TensorDesc desc = _dataVec[i]->getTensorDesc();
            desc.setDims(_shapes[i]);
            // special case of Shape layer: no input data, but blob contains info about dimensions, layout and etc...
            auto blob = make_blob_with_precision(desc);
            _inferedData[i] = blob;
        }
    }
    return _inferedData;
}

void InputController::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {
    long pos = getPositionByName(dataName);
    _inferedData[pos] = blob;
}

OutputController::OutputController(const std::vector<DataPtr>& data, const std::string& layerName,
                                   const DefaultChecker::Ptr& checker)
        : InputController(data, layerName, checker) {}

void OutputController::propagateShapes(const std::set<ReshapeLauncher::Ptr>& launchers) {
    checkCorrespondence();
    unsigned idx = 0;
    for (auto const& outData : _dataVec) {
        for (auto const& inputTo : outData->inputTo) {
            CNNLayerPtr layer = inputTo.second;
            if (layer == nullptr) {
                THROW_IE_EXCEPTION << "Failed to propagate shapes for layer (" << inputTo.first
                                   << "): connected layer is null";
            }
            auto layerName = layer->name;
            auto foundLauncher = std::find_if(launchers.begin(), launchers.end(),
                                              [&layerName](const ReshapeLauncher::Ptr& launcher) {
                                                  return launcher->getLayerName() == layerName;
                                              });
            if (foundLauncher == launchers.end())
                THROW_IE_EXCEPTION << "Failed to find ReshapeLauncher for layer: '" << layerName << "'";
            (*foundLauncher)->setShapeByName(_shapes[idx], outData->name);
        }
        idx++;
    }
}

// Combine with propagate shapes
void OutputController::propagateBlobs(const std::set<ReshapeLauncher::Ptr>& launchers) {
    unsigned idx = 0;
    for (auto const& outData : _dataVec) {
        for (auto const& inputTo : outData->inputTo) {
            CNNLayerPtr layer = inputTo.second;
            if (layer == nullptr) {
                THROW_IE_EXCEPTION << "Failed to propagate shapes for layer (" << inputTo.first
                                   << "): connected layer is null";
            }
            auto layerName = layer->name;
            auto foundLauncher = std::find_if(launchers.begin(), launchers.end(),
                                              [&layerName](const ReshapeLauncher::Ptr& launcher) {
                                                  return launcher->getLayerName() == layerName;
                                              });
            if (foundLauncher == launchers.end())
                THROW_IE_EXCEPTION << "Failed to find ReshapeLauncher for layer: '" << layerName << "'";
            (*foundLauncher)->setBlobByName(_inferedData[idx], outData->name);
        }
        idx++;
    }
}

void OutputController::setShapes(const std::vector<SizeVector>& shapes) {
    _shapes = shapes;
}

void OutputController::setBlobs(const std::vector<Blob::Ptr>& blobs) {
    _inferedData.clear();
    for (const auto& blob : blobs) {
        _inferedData.push_back(blob);
    }
}

std::vector<Blob::Ptr> OutputController::createBlobs() {
    std::vector<Blob::Ptr> blobs;
    for (int i = 0; i < _dataVec.size(); i++) {
        TensorDesc desc = _dataVec[i]->getTensorDesc();
        desc.setDims(_shapes[i]);
        auto blob = make_blob_with_precision(desc);
        blob->allocate();
        blobs.push_back(blob);
    }
    return blobs;
}

