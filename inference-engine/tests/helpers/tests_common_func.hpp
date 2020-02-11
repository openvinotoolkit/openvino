// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <details/ie_cnn_network_tools.h>
#include "inference_engine.hpp"
#include "low_precision_transformations/network_helper.hpp"

// use to display additional test info:
//   1. low precision transformation parameters
//   2. reference and actual outputs
// #define DISPLAY_RESULTS

using namespace InferenceEngine;

class TestsCommonFunc {
protected:

    InferenceEngine::Blob::Ptr readInput(std::string path, int batch = 1);

    CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName) {
        std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
        for (CNNLayerPtr layer : layers) {
            if (layer->name == layerName) {
                return layer;
            }
        }

        return nullptr;
    }

   void checkLayerOuputPrecision(
        const ICNNNetwork& network,
        const std::vector<std::string>& layerNames,
        const Precision expectedPrecision,
        const std::string& type = "") {
        for (const std::string& layerName : layerNames) {
            if (!type.empty()) {
                const CNNLayerPtr layer = getLayer(network, layerName);
                if (layer == nullptr) {
                    THROW_IE_EXCEPTION << "layer was not found " << layerName;
                }

                if (layer->type != type) {
                    THROW_IE_EXCEPTION << "layer '" << layer->name << "' type '" << layer->type << "' is not correct, expected " << type;
                }
            }
            checkLayerOuputPrecision(network, layerName, expectedPrecision);
        }
    }

    void checkLayerOuputPrecision(const ICNNNetwork& network, const std::string& layerName, Precision expectedPrecision) {
        CNNLayerPtr layer = getLayer(network, layerName);
        if (layer == nullptr) {
            THROW_IE_EXCEPTION << "layer '" << layerName << "' was not found";
        }
        for (DataPtr data : layer->outData) {
            ASSERT_EQ(expectedPrecision, data->getPrecision()) << " unexpected precision " << data->getPrecision() << " for layer " << layerName;
        }
    }

    void checkLayerInputPrecision(const ICNNNetwork& network, const std::string& layerName, Precision expectedPrecision, int inputIndex = -1) {
        CNNLayerPtr layer = getLayer(network, layerName);
        if (layer == nullptr) {
            THROW_IE_EXCEPTION << "layer '" << layerName << "' was not found";
        }
        for (size_t index = 0ul; index < layer->insData.size(); ++index) {
            if ((inputIndex != -1) && (index != inputIndex)) {
                continue;
            }

            const DataWeakPtr weakData = layer->insData[index];
            ASSERT_EQ(expectedPrecision, weakData.lock()->getPrecision()) << " unexpected precision " << weakData.lock()->getPrecision() << " for layer " << layerName;
        }
    }

    void checkLayerOuputPrecision(const ICNNNetwork& network, const std::string& layerName, std::vector<Precision> expectedPrecisions) {
        CNNLayerPtr layer = getLayer(network, layerName);
        if (layer == nullptr) {
            THROW_IE_EXCEPTION << "layer '" << layerName << "' was not found";
        }
        for (DataPtr data : layer->outData) {
            ASSERT_TRUE(std::any_of(
                expectedPrecisions.begin(),
                expectedPrecisions.end(),
                [&](const Precision precision) { return precision == data->getTensorDesc().getPrecision(); })) <<
                " unexpected precision " << data->getPrecision() << " for layer " << layerName;
        }
    }

    bool hasBlobEqualsValues(Blob& blob) {
        const float* buffer = blob.buffer().as<float*>();
        for (int i = 0; i < (blob.size() - 1); ++i) {
            if (buffer[i] != buffer[i + 1]) {
                return false;
            }
        }
        return true;
    }

    bool checkScalesAndShifts(const CNNLayer& scaleShift, const bool equals) {
        const Blob::Ptr scalesBlob = InferenceEngine::details::CNNNetworkHelper::getBlob(std::make_shared<CNNLayer>(scaleShift), "weights");
        if (equals != hasBlobEqualsValues(*scalesBlob)) {
            return false;
        }

        const Blob::Ptr shiftsBlob = InferenceEngine::details::CNNNetworkHelper::getBlob(std::make_shared<CNNLayer>(scaleShift), "biases");
        if (equals != hasBlobEqualsValues(*shiftsBlob)) {
            return false;
        }

        return true;
    }

    bool compareTop(
        InferenceEngine::Blob& blob,
        std::vector<std::pair<int, float>> &ref_top,
        int batch_to_compare = 0,
        float threshold = 0.005f,
        const size_t classesCanBeChangedIndex = 9999,
        const bool compareRawValues = true);
};
