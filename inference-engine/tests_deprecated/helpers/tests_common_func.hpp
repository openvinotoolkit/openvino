// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <details/ie_cnn_network_tools.h>
#include "low_precision_transformations/network_helper.hpp"

// use to display additional test info:
//   1. low precision transformation parameters
//   2. reference and actual outputs
// #define DISPLAY_RESULTS

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

class TestsCommonFunc {
    static CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName) {
        std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
        for (CNNLayerPtr layer : layers) {
            if (layer->name == layerName) {
                return layer;
            }
        }

        return nullptr;
    }
public:

    InferenceEngine::Blob::Ptr readInput(std::string path, int batch = 1);

    static void checkLayerOuputPrecision(
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

    static void checkLayerOuputPrecision(const ICNNNetwork& network, const std::string& layerName, Precision expectedPrecision) {
        CNNLayerPtr layer = getLayer(network, layerName);
        if (layer == nullptr) {
            THROW_IE_EXCEPTION << "layer '" << layerName << "' was not found";
        }
        for (DataPtr data : layer->outData) {
            ASSERT_EQ(expectedPrecision, data->getPrecision()) << " unexpected precision " << data->getPrecision() << " for layer " << layerName;
        }
    }

    static void checkLayerOuputPrecision(const ICNNNetwork& network, const std::string& layerName, std::vector<Precision> expectedPrecisions) {
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

    bool compareTop(
        InferenceEngine::Blob& blob,
        std::vector<std::pair<int, float>> &ref_top,
        int batch_to_compare = 0,
        float threshold = 0.005f,
        const size_t classesCanBeChangedIndex = 9999,
        const bool compareRawValues = true);
};

IE_SUPPRESS_DEPRECATED_END
