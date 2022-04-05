// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>

#include <legacy/ie_util_internal.hpp>
#include <tests_common.hpp>
#include <legacy/graph_transformer.h>
#include "blob_factory.hpp"
#include "debug.h"
#include "util_test.hpp"

namespace IE = InferenceEngine;

class ConstTransformatorTest : public IE::ConstTransformer {
public:
    explicit ConstTransformatorTest(IE::details::CNNNetworkImpl* network) : IE::ConstTransformer(network) {}

    const std::map<std::string, bool>
    getConstLayers(const std::vector<InferenceEngine::CNNLayerPtr>& sortedLayers) override {
        return ConstTransformer::getConstLayers(sortedLayers);
    }

    const InferenceEngine::BlobMap getConstData(const std::map<std::string, bool>& constLayers,
                                                    const std::vector<InferenceEngine::CNNLayerPtr>& sortedLayers) override {
        return ConstTransformer::getConstData(constLayers, sortedLayers);
    }

    std::vector<InferenceEngine::CNNLayerPtr>
    foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers, const IE::BlobMap& constData,
                               const std::vector<IE::CNNLayerPtr>& sortedLayers) override {
        auto layers = ConstTransformer::foldConstSubgraphsInternal(constLayers, constData, sortedLayers);
        ConstTransformer::cleanup();
        return layers;
    }

    void trimShapeInputs(const std::vector<InferenceEngine::CNNLayerPtr>& constLayers,
                         std::vector<InferenceEngine::CNNLayerPtr>& allLayers) override {
        ConstTransformer::trimShapeInputs(constLayers, allLayers);
    }

};

class RemoveLayerTests : public testing::Test {
protected:
    void SetUp() override;

    //
    // I1-d1-L1-d4              I4
    //       / \  \              \
    //      |  d7  \            d10
    //      |  |    \            /
    //  I2-d2-L2-d5-L4-d6-L5-d9-L10
    //        /           /
    //       /  ____d8___/
    //      /  /
    // I3-d3-L3
    //
    IE::details::CNNNetworkImplPtr getNetwork();

    IE::CNNLayerPtr getLayer(const std::string& name);

    IE::DataPtr getData(const std::string& name);

    IE::BlobMap fillConstData(const std::vector<std::string>& constLayers);

    IE::BlobMap initConstLayers(const std::vector<std::string>& constLayers);

    IE::BlobMap fillConstDataDiffPrec(const std::vector<std::string>& constLayers);

    IE::BlobMap initConstLayersDiffPrec(const std::vector<std::string>& constLayers);

    NetBuilder netBuilder;
    IE::details::CNNNetworkImplPtr net;
    size_t originalLayersNum;
    std::unique_ptr<ConstTransformatorTest> testTransformator;
};

class AdvancedShapeInferTests : public RemoveLayerTests {
protected:
    void SetUp() override {};
};
