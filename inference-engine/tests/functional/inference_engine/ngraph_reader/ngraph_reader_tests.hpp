// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <vector>
#include <string>
#include <sstream>

#include <ie_core.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "network_utils.hpp"

using namespace testing;
using namespace InferenceEngine;

class NGraphReaderTests : public CommonTestUtils::TestsCommon {
protected:
    void TearDown() override {}
    void SetUp() override {}

    void compareIRs(const std::string& modelV10, const std::string& oldModel, size_t weightsSize = 0, const std::function<void(Blob::Ptr&)>& fillBlob = {}) {
        Core ie;
        Blob::Ptr weights;

        if (weightsSize) {
            weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {weightsSize}, Layout::C));
            weights->allocate();
            CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->size() / sizeof(float));
            if (fillBlob)
                fillBlob(weights);
        }

        auto network = ie.ReadNetwork(modelV10, weights);
        auto f = network.getFunction();
        // WA: we have to resolve dynamysm manually to compare resulting function with v7 IR
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertNMS5ToLegacyMatcher>();
        manager.run_passes(f);
        network = CNNNetwork(f);
        auto cnnNetwork = ie.ReadNetwork(oldModel, weights);

        IE_SUPPRESS_DEPRECATED_START
        auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(network);

        FuncTestUtils::compareCNNNetworks(InferenceEngine::CNNNetwork(convertedNetwork), cnnNetwork, false);

        for (auto it = details::CNNNetworkIterator(convertedNetwork.get()); it != details::CNNNetworkIterator(); it++) {
            InferenceEngine::CNNLayerPtr layer = *it;
            ASSERT_NE(nullptr, layer->getNode());
        }

        ASSERT_EQ(nullptr, cnnNetwork.getFunction());
        for (auto it = details::CNNNetworkIterator(cnnNetwork); it != details::CNNNetworkIterator(); it++) {
            InferenceEngine::CNNLayerPtr layer = *it;
            ASSERT_EQ(nullptr, layer->getNode());
        }
        IE_SUPPRESS_DEPRECATED_END
    }
};
