// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>
#include <string>
#include <sstream>

#include <ie_core.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "functional_test_utils/network_utils.hpp"

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
        auto cnnNetwork = ie.ReadNetwork(oldModel, weights);

        FuncTestUtils::compareCNNNetworks(network, cnnNetwork, false);
        IE_SUPPRESS_DEPRECATED_START
        for (auto it = network.begin(); it != network.end(); it++) {
            InferenceEngine::CNNLayerPtr layer = *it;
            ASSERT_NE(nullptr, layer->getNode());
        }

        for (auto it = cnnNetwork.begin(); it != cnnNetwork.end(); it++) {
            InferenceEngine::CNNLayerPtr layer = *it;
            ASSERT_EQ(nullptr, layer->getNode());
        }
        IE_SUPPRESS_DEPRECATED_END
    }
};
