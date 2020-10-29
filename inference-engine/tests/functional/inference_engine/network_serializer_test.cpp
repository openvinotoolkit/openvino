// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/network_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"

class CNNNetworkSerializerTest
        : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<InferenceEngine::Precision> {
protected:
    void SetUp() override {
        _netPrc = GetParam();
        /* generate test model */
        FuncTestUtils::TestModel::generateTestModel(_modelPath, _weightsPath, _netPrc);
    }

    void TearDown() override {
        CommonTestUtils::removeIRFiles(_modelPath, _weightsPath);
    }

    InferenceEngine::Precision _netPrc;
    const std::string _modelPath = "NetworkSerializer_test.xml";
    const std::string _weightsPath = "NetworkSerializer_test.bin";
};

TEST_P(CNNNetworkSerializerTest, SerializeEmptyFilePathsThrowsException) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(_modelPath, _weightsPath);
    ASSERT_THROW(network.serialize("", ""), InferenceEngine::details::InferenceEngineException);
}

TEST_P(CNNNetworkSerializerTest, Serialize) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork originalNetwork = ie.ReadNetwork(_modelPath, _weightsPath);
    IE_SUPPRESS_DEPRECATED_START
    originalNetwork.begin();
    IE_SUPPRESS_DEPRECATED_END
    originalNetwork.getInputsInfo().begin()->second->setPrecision(_netPrc);
    originalNetwork.getOutputsInfo().begin()->second->setPrecision(_netPrc);

    std::string xmlFilePath = "NetworkSerializer_test_serialized.xml";
    std::string binFileName = "NetworkSerializer_test_serialized.bin";
    try {
        originalNetwork.serialize(xmlFilePath, binFileName);

        InferenceEngine::CNNNetwork serializedNetwork = ie.ReadNetwork(xmlFilePath, binFileName);
        serializedNetwork.getInputsInfo().begin()->second->setPrecision(_netPrc);
        serializedNetwork.getOutputsInfo().begin()->second->setPrecision(_netPrc);

        FuncTestUtils::compareCNNNetworks(originalNetwork, serializedNetwork);

        CommonTestUtils::removeIRFiles(xmlFilePath, binFileName);
    } catch (...) {
        CommonTestUtils::removeIRFiles(xmlFilePath, binFileName);
        throw;
    }
}

std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> params) {
    return params.param.name();
}

INSTANTIATE_TEST_CASE_P(
        SerializerTest,
        CNNNetworkSerializerTest,
        testing::Values(InferenceEngine::Precision::FP32,
                        InferenceEngine::Precision::FP16),
        getTestCaseName);
