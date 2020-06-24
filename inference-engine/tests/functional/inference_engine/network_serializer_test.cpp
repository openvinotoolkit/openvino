// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>
#include <network_serializer.h>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/network_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

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
    {
        IE_SUPPRESS_DEPRECATED_START
        // convert to old representation
        originalNetwork.getInputsInfo().begin()->second->getInputData()->getCreatorLayer();
        IE_SUPPRESS_DEPRECATED_END
    }
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

TEST_P(CNNNetworkSerializerTest, TopoSortResultUnique) {
    InferenceEngine::CNNNetwork network(ngraph::builder::subgraph::makeConvPoolRelu());
    auto sorted = InferenceEngine::Serialization::TopologicalSort(network);

    std::vector<std::string> actualLayerNames;
    for (auto&& layer : sorted) {
        IE_SUPPRESS_DEPRECATED_START
        actualLayerNames.emplace_back(layer->name);
        IE_SUPPRESS_DEPRECATED_END
    }

    std::vector<std::string> expectedLayerNames = {
            "Param_1", "Const_1", "Reshape_1", "Conv_1", "Pool_1", "Relu_1", "Const_2", "Reshape_2"
    };

    ASSERT_EQ(expectedLayerNames, actualLayerNames);
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
