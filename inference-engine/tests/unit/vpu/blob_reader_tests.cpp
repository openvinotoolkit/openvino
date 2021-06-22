// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <memory>

#include <ie_common.h>
#include <cpp/ie_cnn_network.h>

#include <vpu/blob_reader.hpp>
#include <vpu/graph_transformer.hpp>
#include <vpu/utils/logger.hpp>

#include "graph_transformer_tests.hpp"

#include <ngraph/op/util/attr_types.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include <unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp>

using namespace ::testing;
using namespace vpu;
using namespace InferenceEngine;

class VPUBlobReaderHeaderTests: public ::testing::Test, public testing::WithParamInterface<std::vector<size_t>> {
private:
    std::vector<size_t> inputShape;

public:
    size_t getElemSizeByPrecision(Precision precision) {
        size_t elemSize = 0;
        switch (precision) {
        case Precision::U8:
            elemSize = 1;
        case Precision::FP16:
            elemSize = 2;
            break;
        case Precision::FP32:
            elemSize = 4;
            break;
        default:
            throw std::runtime_error(std::string("unsupported precision: ") + precision.name() );
        }

        return elemSize;
    }

    void SetUp() override {
        auto fn_ptr = ngraph::builder::subgraph::makeSplitConvConcat();
        ASSERT_NO_THROW(_network = InferenceEngine::CNNNetwork(fn_ptr));

        auto log = std::make_shared<Logger>("GraphCompiler", LogLevel::None, consoleOutput());
        _compiledGraph = compileNetwork(_network, ncDevicePlatform_t::NC_MYRIAD_X, createConfiguration(), log, &_mockCore);
    }

    CNNNetwork _network;
    CompiledGraph::Ptr _compiledGraph;
    MockICore _mockCore;
};

TEST_P(VPUBlobReaderHeaderTests, canReadCorrectMagicNumber) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    ASSERT_EQ(BLOB_MAGIC_NUMBER, blobReader.getMagicNumber());
}

TEST_P(VPUBlobReaderHeaderTests, canReadCorrectStageCount) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    ASSERT_EQ(_compiledGraph->numActiveStages, blobReader.getStageCount());
}

TEST_P(VPUBlobReaderHeaderTests, canReadCorrectBlobVersion) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    ASSERT_EQ(BLOB_VERSION_MAJOR, blobReader.getVersionMajor());
    ASSERT_EQ(BLOB_VERSION_MINOR, blobReader.getVersionMinor());
}

using VPUBlobReaderInputTests = VPUBlobReaderHeaderTests;

TEST_P(VPUBlobReaderInputTests, areEqualTotalInputSizeFromBlobAndCalculatedFromInputDesc) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    size_t inputTotalSize = 0;
    for (const auto &input : blobReader.getNetworkInputs()) {
        auto dims = input.second->getTensorDesc().getDims();
        auto precision = blobReader.getNetworkInputs().begin()->second->getPrecision();

        inputTotalSize += std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()) * getElemSizeByPrecision(precision);
    }

    auto inputInfo = blobReader.getInputInfo();
    ASSERT_GE(inputInfo.totalSize, inputTotalSize);
}

TEST_P(VPUBlobReaderInputTests, canGetCorrectInputDimsFromImportedNetwork) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    auto parsedNetworkInputs = blobReader.getNetworkInputs();
    auto expectedNetworkInputs = _network.getInputsInfo();

    for (auto&& actual : parsedNetworkInputs) {
        auto actualDims = actual.second->getTensorDesc().getDims();
        size_t actualTotalSize = std::accumulate(actualDims.begin(), actualDims.end(), 1, std::multiplies<size_t>());

        ASSERT_GT(expectedNetworkInputs.count(actual.first), 0);
        auto expectedDims = expectedNetworkInputs[actual.first]->getTensorDesc().getDims();
        size_t expectedTotalSize = std::accumulate(expectedDims.begin(), expectedDims.end(), 1, std::multiplies<size_t>());

        ASSERT_EQ(actualTotalSize, expectedTotalSize);
    }
}

TEST_P(VPUBlobReaderInputTests, canGetCorrectInputNamesFromImportedNetwork) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    auto parsedNetworkInputs   = blobReader.getNetworkInputs();
    auto expectedNetworkInputs = _network.getInputsInfo();

    for (auto&& actual : parsedNetworkInputs) {
        ASSERT_GT(expectedNetworkInputs.count(actual.first), 0);
    }

    for (auto&& expected : expectedNetworkInputs) {
        ASSERT_GT(parsedNetworkInputs.count(expected.first), 0);
    }
}

using VPUBlobReaderOutputTests = VPUBlobReaderHeaderTests;

TEST_P(VPUBlobReaderOutputTests, areEqualTotalOutputSizeFromBlobAndCalculatedFromOutputDesc) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    size_t outputTotalSize = 0;
    for (const auto &input : blobReader.getNetworkOutputs()) {
        auto dims = input.second->getDims();
        auto precision = blobReader.getNetworkOutputs().begin()->second->getPrecision();

        outputTotalSize += std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()) * getElemSizeByPrecision(precision);
    }

    auto outputInfo = blobReader.getOutputInfo();
    ASSERT_GE(outputInfo.totalSize, outputTotalSize);
}

TEST_P(VPUBlobReaderOutputTests, canGetCorrectOutputDimsFromImportedNetwork) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    auto parsedNetworkOutputs = blobReader.getNetworkOutputs();
    auto expectedNetworkOutputs = _network.getOutputsInfo();

    for (auto&& actual : parsedNetworkOutputs) {
        auto actualDims = actual.second->getDims();
        size_t actualTotalSize = std::accumulate(actualDims.begin(), actualDims.end(), 1, std::multiplies<size_t>());

        ASSERT_GT(expectedNetworkOutputs.count(actual.first), 0);
        auto expectedDims = expectedNetworkOutputs[actual.first]->getDims();
        size_t expectedTotalSize = std::accumulate(expectedDims.begin(), expectedDims.end(), 1, std::multiplies<size_t>());

        ASSERT_EQ(actualTotalSize, expectedTotalSize);
    }
}

TEST_P(VPUBlobReaderOutputTests, canGetCorrectOutputNamesFromImportedNetwork) {
    SetUp();
    BlobReader blobReader;
    ASSERT_NO_THROW(blobReader.parse(_compiledGraph->blob));

    auto parsedNetworkOutputs   = blobReader.getNetworkOutputs();
    auto expectedNetworkOutputs = _network.getOutputsInfo();

    for (auto&& actual : parsedNetworkOutputs) {
        ASSERT_GT(expectedNetworkOutputs.count(actual.first), 0);
    }

    for (auto&& expected : expectedNetworkOutputs) {
        ASSERT_GT(parsedNetworkOutputs.count(expected.first), 0);
    }
}


const std::vector<size_t> inputShape = {{1, 4, 10, 10}};

INSTANTIATE_TEST_SUITE_P(myriadBlobReader_nightly, VPUBlobReaderHeaderTests, ::testing::Values(inputShape));
INSTANTIATE_TEST_SUITE_P(myriadBlobReader_nightly, VPUBlobReaderInputTests, ::testing::Values(inputShape));
INSTANTIATE_TEST_SUITE_P(myriadBlobReader_nightly, VPUBlobReaderOutputTests, ::testing::Values(inputShape));
