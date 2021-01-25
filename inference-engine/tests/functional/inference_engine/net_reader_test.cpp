// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <legacy/details/ie_cnn_network_tools.h>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/network_utils.hpp"


#ifdef ENABLE_UNICODE_PATH_SUPPORT

#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#include <codecvt>

#endif

using NetReaderNoParamTest = CommonTestUtils::TestsCommon;

TEST_F(NetReaderNoParamTest, IncorrectModel) {
    InferenceEngine::Core ie;
    ASSERT_THROW(ie.ReadNetwork("incorrectFilePath"), InferenceEngine::details::InferenceEngineException);
}

using NetReaderTestParams = std::tuple<InferenceEngine::SizeVector, InferenceEngine::Precision>;

class NetReaderTest
        : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<NetReaderTestParams> {
protected:
    static void read(const std::string &modelPath, const std::string &weightsPath, InferenceEngine::Core &ie,
                     InferenceEngine::CNNNetwork &network) {
        network = ie.ReadNetwork(modelPath, weightsPath);
    }

    void SetUp() override {
        std::tie(_inputDims, _netPrc) = GetParam();
        (void) FuncTestUtils::TestModel::generateTestModel(_modelPath,
                                                             _weightsPath,
                                                             _netPrc,
                                                             _inputDims,
                                                             &_refLayers);
    }

    void TearDown() override {
        CommonTestUtils::removeIRFiles(_modelPath, _weightsPath);
    }

    /* validates a read network with the reference map of CNN layers */
    void compareWithRef(const InferenceEngine::CNNNetwork &network,
                        const std::vector<InferenceEngine::CNNLayerPtr> &refLayersVec) {
        IE_SUPPRESS_DEPRECATED_START
        auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(network);
        ASSERT_NO_THROW(FuncTestUtils::compareLayerByLayer(
                InferenceEngine::details::CNNNetSortTopologically(InferenceEngine::CNNNetwork(convertedNetwork)),
                refLayersVec, false));
        IE_SUPPRESS_DEPRECATED_END
    }

    const std::string _modelPath = "NetReader_test.xml";
    const std::string _weightsPath = "NetReader_test.bin";
    InferenceEngine::SizeVector _inputDims;
    InferenceEngine::Precision _netPrc;

    std::vector<InferenceEngine::CNNLayerPtr> _refLayers;
};

TEST_P(NetReaderTest, ReadCorrectModelWithWeightsAndValidate) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    read(_modelPath, _weightsPath, ie, network);

    for (auto input : network.getInputsInfo()) {
        input.second->setPrecision(_netPrc);
    }
    for (auto input : network.getOutputsInfo()) {
        input.second->setPrecision(_netPrc);
    }

    compareWithRef(network, _refLayers);
}

TEST_P(NetReaderTest, ReadNetworkTwiceSeparately) {
    InferenceEngine::Core ie;

    InferenceEngine::CNNNetwork network;
    read(_modelPath, _weightsPath, ie, network);

    InferenceEngine::CNNNetwork network2;
    read(_modelPath, _weightsPath, ie, network2);

    IE_SUPPRESS_DEPRECATED_START

    auto& icnn = static_cast<InferenceEngine::ICNNNetwork &>(network);
    auto& icnn2 = static_cast<InferenceEngine::ICNNNetwork &>(network2);

    ASSERT_NE(&icnn,
              &icnn2);
    ASSERT_NO_THROW(FuncTestUtils::compareCNNNetworks(network, network2));

    IE_SUPPRESS_DEPRECATED_END
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

TEST_P(NetReaderTest, ReadCorrectModelWithWeightsUnicodePath) {
    GTEST_COUT << "params.modelPath: '" << _modelPath << "'" << std::endl;
    GTEST_COUT << "params.weightsPath: '" << _weightsPath << "'" << std::endl;
    GTEST_COUT << "params.netPrc: '" << _netPrc.name() << "'" << std::endl;
    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring modelPath = CommonTestUtils::addUnicodePostfixToPath(_modelPath, postfix);
        std::wstring weightsPath = CommonTestUtils::addUnicodePostfixToPath(_weightsPath, postfix);
        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(_modelPath, modelPath);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << _modelPath << "' to '"
                       << FileUtils::wStringtoMBCSstringChar(modelPath) << "'";
            }
            is_copy_successfully = CommonTestUtils::copyFile(_weightsPath, weightsPath);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << _weightsPath << "' to '"
                       << FileUtils::wStringtoMBCSstringChar(weightsPath) << "'";
            }
            GTEST_COUT << "Test " << testIndex << std::endl;
            InferenceEngine::Core ie;
            ASSERT_NO_THROW(ie.ReadNetwork(modelPath, weightsPath));
            CommonTestUtils::removeFile(modelPath);
            CommonTestUtils::removeFile(weightsPath);
            GTEST_COUT << "OK" << std::endl;
        }
        catch (const InferenceEngine::details::InferenceEngineException &e_next) {
            CommonTestUtils::removeFile(modelPath);
            CommonTestUtils::removeFile(weightsPath);
            FAIL() << e_next.what();
        }
    }
}

#endif

TEST(NetReaderTest, IRSupportModelDetection) {
    InferenceEngine::Core ie;

    static char const *model = R"V0G0N(<net name="Network" version="10" some_attribute="Test Attribute">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="Abs" id="1" type="Abs" version="experimental">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    // For supported model detection the IRReader uses first 512 bytes from model.
    // These headers shifts the trim place.

    std::string headers[] = {
        R"()",
        R"(<!-- <net name="Network" version="10" some_attribute="Test Attribute"> -->)",
        R"(<!-- <net name="Network" version="10" some_attribute="Test Attribute"> -->
<!-- <net name="Network" version="10" some_attribute="Test Attribute"> -->
<!-- <net name="Network" version="10" some_attribute="Test Attribute"> -->
<!-- <net name="Network" version="10" some_attribute="Test Attribute"> -->
<!-- The quick brown fox jumps over the lazy dog -->
<!-- The quick brown fox jumps over the lazy dog -->
<!-- The quick brown fox jumps over the lazy dog -->)"
    };

    InferenceEngine::Blob::CPtr weights;

    for (auto header : headers) {
        ASSERT_NO_THROW(ie.ReadNetwork(header + model, weights));
    }
}

std::string getTestCaseName(testing::TestParamInfo<NetReaderTestParams> testParams) {
    InferenceEngine::SizeVector dims;
    InferenceEngine::Precision prc;
    std::tie(dims, prc) = testParams.param;

    std::ostringstream ss;
    std::copy(dims.begin(), dims.end()-1, std::ostream_iterator<size_t>(ss, "_"));
    ss << dims.back() << "}_" << prc.name();
    return "{" + ss.str();
}

static const auto params = testing::Combine(
        testing::Values(InferenceEngine::SizeVector{1, 3, 227, 227}),
        testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16));

INSTANTIATE_TEST_CASE_P(
        NetReaderTest,
        NetReaderTest,
        params,
        getTestCaseName);

#ifdef GTEST_COUT
#undef GTEST_COUT
#endif
