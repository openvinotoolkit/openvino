// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/legacy_api/include/legacy/ngraph_ops/crop_ie.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>, //input shape
        std::vector<int64_t>, //dims
        std::vector<int64_t> // offset
        > testCaseParams;

typedef std::tuple<
        testCaseParams,
        InferenceEngine::Precision,        // Net precision. We'll use only the net precision because the primitive is not supposed to convert precisions.
        std::string,                       // Device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> CropLayerCPUTestParamSet;

class CropLayerCPUTest : public testing::WithParamInterface<CropLayerCPUTestParamSet>,
                        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CropLayerCPUTestParamSet> obj) {
        testCaseParams testCase;
        InferenceEngine::Precision netPrc;
        std::string targetName;
        std::map<std::string, std::string> additionalConfig;

        CPUSpecificParams cpuParams;
        std::tie(testCase, netPrc, targetName, additionalConfig, cpuParams) = obj.param;

        std::ostringstream result;
        result << "inShape=" << CommonTestUtils::vec2str(std::get<0>(testCase)) << "_";
        result << "dims=" << CommonTestUtils::vec2str(std::get<1>(testCase)) << "_";
        result << "offset=" << CommonTestUtils::vec2str(std::get<2>(testCase)) << "_";
        result << "netPRC=" << netPrc.name() << "_";
        result << "targetDevice=" << targetName;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        testCaseParams testCase;
        std::vector<size_t> inpShape;
        std::vector<int64_t> dims;
        std::vector<int64_t> offset;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additionalConfig;
        CPUSpecificParams cpuParams;
        std::tie(testCase, netPrecision, targetDevice, additionalConfig, cpuParams) = this->GetParam();
        std::tie(inpShape, dims, offset) = testCase;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        inPrc = outPrc = netPrecision;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inpShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<int64_t> axes;
        for (size_t i = 0; i < inpShape.size(); ++i) {
            axes.push_back(i);
        }
        auto ss = std::make_shared<ngraph::op::CropIE>(paramOuts[0], axes, dims, offset);

        std::string strExpectedPrc;
        if (Precision::BF16 == inPrc) {
            strExpectedPrc = "BF16";
        } else if (Precision::FP32 == inPrc) {
            strExpectedPrc = "FP32";
        }

        selectedType = "unknown_" + strExpectedPrc;

        ss->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
        function = std::make_shared<ngraph::Function>(results, params, "Crop");
    }
};

TEST_P(CropLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Crop");
}

namespace {
// Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
std::map<std::string, std::string> additional_config = {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}};

std::vector<Precision> netPrc = {Precision::BF16, Precision::FP32};

std::vector<testCaseParams> testCasesPlain2D = {testCaseParams{{32, 32}, {32, 10}, {0, 20}},
                                                testCaseParams{{32, 20}, {30, 10}, {2, 10}}};

const auto CropParamsPlain2D = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain2D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Plain_2D, CropLayerCPUTest, CropParamsPlain2D, CropLayerCPUTest::getTestCaseName);

std::vector<testCaseParams> testCasesPlain4D = {testCaseParams{{1, 5, 32, 32}, {1, 2, 23, 23}, {0, 2, 5, 4}},
                                                testCaseParams{{1, 5, 32, 32}, {1, 5, 5, 5}, {0, 0, 20, 20}},
                                                testCaseParams{{1, 5, 32, 32}, {1, 5, 32, 10}, {0, 0, 0, 20}},
                                                testCaseParams{{1, 5, 32, 20}, {1, 5, 30, 10}, {0, 0, 2, 10}}};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const auto CropParamsPlain4D = ::testing::Combine(
                ::testing::ValuesIn(testCasesPlain4D),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config),
                ::testing::Values(cpuParams_4D.at(1)));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Plain_4D, CropLayerCPUTest, CropParamsPlain4D, CropLayerCPUTest::getTestCaseName);

std::vector<testCaseParams> testCasesBlocked4D = {testCaseParams{{1, 16, 32, 32}, {1, 16, 5, 5}, {0, 0, 20, 20}},
                                                  testCaseParams{{1, 32, 32, 32}, {1, 16, 32, 10}, {0, 0, 0, 20}}};

const auto CropParamsBlocked4D = ::testing::Combine(
        ::testing::ValuesIn(testCasesBlocked4D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(filterCPUSpecificParams(cpuParams_4D).front()));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Blocked_4D, CropLayerCPUTest, CropParamsBlocked4D, CropLayerCPUTest::getTestCaseName);

std::vector<testCaseParams> testCasesPlain4DynBatch = {testCaseParams{{10, 5, 32, 32}, {1, 2, 23, 23}, {0, 2, 5, 4}},
                                                       testCaseParams{{10, 5, 32, 32}, {1, 5, 5, 5}, {0, 0, 20, 20}},
                                                       testCaseParams{{10, 5, 32, 32}, {1, 5, 32, 10}, {0, 0, 0, 20}},
                                                       testCaseParams{{10, 5, 32, 20}, {1, 5, 30, 10}, {0, 0, 2, 10}}};

std::map<std::string, std::string> additional_config_dyn_batch = {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO},
                                                                  {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}};

const auto CropParamsPlain4DynBatch = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain4DynBatch),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config_dyn_batch),
        ::testing::Values(cpuParams_4D.at(1)));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Blocked_4DynBatch, CropLayerCPUTest, CropParamsPlain4DynBatch, CropLayerCPUTest::getTestCaseName);

std::vector<testCaseParams> testCasesPlain5D = {testCaseParams{{1, 5, 32, 20, 14}, {1, 5, 30, 10, 8}, {0, 0, 2, 10, 6}},
                                                testCaseParams{{5, 9, 32, 20, 14}, {2, 5, 30, 10, 8}, {3, 4, 2, 10, 6}}};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto CropParamsPlain5D = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain5D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(cpuParams_5D.at(1)));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Plain_5D, CropLayerCPUTest, CropParamsPlain5D, CropLayerCPUTest::getTestCaseName);

std::vector<testCaseParams> testCasesBlocked5D = {testCaseParams{{1, 32, 32, 20, 14}, {1, 16, 30, 10, 8}, {0, 0, 2, 10, 6}},
                                                  testCaseParams{{5, 32, 32, 20, 14}, {2, 32, 30, 10, 8}, {3, 0, 2, 10, 6}}};

const auto CropParamsBlocked5D = ::testing::Combine(
        ::testing::ValuesIn(testCasesBlocked5D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(cpuParams_5D.at(0)));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Blocked_5D, CropLayerCPUTest, CropParamsBlocked5D, CropLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions

