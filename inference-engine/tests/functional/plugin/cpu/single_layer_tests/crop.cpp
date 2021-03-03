// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/strided_slice.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

// Given that the ngraph opset does not contain crop operation, we use the StridedSlice operation instead, since it is mapped to the Crop node if certain
// conditions are met.

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        StridedSliceSpecificParams,
        InferenceEngine::Precision,        // Net precision
        std::string,                       // Device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> CropLayerCPUTestParamSet;

class CropLayerCPUTest : public testing::WithParamInterface<CropLayerCPUTestParamSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CropLayerCPUTestParamSet> obj) {
        StridedSliceSpecificParams params;
        InferenceEngine::Precision netPrc;
        std::string targetName;
        std::map<std::string, std::string> additionalConfig;
        CPUSpecificParams cpuParams;
        std::tie(params, netPrc, targetName, additionalConfig, cpuParams) = obj.param;

        std::ostringstream result;
        result << "inShape=" << CommonTestUtils::vec2str(params.inputShape) << "_";
        result << "netPRC=" << netPrc.name() << "_";
        result << "begin=" << CommonTestUtils::vec2str(params.begin) << "_";
        result << "end=" << CommonTestUtils::vec2str(params.end) << "_";
        result << "stride=" << CommonTestUtils::vec2str(params.strides) << "_";
        result << "begin_m=" << CommonTestUtils::vec2str(params.beginMask) << "_";
        result << "end_m=" << CommonTestUtils::vec2str(params.endMask) << "_";
        if (!params.newAxisMask.empty()) {
            result << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.newAxisMask)) << "_";
        }
        if (!params.shrinkAxisMask.empty()) {
            result << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.shrinkAxisMask)) << "_";
        }
        if (!params.ellipsisAxisMask.empty()) {
            result << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.ellipsisAxisMask)) << "_";
        }
        result << "trgDev=" << targetName;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        StridedSliceSpecificParams ssParams;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additionalConfig;
        CPUSpecificParams cpuParams;
        std::tie(ssParams, netPrecision, targetDevice, additionalConfig, cpuParams) = this->GetParam();
        inPrc = outPrc = netPrecision; // because crop does not convert Precisions, but only moves the data
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {ssParams.inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto ss = ngraph::builder::makeStridedSlice(paramOuts[0], ssParams.begin, ssParams.end, ssParams.strides, ngPrc, ssParams.beginMask,
                                                    ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);

        selectedType = std::string("unknown_") + inPrc.name();

        ss->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
        function = std::make_shared<ngraph::Function>(results, params, "StridedSlice");
    }
};

TEST_P(CropLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Crop");
}

namespace {
const std::map<std::string, std::string> additional_config;

const std::vector<Precision> netPrc = {Precision::BF16, Precision::FP32};

const std::vector<StridedSliceSpecificParams> testCasesPlain2D = {StridedSliceSpecificParams{ { 32, 32 }, { 0, 20 }, { 32, 30 }, { 1, 1 },
                                                                                              { 0, 0 }, { 0, 0 },  { },  { },  { } },
                                                                  StridedSliceSpecificParams{ { 32, 20 }, { 2, 10 }, { 32, 20 }, { 1, 1 },
                                                                                              { 0, 0 }, { 0, 0 },  { },  { },  { } } };

const auto CropParamsPlain2D = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain2D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Plain_2D, CropLayerCPUTest, CropParamsPlain2D, CropLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesPlain4D = {
        StridedSliceSpecificParams{ { 1, 5, 32, 32 }, { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 32 }, { 0, 0, 20, 20 }, { 1, 5, 25, 25 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 32 }, { 0, 0, 0, 20 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 20 }, { 0, 0, 2, 10 }, { 1, 5, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } }
};

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

const std::vector<StridedSliceSpecificParams> testCasesBlocked4D = {
        StridedSliceSpecificParams{ { 1, 16, 32, 32 }, { 0, 0, 20, 20 }, { 1, 16, 25, 25 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 32, 32, 32 }, { 0, 0, 0, 20 }, { 1, 16, 32, 30 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
};

const auto CropParamsBlocked4D = ::testing::Combine(
        ::testing::ValuesIn(testCasesBlocked4D),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(filterCPUSpecificParams(cpuParams_4D).front()));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Blocked_4D, CropLayerCPUTest, CropParamsBlocked4D, CropLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesPlain4DynBatch = {
        StridedSliceSpecificParams{ { 10, 5, 32, 32 }, { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 10, 5, 32, 32 }, { 0, 0, 20, 20 }, { 1, 5, 25, 25 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 10, 5, 32, 32 }, { 0, 0, 0, 20 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 10, 5, 32, 20 }, { 0, 0, 2, 10 }, { 1, 5, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } }
};

std::map<std::string, std::string> additional_config_dyn_batch = {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO},
                                                                  {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}};

const auto CropParamsPlain4DynBatch = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain4DynBatch),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config_dyn_batch),
        ::testing::Values(cpuParams_4D.at(1)));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_Blocked_4DynBatch, CropLayerCPUTest, CropParamsPlain4DynBatch, CropLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions

