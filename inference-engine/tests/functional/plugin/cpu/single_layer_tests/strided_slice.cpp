// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/strided_slice.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"


using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        StridedSliceSpecificParams,
        InferenceEngine::Precision,         // Net precision
        std::string,                        // Device name
        std::map<std::string, std::string>, // Additional network configuration
        CPUSpecificParams> StridedSliceLayerCPUTestParamSet;

class StridedSliceLayerCPUTest : public testing::WithParamInterface<StridedSliceLayerCPUTestParamSet>,
                                 virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<StridedSliceLayerCPUTestParamSet> obj) {
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
        result << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.newAxisMask)) << "_";
        result << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.shrinkAxisMask)) << "_";
        result << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.ellipsisAxisMask)) << "_";
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
        inPrc = outPrc = netPrecision;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {ssParams.inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto ss = ngraph::builder::makeStridedSlice(paramOuts[0], ssParams.begin, ssParams.end, ssParams.strides, ngPrc, ssParams.beginMask,
                                                    ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);

        selectedType = std::string("ref_") + inPrc.name();

        ss->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
        function = std::make_shared<ngraph::Function>(results, params, "StridedSlice");
    }
};

TEST_P(StridedSliceLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "StridedSlice");
}

namespace {

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {ncdhw}, {}, {}};

const std::map<std::string, std::string> additional_config;

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<StridedSliceSpecificParams> testCasesPlain2D = {
        StridedSliceSpecificParams{ { 32, 32 }, { 0, 20 }, { 32, 30 }, { 1, 1 },
                                    { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 32, 20 }, { 2, 10 }, { 32, 20 }, { 1, 1 },
                                    { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 32, 20 }, { 2, 10 }, { 32, 20 }, { 1, 2 },
                                    { 0, 1 }, { 1, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 32, 20 }, { 2, 10 }, { 32, 20 }, { 2, 1 },
                                    { 0, 0 }, { 1, 0 },  { },  { },  { } },
};

const auto StridedSliceParamsPlain2D = ::testing::Combine(
        ::testing::ValuesIn(testCasesPlain2D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_2D, StridedSliceLayerCPUTest, StridedSliceParamsPlain2D, StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesCommon4D = {
        StridedSliceSpecificParams{ { 1, 5, 32, 32 }, { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 20 }, { 0, 1, 0, 0 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 32, 20 }, { 0, 0, 10, 0 }, { 1, 3, 20, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 1, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 32 }, { 0, 0, 20, 20 }, { 1, 5, 25, 26 }, { 1, 1, 1, 2 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 32, 32 }, { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 },
                                    { 0, 0, 0, 1 }, { 0, 1, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 20 }, { 0, 0, 2, 10 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 1, 1 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 32, 32 }, { 0, 1, 0, 10 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 },
                                    { 0, 1, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 32, 20 }, { 0, 1, 2, 10 }, { 1, 5, 32, 18 }, { 1, 1, 1, 2 },
                                    { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 8, 32, 20 }, { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 },
                                    { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 8, 32, 20 }, { 0, 0, 10 }, { 0, 32, 18 }, { 1, 1, 1 },
                                    { 1, 1, 0 }, { 1, 1, 0 },  { },  { },  { 1, 0, 0 } },
        StridedSliceSpecificParams{ { 2, 8, 32, 20 }, { 0, 0, 10 }, { 1, 0, 20 }, { 1, 1, 1 },
                                    { 1, 1, 0 }, { 0, 1, 1 },  { },  { },  { 0, 1, 0 } },
        StridedSliceSpecificParams{ { 2, 8, 32, 20 }, { 0, 4, 10 }, { 2, 8, 0 }, { 1, 1, 1 },
                                    { 1, 0, 1 }, { 1, 1, 1 },  { },  { },  { 0, 0, 1 } }
};

const std::vector<CPUSpecificParams> CPUParamsCommon4D = {
        cpuParams_nchw,
        cpuParams_nhwc,
};

const auto StridedSliceParamsCommon4D = ::testing::Combine(
        ::testing::ValuesIn(testCasesCommon4D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParamsCommon4D));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_4D, StridedSliceLayerCPUTest, StridedSliceParamsCommon4D, StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesBlocked4D = {
        StridedSliceSpecificParams{ { 1, 16, 32, 32 }, { 0, 0, 5, 4 }, { 1, 16, 28, 27 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 10, 10 }, { 0, 16, 0, 0 }, { 1, 32, 10, 10 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 16, 32, 20 }, { 0, 0, 10, 0 }, { 1, 16, 20, 10 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 32, 32 }, { 0, 0, 20, 20 }, { 1, 32, 25, 25 }, { 1, 1, 1, 1 },
                                    { 0, 1, 0, 0 }, { 0, 1, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 48, 32, 32 }, { 0, 16, 0, 20 }, { 1, 32, 32, 30 }, { 1, 1, 1, 2 },
                                    { 1, 0, 1, 0 }, { 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 32, 20 }, { 0, 16, 2, 10 }, { 1, 32, 32, 20 }, { 1, 1, 2, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 32, 20 }, { 0, 16, 0, 0 }, { 2, 64, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 32, 20 }, { 0, 32, 0, 0 }, { 2, 50, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 32, 20 }, { 0, 0, 0, 0 }, { 2, 12, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 64, 32, 20 }, { 0, -16, 0, 10 }, { 2, 100, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 32, 20 }, { 0, -16, 0, 0 }, { 2, -4, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 32, 20 }, { 0, -32, 0, 0 }, { 2, -12, 32, 20 }, { 1, 1, 1, 1 },
                                    { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 32, 20 }, { 0, 10 }, { 0, 20 }, { 1, 1 },
                                    { 1, 0 }, { 1, 0 },  { },  { },  { 1, 0 } },
        StridedSliceSpecificParams{ { 2, 32, 32, 20 }, { 0, 16, 0 }, { 2, 32, 0 }, { 1, 1, 1 },
                                    { 1, 0, 1 }, { 1, 1, 1 },  { },  { },  { 0, 0, 1 } },
};

const std::vector<CPUSpecificParams> CPUParamsBlocked4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
};

const auto StridedSliceParamsBlocked4D = ::testing::Combine(
        ::testing::ValuesIn(testCasesBlocked4D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParamsBlocked4D));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Blocked_4D, StridedSliceLayerCPUTest, StridedSliceParamsBlocked4D, StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesCommon5D = {
        StridedSliceSpecificParams{ { 1, 5, 20, 32, 32 }, { 0, 2, 0, 5, 4 }, { 1, 4, 5, 28, 27 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 20, 32, 20 }, { 0, 0, 10, 0, 0 }, { 1, 5, 20, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 20, 32, 20 }, { 0, 1, 10, 0, 0 }, { 1, 3, 20, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 20, 32, 32 }, { 0, 0, 0, 20, 20 }, { 1, 5, 20, 30, 26 }, { 1, 1, 1, 2, 2 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 20, 32, 32 }, { 0, 0, 10, 0, 20 }, { 1, 2, 20, 30, 30 }, { 1, 1, 2, 1, 1 },
                                    { 0, 0, 0, 0, 1 }, { 0, 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 20, 32, 32 }, { 0, 0, 2, 10, 0 }, { 1, 5, 10, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 1, 1, 0 }, { 0, 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 5, 20, 32, 32 }, { 0, 1, 0, 10, 0 }, { 1, 5, 20, 32, 32 }, { 1, 1, 1, 1, 1 },
                                    { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 5, 20, 32, 32 }, { 0, 0, 0, 0, 0 }, { 1, 5, 10, 16, 16 }, { 1, 1, 2, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 8, 20, 32, 32 }, { 0, 2, 0, 0, 0 }, { 1, 8, 10, 16, 16 }, { 1, 2, 1, 1, 2 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 8, 20, 32, 32 }, { 0, 2, 0, 0, 16 }, { 2, 8, 20, 32, 32 }, { 1, 2, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 8, 10, 10, 10 }, { 0, 5 }, { 0, 10 }, { 1, 1 },
                                    { 1, 0 }, { 1, 0 },  { },  { },  { 1, 0 } },
        StridedSliceSpecificParams{ { 2, 8, 10, 10, 10 }, { 0, 0, 5 }, { 0, 0, 10 }, { 1, 1, 1 },
                                    { 1, 1, 0 }, { 1, 1, 0 },  { },  { },  { 0, 1, 0 } },
        StridedSliceSpecificParams{ { 2, 8, 10, 10, 10 }, { 0, 2, 0 }, { 2, 8, 0 }, { 1, 1, 1 },
                                    { 1, 0, 1 }, { 1, 1, 1 },  { },  { },  { 0, 0, 1 } }
};

const std::vector<CPUSpecificParams> CPUParamsCommon5D = {
        cpuParams_ncdhw,
        cpuParams_ndhwc,
};

const auto StridedSliceParamsCommon5D = ::testing::Combine(
        ::testing::ValuesIn(testCasesCommon5D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParamsCommon5D));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_5D, StridedSliceLayerCPUTest, StridedSliceParamsCommon5D, StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceSpecificParams> testCasesBlocked5D = {
        StridedSliceSpecificParams{ { 1, 16, 20, 32, 32 }, { 0, 0, 0, 5, 4 }, { 1, 16, 5, 28, 27 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 32, 20, 32, 20 }, { 0, 0, 10, 0, 0 }, { 1, 16, 20, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 16, 20, 32, 20 }, { 0, 0, 10, 0, 0 }, { 1, 16, 20, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 16, 20, 32, 32 }, { 0, 0, 0, 20, 20 }, { 1, 16, 20, 30, 26 }, { 1, 1, 1, 2, 2 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 16, 20, 32, 32 }, { 0, 0, 10, 0, 20 }, { 1, 16, 20, 30, 30 }, { 1, 1, 2, 1, 1 },
                                    { 0, 0, 0, 0, 1 }, { 0, 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 16, 20, 32, 32 }, { 0, 0, 2, 10, 0 }, { 1, 16, 10, 32, 20 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 1, 1, 0 }, { 0, 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 1, 16, 20, 32, 32 }, { 0, 0, 0, 10, 0 }, { 1, 8, 20, 32, 32 }, { 1, 1, 1, 1, 1 },
                                    { 0, 1, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 16, 20, 32, 32 }, { 0, 0, 0, 0, 0 }, { 1, 16, 10, 16, 16 }, { 1, 1, 2, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 0, 0, 0, 0 }, { 1, 25, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 16, 0, 0, 0 }, { 1, 25, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 16, 0, 0, 0 }, { 1, 64, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 0, 0, 0, 0 }, { 2, 25, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 0, 0, 0, 0 }, { 2, 60, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 32, 0, 0, 0 }, { 2, 40, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 2, 64, 20, 10, 10 }, { 0, 16, 0, 0, 0 }, { 2, 64, 20, 10, 10 }, { 1, 1, 1, 1, 1 },
                                    { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } }
};

const std::vector<CPUSpecificParams> CPUParamsBlocked5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
};

const auto StridedSliceParamsBlocked5D = ::testing::Combine(
        ::testing::ValuesIn(testCasesBlocked5D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config),
        ::testing::ValuesIn(CPUParamsBlocked5D));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Blocked_5D, StridedSliceLayerCPUTest, StridedSliceParamsBlocked5D, StridedSliceLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions

