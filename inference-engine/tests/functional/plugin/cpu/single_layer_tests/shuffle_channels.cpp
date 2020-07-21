// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/shuffle_channels.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::ShuffleChannelsLayerTestParams,
        CPUSpecificParams> ShuffleChannelsLayerCPUTestParamsSet;

class ShuffleChannelsLayerCPUTest : public testing::WithParamInterface<ShuffleChannelsLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::ShuffleChannelsLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ShuffleChannelsLayerTest::getTestCaseName(
                     testing::TestParamInfo<LayerTestsDefinitions::ShuffleChannelsLayerTestParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::ShuffleChannelsLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        LayerTestsDefinitions::ShuffleChannelsSpecificParams shuffleChannelsParams;
        std::vector<size_t> inputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(shuffleChannelsParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        int axis, group;
        std::tie(axis, group) = shuffleChannelsParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto shuffleChannels = std::dynamic_pointer_cast<ngraph::opset3::ShuffleChannels>(
                ngraph::builder::makeShuffleChannels(paramOuts[0], axis, group));
        shuffleChannels->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(shuffleChannels)};
        function = std::make_shared<ngraph::Function>(results, params, "shuffleChannels");
    }
};

TEST_P(ShuffleChannelsLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "ShuffleChannels");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDeviceAxis1() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx512"}, "jit_avx512_FP32"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP32
};

const std::vector<int> axes = {-4, -2, -1, 0, 2, 3};
const std::vector<int> groups = {1, 2, 3};

const auto shuffleChannelsParams4D = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(groups)
);

const auto shuffleChannelsParams4DAxis1 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>({1, -3})),
        ::testing::ValuesIn(groups)
);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({6, 6, 6, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels4DAxis1, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams4DAxis1,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({6, 6, 6, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDeviceAxis1())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
