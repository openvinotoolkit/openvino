// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/shuffle_channels.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::shuffleChannelsLayerTestParamsSet,
        CPUSpecificParams> ShuffleChannelsLayerCPUTestParamsSet;

class ShuffleChannelsLayerCPUTest : public testing::WithParamInterface<ShuffleChannelsLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::shuffleChannelsLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ShuffleChannelsLayerTest::getTestCaseName(
                     testing::TestParamInfo<LayerTestsDefinitions::shuffleChannelsLayerTestParamsSet>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::shuffleChannelsLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        LayerTestsDefinitions::shuffleChannelsSpecificParams shuffleChannelsParams;
        std::vector<size_t> inputShape;
        Precision netPrecision;
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

        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType.push_back('_');
        selectedType += netPrecision.name();
    }
};

TEST_P(ShuffleChannelsLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "ShuffleChannels");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice4D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice5D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice4DBlock() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice5DBlock() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const auto shuffleChannelsParams4D = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>{-4, -2, 0, 1, 2, 3}),
        ::testing::ValuesIn(std::vector<int>{1, 2, 4, 8})
);

const auto shuffleChannelsParams5D = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>{-5, -1, 0, 1, 2, 3, 4}),
        ::testing::ValuesIn(std::vector<int>{1, 2, 3, 6})
);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({16, 24, 32, 40})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice4D())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels5D, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams5D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({12, 18, 12, 18, 24})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice5D())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

const auto shuffleChannelsParams4DBlock = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>{-4, -2, -1, 0, 2, 3}),
        ::testing::ValuesIn(std::vector<int>{1, 2, 4, 8})
);

const auto shuffleChannelsParams5DBlock = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>{-5, -2, -1, 0, 2, 3, 4}),
        ::testing::ValuesIn(std::vector<int>{1, 2, 3, 6})
);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels4DBlock, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams4DBlock,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({40, 32, 24, 16})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice4DBlock())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels5DBlock, ShuffleChannelsLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                shuffleChannelsParams5DBlock,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({18, 12, 18, 12, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice5DBlock())),
        ShuffleChannelsLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
