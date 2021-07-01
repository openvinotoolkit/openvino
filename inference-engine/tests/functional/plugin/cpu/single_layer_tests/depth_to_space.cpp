// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/depth_to_space.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::depthToSpaceParamsTuple,
        CPUSpecificParams
> DepthToSpaceLayerCPUTestParamSet;

class DepthToSpaceLayerCPUTest : public testing::WithParamInterface<DepthToSpaceLayerCPUTestParamSet>,
                        virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceLayerCPUTestParamSet> obj) {
        LayerTestsDefinitions::depthToSpaceParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::DepthToSpaceLayerTest::getTestCaseName(
                testing::TestParamInfo<LayerTestsDefinitions::depthToSpaceParamsTuple>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        LayerTestsDefinitions::depthToSpaceParamsTuple basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<size_t> inputShape;
        DepthToSpace::DepthToSpaceMode mode;
        std::size_t blockSize;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, mode, blockSize, targetDevice) = basicParamsSet;

        inPrc = outPrc = netPrecision;
        selectedType = getPrimitiveType() + "_" + inPrc.name();
        auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(inPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto d2s = ngraph::builder::makeDepthToSpace(paramOuts[0], mode, blockSize);
        d2s->get_rt_info() = getCPUInfo();
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(d2s)};
        function = std::make_shared<ngraph::Function>(results, params, "DepthToSpace");
    }
};

TEST_P(DepthToSpaceLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "DepthToSpace");
}

namespace {

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {"jit_avx512"}, {"jit_avx512"}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, {"jit_avx512"}};

const auto cpuParams_nChw8c_avx2 = CPUSpecificParams {{nChw8c}, {nChw8c}, {"jit_avx2"}, {"jit_avx2"}};
const auto cpuParams_nCdhw8c_avx2 = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, {"jit_avx2"}};

const auto cpuParams_nChw8c_sse42 = CPUSpecificParams {{nChw8c}, {nChw8c}, {"jit_sse42"}, {"jit_sse42"}};
const auto cpuParams_nCdhw8c_sse42 = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, {"jit_sse42"}};

const auto cpuParams_nhwc_avx2 = CPUSpecificParams {{nhwc}, {nhwc}, {"jit_avx2"}, {"jit_avx2"}};
const auto cpuParams_ndhwc_avx2 = CPUSpecificParams {{ndhwc}, {ndhwc}, {"jit_avx2"}, {"jit_avx2"}};

const auto cpuParams_nhwc_sse42 = CPUSpecificParams {{nhwc}, {nhwc}, {"jit_sse42"}, {"jit_sse42"}};
const auto cpuParams_ndhwc_sse42 = CPUSpecificParams {{ndhwc}, {ndhwc}, {"jit_sse42"}, {"jit_sse42"}};

const auto cpuParams_nhwc_ref = CPUSpecificParams {{nhwc}, {nhwc}, {"ref_any"}, {"ref_any"}};
const auto cpuParams_ndhwc_ref = CPUSpecificParams {{ndhwc}, {ndhwc}, {"ref_any"}, {"ref_any"}};


const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

const std::vector<DepthToSpace::DepthToSpaceMode> depthToSpaceModes = {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

const std::vector<std::vector<size_t >> inputShapesBS2_4D = {
        {1, 64, 1, 1}, {1, 64, 1, 3}, {1, 128, 3, 3}, {2, 128, 1, 1}, {1, 192, 2, 2}, {2, 256, 2, 3}, {1, 512, 2, 1}
};

const std::vector<std::vector<size_t >> inputShapesBS3_4D = {
        {1, 27, 1, 1}, {1, 27, 2, 3}, {1, 18, 2, 3}, {3, 18, 1, 1}, {2, 18, 3, 1}
};

const std::vector<CPUSpecificParams> CPUParamsBS2_4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c_avx2,
        cpuParams_nChw8c_sse42,
        cpuParams_nhwc_avx2,
        cpuParams_nhwc_sse42,
        cpuParams_nhwc_ref,
};

const auto depthToSpaceBS2_4DParams = testing::Combine(
        testing::Combine(
                testing::ValuesIn(inputShapesBS2_4D),
                testing::ValuesIn(inputPrecisions),
                testing::ValuesIn(depthToSpaceModes),
                testing::Values(1, 2),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS2_4D))
);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS2_4D, DepthToSpaceLayerCPUTest, depthToSpaceBS2_4DParams, DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParamsBS3_4D = {
        cpuParams_nhwc_avx2,
        cpuParams_nhwc_sse42,
        cpuParams_nhwc_ref,
};

const auto depthToSpaceBS3_4DParams = testing::Combine(
        testing::Combine(
                testing::ValuesIn(inputShapesBS3_4D),
                testing::ValuesIn(inputPrecisions),
                testing::ValuesIn(depthToSpaceModes),
                testing::Values(1, 3),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS3_4D))
);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS3_4D, DepthToSpaceLayerCPUTest, depthToSpaceBS3_4DParams, DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<std::vector<size_t >> inputShapesBS2_5D = {
        {1, 128, 1, 1, 1}, {1, 128, 2, 1, 2}, {1, 256, 2, 1, 3}, {2, 256, 3, 1, 1}, {1, 384, 1, 2, 2}, {2, 512, 1, 2, 1}
};

const std::vector<std::vector<size_t >> inputShapesBS3_5D = {
        {1, 54, 1, 1, 1}, {1, 54, 2, 1, 2}, {3, 54, 1, 1, 1}, {2, 54, 3, 1, 2}, {1, 54, 3, 2, 2}
};

const std::vector<CPUSpecificParams> CPUParamsBS2_5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c_avx2,
        cpuParams_nCdhw8c_sse42,
        cpuParams_ndhwc_avx2,
        cpuParams_ndhwc_sse42,
        cpuParams_ndhwc_ref,
};

const auto depthToSpaceBS2_5DParams = testing::Combine(
        testing::Combine(
                testing::ValuesIn(inputShapesBS2_5D),
                testing::ValuesIn(inputPrecisions),
                testing::ValuesIn(depthToSpaceModes),
                testing::Values(1, 2),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS2_5D))
);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS2_5D, DepthToSpaceLayerCPUTest, depthToSpaceBS2_5DParams, DepthToSpaceLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParamsBS3_5D = {
        cpuParams_ndhwc_avx2,
        cpuParams_ndhwc_sse42,
        cpuParams_ndhwc_ref,
};

const auto depthToSpaceBS3_5DParams = testing::Combine(
        testing::Combine(
                testing::ValuesIn(inputShapesBS3_5D),
                testing::ValuesIn(inputPrecisions),
                testing::ValuesIn(depthToSpaceModes),
                testing::Values(1, 3),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParamsBS3_5D))
);

INSTANTIATE_TEST_SUITE_P(smoke_CPUDepthToSpaceBS3_5D, DepthToSpaceLayerCPUTest, depthToSpaceBS3_5DParams, DepthToSpaceLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
