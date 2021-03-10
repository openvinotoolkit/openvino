// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/normalize_l2.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

using NormalizeL2LayerCPUTestParamSet = std::tuple<NormalizeL2LayerTestParams,
                                                   CPUSpecificParams,
                                                   fusingSpecificParams>;

class NormalizeL2LayerCPUTest : public testing::WithParamInterface<NormalizeL2LayerCPUTestParamSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2LayerCPUTestParamSet> obj) {
        NormalizeL2LayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << NormalizeL2LayerTest::getTestCaseName(testing::TestParamInfo<NormalizeL2LayerTestParams>(basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
        NormalizeL2LayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::vector<int64_t> axes;
        float eps;
        op::EpsMode eps_mode;
        SizeVector inputShapes;
        std::tie(axes, eps, eps_mode, inputShapes, inPrc, targetDevice) = basicParamsSet;

        outPrc = inPrc;
        auto netPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto params = builder::makeParams(netPrc, {inputShapes});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
        auto normalize = builder::makeNormalizeL2(paramOuts[0], axes, eps, eps_mode);

        function = makeNgraphFunction(netPrc, params, normalize, "Normalize");

        selectedType = "unknown_" + std::string(inPrc.name());
        threshold = 0.015f;
    }
};

TEST_P(NormalizeL2LayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "NormalizeL2");
}

namespace {

/* ============= Common params ============= */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingMultiplyPerTensor,
        fusingMultiplyPerChannel,
        fusingAddPerTensor,
        fusingAddPerChannel,
        fusingSubtractPerTensor,
        fusingSubtractPerChannel,
        fusingDividePerTensor,
        fusingDividePerChannel,
        fusingPReluPerChannel,
        fusingPReluPerTensor,
        fusingRelu,
        fusingGelu,
        fusingReluScaleShift
};

const float epsilon = 1e-4f;
const op::EpsMode epsMode = op::EpsMode::ADD;
const std::vector<Precision> netPrecisions = {
    Precision::FP32,
    Precision::BF16
};

/* ============= 2D ============= */
const std::vector<std::vector<size_t>> inputShape_2D = {
    {2, 3},
    {2, 16},
    {3, 20}
};

const std::vector<std::vector<int64_t>> axes_2D = {
    {1}
};

const auto normalizeParams_2D = ::testing::Combine(::testing::ValuesIn(axes_2D),
                                                   ::testing::Values(epsilon),
                                                   ::testing::Values(epsMode),
                                                   ::testing::ValuesIn(inputShape_2D),
                                                   ::testing::ValuesIn(netPrecisions),
                                                   ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto testParams_2D = ::testing::Combine(normalizeParams_2D,
                                              ::testing::Values(CPUSpecificParams{}),
                                              ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_CASE_P(smoke_2D, NormalizeL2LayerCPUTest, testParams_2D, NormalizeL2LayerCPUTest::getTestCaseName);

/* ============= 3D ============= */
const std::vector<std::vector<size_t>> inputShape_3D = {
    {2, 3, 4},
    {2, 16, 6},
    {3, 20, 10}
};

const std::vector<std::vector<int64_t>> axes_3D = {
    {1, 2},
    {1}
};

const auto normalizeParams_3D = ::testing::Combine(::testing::ValuesIn(axes_3D),
                                                   ::testing::Values(epsilon),
                                                   ::testing::Values(epsMode),
                                                   ::testing::ValuesIn(inputShape_3D),
                                                   ::testing::ValuesIn(netPrecisions),
                                                   ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto testParams_3D = ::testing::Combine(normalizeParams_3D,
                                              ::testing::Values(CPUSpecificParams{}),
                                              ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_CASE_P(smoke_3D, NormalizeL2LayerCPUTest, testParams_3D, NormalizeL2LayerCPUTest::getTestCaseName);

/* ============= 4D ============= */
const std::vector<std::vector<size_t>> inputShape_4D = {
    {2, 3, 4, 4},
    {2, 16, 7, 6},
    {3, 20, 2, 10}
};

const std::vector<std::vector<int64_t>> axes_4D = {
    {1, 2, 3},
    {1}
};

std::vector<CPUSpecificParams> getCPUSpecificParams() {
    std::vector<CPUSpecificParams> result;
    result.push_back(CPUSpecificParams({nchw}, {nchw}, {}, {}));
    if (with_cpu_x86_sse42()) {
        result.push_back(CPUSpecificParams({nhwc}, {nhwc}, {}, {}));
        if (with_cpu_x86_avx512f()) {
            result.push_back(CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}));
        } else if (with_cpu_x86_avx2()) {
            result.push_back(CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}));
        }
    }
    return result;
}

const auto normalizeParams_4D = ::testing::Combine(::testing::ValuesIn(axes_4D),
                                                   ::testing::Values(epsilon),
                                                   ::testing::Values(epsMode),
                                                   ::testing::ValuesIn(inputShape_4D),
                                                   ::testing::ValuesIn(netPrecisions),
                                                   ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto testParams_4D = ::testing::Combine(normalizeParams_4D,
                                              ::testing::ValuesIn(getCPUSpecificParams()),
                                              ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_CASE_P(smoke_4D, NormalizeL2LayerCPUTest, testParams_4D, NormalizeL2LayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
