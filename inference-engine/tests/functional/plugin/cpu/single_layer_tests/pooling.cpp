// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        poolLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams
> poolLayerCpuTestParamsSet;

class PoolingLayerCPUTest : public testing::WithParamInterface<poolLayerCpuTestParamsSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
        poolLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << PoolingLayerTest::getTestCaseName(testing::TestParamInfo<poolLayerTestParamsSet>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() {
        poolLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

        poolSpecificParams poolParams;
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (outPrc == Precision::UNSPECIFIED) {
            outPrc = inPrc;
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }

        selectedType = selectedType + "_" + inPrc.name();

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        bool excludePad;
        std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(paramOuts[0],
                                                                             stride,
                                                                             padBegin,
                                                                             padEnd,
                                                                             kernel,
                                                                             roundingType,
                                                                             padType,
                                                                             excludePad,
                                                                             poolType);


        function = makeNgraphFunction(ngPrc, params, pooling, "Pooling");
    }
};

TEST_P(PoolingLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Pooling");
}

namespace {
const auto avx512 = CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
const auto avx = CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"};
const auto sse42 = CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};
const auto ref = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<CPUSpecificParams> vecCpuConfigs = {ref, sse42, avx, avx512};
const std::vector<Precision> inpOutPrecision = {Precision::FP32, Precision::BF16};

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{3, 4, 64, 64},
        std::vector<size_t>{2, 8, 8, 12},
        std::vector<size_t>{1, 16, 16, 12},
        std::vector<size_t>{1, 21, 8, 4},
        std::vector<size_t>{1, 32, 8, 8},
};

const std::vector<std::vector<size_t>> inputShapes5D = {
        std::vector<size_t>{1, 4, 16, 16, 16},
        std::vector<size_t>{2, 8, 8, 8, 8},
        std::vector<size_t>{2, 16, 12, 16, 20},
        std::vector<size_t>{1, 19, 16, 20, 8},
        std::vector<size_t>{1, 32, 16, 8, 12},
};

const std::vector<poolSpecificParams> paramsMax4D = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<poolSpecificParams> paramsAvg4D = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<poolSpecificParams> paramsAvg4D_RefOnly = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::ValuesIn(paramsMax4D),
                                ::testing::Values(Precision::FP32),
                                ::testing::ValuesIn(inpOutPrecision),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes4D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                        ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg4D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes4D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                                ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg4D_RefOnly),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes4D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(ref),
                                ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

const std::vector<poolSpecificParams> paramsMax5D = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 3, 4}, {2, 2, 2}, {1, 1, 1}, {1, 2, 3},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<poolSpecificParams> paramsAvg5D = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<poolSpecificParams> paramsAvg5D_RefOnly = {
        poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_5D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsMax5D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes5D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                                ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg5D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes5D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                                ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg5D_RefOnly),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes5D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(ref),
                                ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

/* === Fusing === */

const auto avx512_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"};
const auto avx512_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx512"}, "jit_avx512"};

const auto avx2_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"};
const auto avx2_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx2"}, "jit_avx2"};

const auto sse42_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"};
const auto sse42_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_sse42"}, "jit_sse42"};

const std::vector<CPUSpecificParams> vecCpuConfigsFusing_4D = {sse42_nhwc, avx2_nhwc, avx512_nhwc};
const std::vector<CPUSpecificParams> vecCpuConfigsFusing_5D = {sse42_ndhwc, avx2_ndhwc, avx512_ndhwc};

std::vector<fusingSpecificParams> fusingParamsSet {
    emptyFusingSpec,
    fusingFakeQuantizePerTensor,
    fusingFakeQuantizePerChannel,
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_I8, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg4D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::I8),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes4D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_4D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_I8, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg5D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::I8),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::ValuesIn(inputShapes5D),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_5D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        PoolingLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions