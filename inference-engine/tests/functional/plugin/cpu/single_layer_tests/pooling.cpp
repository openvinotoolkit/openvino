// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        poolLayerTestParamsSet,
        CPUSpecificParams
> poolLayerCpuTestParamsSet;

class PoolingLayerCPUTest : public testing::WithParamInterface<poolLayerCpuTestParamsSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
        poolLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << PoolingLayerTest::getTestCaseName(testing::TestParamInfo<poolLayerTestParamsSet>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        poolLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        poolSpecificParams poolParams;
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (outPrc == Precision::UNSPECIFIED) {
            outPrc = inPrc;
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

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

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Combine(
                                ::testing::ValuesIn(paramsMax4D),
                                ::testing::Values(Precision::FP32),
                                ::testing::ValuesIn(inpOutPrecision),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 64, 64})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg4D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({1, 4, 64, 64})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg4D_RefOnly),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({1, 4, 64, 64})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(ref)),
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

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_CPU_5D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsMax5D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({1, 3, 16, 32, 32})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg5D),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({1, 4, 32, 32, 32})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(paramsAvg5D_RefOnly),
                                        ::testing::Values(Precision::FP32),
                                        ::testing::ValuesIn(inpOutPrecision),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({1, 4, 16, 16, 16})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(ref)),
                        PoolingLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions