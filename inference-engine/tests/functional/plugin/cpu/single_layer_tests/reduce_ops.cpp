// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/reduce_ops.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

class ReduceOpsCPULayerTest : public LayerTestsDefinitions::ReduceOpsLayerTest, public CPUTestsBase {
};

TEST_P(ReduceOpsCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
    configuration.insert({PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO});

    std::string strExpectedPrc;
    if (Precision::BF16 == inPrc) {
        strExpectedPrc = "BF16";
    } else if (Precision::FP32 == inPrc) {
        strExpectedPrc = "FP32";
    }

    std::string isaType;
    if (with_cpu_x86_avx512f()) {
        isaType = "jit_avx512";
    } else if (with_cpu_x86_avx2()) {
        isaType = "jit_avx2";
    } else if (with_cpu_x86_sse42()) {
        isaType = "jit_sse42";
    } else {
        isaType = "ref";
    }
    selectedType = isaType + "_" + strExpectedPrc;

    auto ops = function->get_ordered_ops();
    std::string name = (*(++ops.rbegin()))->get_type_name();

    if ("ReduceLogicalAnd" == name) {
        name = "ReduceAnd";
    }
    if ("ReduceLogicalOr" == name) {
        name = "ReduceOr";
    }

    Run();
    CheckCPUImpl(executableNetwork, name);
}
namespace {
std::vector<Precision> inpOutPrc = {Precision::BF16, Precision::FP32};

const std::vector<bool> keepDims = {
        true,
        false,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<int>> axes = {
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3}
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
//        ngraph::helpers::ReductionType::Mean, //optimized out during the graph transformations
//        ngraph::helpers::ReductionType::Max, //optimized out during the graph transformations
//        ngraph::helpers::ReductionType::Sum, //optimized out during the graph transformations
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

const auto paramsOneAxis = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(opTypes),
        testing::Values(true, false),
        testing::ValuesIn(reductionTypes),
        testing::Values(InferenceEngine::Precision::FP32),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto paramsOneAxisLogical = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(opTypes),
        testing::Values(true, false),
        testing::ValuesIn(reductionLogicalTypes),
        testing::Values(InferenceEngine::Precision::BOOL),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);


const auto params_ReductionTypes = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(opTypes[1]),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionTypes),
        testing::Values(InferenceEngine::Precision::FP32),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto params_ReductionTypesLogical = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(opTypes[1]),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionLogicalTypes),
        testing::Values(InferenceEngine::Precision::BOOL),
        testing::ValuesIn(inpOutPrc),
        testing::ValuesIn(inpOutPrc),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceOneAxis_CPU,
        ReduceOpsCPULayerTest,
        paramsOneAxis,
        LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogicalOneAxis_CPU,
        ReduceOpsCPULayerTest,
        paramsOneAxisLogical,
        LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_Reduce_ReductionTypes_CPU,
        ReduceOpsCPULayerTest,
        params_ReductionTypes,
        LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogical_ReductionTypes_CPU,
        ReduceOpsCPULayerTest,
        params_ReductionTypesLogical,
        LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName
);
} // namespace
} // namespace CPULayerTestsDefinitions

