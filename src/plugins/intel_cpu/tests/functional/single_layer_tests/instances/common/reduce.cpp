// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/reduce.hpp"
#include "shared_test_classes/single_layer/reduce_ops.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Reduce {

std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_dynamic_3dims = {
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_dynamic_2dims = {
    {{{2, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_5D = {
    {{{}, {{2, 19, 2, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_6D = {
    {{{}, {{2, 19, 2, 2, 2, 2}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_Int32 = {
    {{{}, {{2, 19, 2, 3}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_SmallChannel = {
    {{{}, {{2, 3, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_SingleBatch = {
    {{{}, {{1, 19, 2, 9}}}},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
//NHWC layout is disabled on ARM due to accuracy issue: https://github.com/ARM-software/ComputeLibrary/issues/1044
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
#endif
};

/* ================================ 1.1 No fusion - Arithmetic ================================ */
const auto params_OneAxis = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::ValuesIn(opTypes()),
            testing::ValuesIn(keepDims()),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_OneAxis_dynamic = testing::Combine(
        testing::Combine(
            testing::Values(1),                                 // ACL supports reduce against static dims only
            testing::ValuesIn(opTypes()),
            testing::ValuesIn(keepDims()),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dynamic_3dims)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND()),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_4D_dynamic = testing::Combine(
        testing::Combine(
                testing::Values(std::vector<int>{0, 1}),           // ACL supports reduce against static dims only
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dynamic_2dims)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_Int32 = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::ValuesIn(keepDims()),
            testing::ValuesIn(reductionTypesInt32()),
            testing::Values(ElementType::i32),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_Int32)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_CPU,
        ReduceCPULayerTest,
        params_OneAxis,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_dynamic_CPU,
        ReduceCPULayerTest,
        params_OneAxis_dynamic,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_dynamic_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_dynamic,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Int32_CPU,
        ReduceCPULayerTest,
        params_Int32,
        ReduceCPULayerTest::getTestCaseName
);

} // namespace Reduce
} // namespace CPULayerTestsDefinitions