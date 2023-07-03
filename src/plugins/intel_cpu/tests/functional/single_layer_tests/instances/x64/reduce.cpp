// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/reduce.hpp"
#include "shared_test_classes/single_layer/reduce_ops.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;


namespace CPULayerTestsDefinitions {
namespace Reduce {
namespace {



std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
};

const std::vector<std::vector<int>> axes5D = {
        {2, 4},
        {0, 2, 4},
        {1, 2, 4},
        {0, 1, 2, 3, 4},
};

const std::vector<std::vector<int>> axes6D = {
        {5},
        {4, 5},
        {3, 4, 5},
        {2, 3, 4, 5},
        {1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5}
};

const std::vector<std::vector<int>> axesNDFusing = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
};

const std::vector<std::vector<int>> axes5DFusing = {
        {2, 4},
        {0, 2, 4},
};

const std::vector<std::vector<int>> axesHW = {
        {2, 3}
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_4D = {
        CPUSpecificParams({nChw16c}, {}, {}, {}),
        CPUSpecificParams({nhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_5D = {
        CPUSpecificParams({nCdhw16c}, {}, {}, {}),
        CPUSpecificParams({ndhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_NHWC_4D = {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {})
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
    ngraph::helpers::ReductionType::LogicalOr,
     ngraph::helpers::ReductionType::LogicalAnd
};

const std::vector<ngraph::helpers::ReductionType> reductionTypesFusing = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<fusingSpecificParams> fusingParamsSet {
        /* activations */
        fusingSwish,
        /* FQ */
        fusingFakeQuantizePerChannelRelu,
        fusingFakeQuantizePerTensorRelu,
        /* another patterns */
        fusingScaleShift
};

// Exclude cases of fusingFakeQuantizePerChannelRelu, where FQ for non-1 channel fallbacks
// to decomposed ngraph reference implementation, so such fusing tests are N/A
const std::vector<fusingSpecificParams> fusingParamsSet_KeepNoDims {
        /* activations */
        fusingSwish,

        /* FQ */
        fusingFakeQuantizePerTensorRelu,
        /* another patterns */
        fusingScaleShift
};

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND()),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims()),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_NHWC_SmallChannel = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesHW),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SmallChannel_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec));

const auto params_SingleBatch = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes()),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SingleBatch_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_6D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_6D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_NHWC_SmallChannel_CPU,
        ReduceCPULayerTest,
        params_NHWC_SmallChannel,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_SingleBatch_CPU,
        ReduceCPULayerTest,
        params_SingleBatch,
        ReduceCPULayerTest::getTestCaseName
);

/* ================================ 1.2 No fusion - Logical ================================ */
const auto params_OneAxis_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::ValuesIn(opTypes()),
            testing::ValuesIn(keepDims()),
            testing::ValuesIn((reductionLogicalTypes)),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND()),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND()),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn((reductionLogicalTypes)),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn((reductionLogicalTypes)),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims()),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_Logical_CPU,
        ReduceCPULayerTest,
        params_OneAxis_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_6D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_6D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

/* ================================ 2.1 Fusion - KeepDims ================================ */
const auto params_OneAxis_fusing = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::ValuesIn(opTypes()),
            testing::Values(true),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND()),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_CPU,
        ReduceCPULayerTest,
        params_OneAxis_fusing,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_fusing_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_fusing,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_fusing_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_fusing,
        ReduceCPULayerTest::getTestCaseName
);

/* ================================ 2.2 Fusion - KeepNoDims ================================ */
const auto params_OneAxis_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::ValuesIn(opTypes()),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

const auto params_MultiAxis_4D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesNDFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

const auto params_MultiAxis_5D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5DFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_OneAxis_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);

} // namespace
} // namespace Reduce
} // namespace CPULayerTestsDefinitions