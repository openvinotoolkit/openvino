// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/reduce.hpp"
#include "shared_test_classes/single_layer/reduce_ops.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;


namespace CPULayerTestsDefinitions {
namespace Reduce {
namespace {

std::vector<std::vector<ov::test::InputShape>> inputShapes_dyn = {
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_5D_dyn = {
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2}, {2, 19, 3, 2, 2}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_6D_dyn = {
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2, 2}, {2, 19, 2, 2, 3, 2}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_Int32_dyn = {
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 3}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_SmallChannel_dyn = {
    {{{{1, 5}, 3, {1, 5}, {1, 10}}, {{2, 3, 2, 2}, {2, 3, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_SingleBatch_dyn = {
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{1, 19, 2, 2}, {1, 19, 2, 9}}}},
};

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

// This custom subgraph is used to test post-ops fusing case with U8/I8 precision on output,
// since Transpose prevents dequantization part to be fused back into Reduce
const auto fusingFakeQuantizeTranspose = fusingSpecificParams{std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape(cfg.input->get_output_partial_shape(0).size(), 1);
            const auto fakeQuantize = ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
            std::vector<size_t> order(newShape.size());
            std::iota(order.begin(), order.end(), 0);
            auto last = order[order.size() - 1];
            order.pop_back();
            order.insert(order.begin(), last);
            const auto transpose = ngraph::builder::subgraph::Transpose(order);
            return ngraph::builder::subgraph::makeTranspose(fakeQuantize, transpose);
        }, "FakeQuantize(PerTensor)"}}), {"FakeQuantize"}};

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

const std::vector<fusingSpecificParams> fusingParamsSet_LowPrecision {
        fusingFakeQuantizeTranspose
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
            testing::ValuesIn(inputShapes_dyn)),
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
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_4D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND()),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_5D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes()),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_6D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::ValuesIn(keepDims()),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_Int32 = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes()),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::ValuesIn(keepDims()),
            testing::ValuesIn(reductionTypesInt32()),
            testing::Values(ElementType::i32),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_Int32_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_NHWC_SmallChannel = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesHW),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SmallChannel_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const auto params_SingleBatch = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes()),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SingleBatch_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_CPU,
        ReduceCPULayerTest,
        params_OneAxis,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D,
        ReduceCPULayerTest::getTestCaseName
);

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
        smoke_Reduce_Int32_CPU,
        ReduceCPULayerTest,
        params_Int32,
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
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_4D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND()),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_5D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_4D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND()),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn((reductionLogicalTypes)),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_5D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn((reductionLogicalTypes)),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_6D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::ValuesIn(keepDims()),
                testing::ValuesIn((reductionLogicalTypes)),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D_dyn)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfigFP32()));

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
        testing::ValuesIn(fusingParamsSet),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_4D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND()),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet),
        testing::ValuesIn(additionalConfig()));

const auto params_MultiAxis_5D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::ValuesIn(fusingParamsSet),
        testing::ValuesIn(additionalConfig()));

const auto params_LowPrecision_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesNDFusing),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet_LowPrecision),
        testing::ValuesIn(additionalConfig()));

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

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_LowPrecision_fusing_CPU,
        ReduceCPULayerTest,
        params_LowPrecision_fusing,
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
        testing::ValuesIn(fusingParamsSet_KeepNoDims),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_4D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesNDFusing),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims),
        testing::ValuesIn(additionalConfigFP32()));

const auto params_MultiAxis_5D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5DFusing),
            testing::Values(ov::test::utils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc()),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D_dyn)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims),
        testing::ValuesIn(additionalConfigFP32()));

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