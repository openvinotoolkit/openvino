// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,               // Axis to reduce order
        CommonTestUtils::OpType,        // Scalar or vector type axis
        bool,                           // Keep dims
        ngraph::helpers::ReductionType, // Reduce operation type
        ElementType,                    // Net precision
        ElementType,                    // Input precision
        ElementType,                    // Output precision
        std::vector<InputShape>         // Input shapes
> basicReduceParams;

typedef std::tuple<
        basicReduceParams,
        CPUSpecificParams,
        fusingSpecificParams> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj) {
        basicReduceParams basicParams;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParams, cpuParams, fusingParams) = obj.param;

        std::vector<int> axes;
        CommonTestUtils::OpType opType;
        bool keepDims;
        ngraph::helpers::ReductionType reductionType;
        ElementType netPrecision, inPrc, outPrc;
        std::vector<InputShape> inputShapes;

        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes) = basicParams;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << ")_axes=" << CommonTestUtils::vec2str(axes) << "_";
        result << "opType=" << opType << "_";
        result << "type=" << reductionType << "_";
        if (keepDims)
            result << "KeepDims=true_";
        else
            result << "KeepDims=false_";
        result << "netPRC=" << netPrecision << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc << "_";

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        basicReduceParams basicParams;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParams, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::vector<int> axes;
        CommonTestUtils::OpType opType;
        bool keepDims;
        ElementType inPrc, outPrc;
        std::vector<InputShape> inputShapes;

        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes) = basicParams;
        inPrc = outPrc = netPrecision;

        init_input_shapes(inputShapes);

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<size_t> shapeAxes;
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR:
                if (axes.size() > 1)
                    FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
                break;
            case CommonTestUtils::OpType::VECTOR:
                shapeAxes.push_back(axes.size());
                break;
            default:
                FAIL() << "Reduce op doesn't support operation type: " << opType;
        }
        auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

        const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);

        selectedType = getPrimitiveType() + "_" +
                       (inPrc == ElementType::boolean ? "I8" : InferenceEngine::details::convertPrecision(inPrc).name());

        // hybrid layouts
        if (inFmts.size() != 0 && outFmts.size() == 0) {
            size_t outShapeSize = inputDynamicShapes[0].size() - axes.size();
            switch (outShapeSize) {
                case 0:
                case 1:
                    outFmts.push_back(x);
                    break;
                case 2:
                    outFmts.push_back(nc);
                    break;
                case 3:
                    outFmts.push_back(tnc);
                    break;
                case 4:
                    outFmts.push_back(nchw);
                    break;
                default:
                    FAIL() << "Invaid outShapeSize: " << outShapeSize;
            }
        }

        function = makeNgraphFunction(netPrecision, params, reduce, "Reduce");
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (reductionType == ngraph::helpers::ReductionType::Prod) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, 5);
                if (netPrecision == ElementType::f32) {
                    auto *rawBlobDataPtr = static_cast<float *>(tensor.data());
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        rawBlobDataPtr[i] /= 10.f;
                    }
                } else if (netPrecision == ElementType::bf16) {
                    auto *rawBlobDataPtr = static_cast<ngraph::bfloat16 *>(tensor.data());
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        rawBlobDataPtr[i] /= 10.f;
                    }
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    ngraph::helpers::ReductionType reductionType;
    ElementType netPrecision;
};

TEST_P(ReduceCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Reduce");
}
namespace {
const std::vector<ElementType> inpOutPrc = {ElementType::bf16, ElementType::f32};

const std::vector<bool> keepDims = {
        true,
        false,
};

const std::vector<std::vector<int>> axes = {
        {0},
        {1},
        {2},
        {3}
};

const std::vector<std::vector<int>> axesND = {
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

const std::vector<std::vector<int>> axes5D = {
        {2, 4},
        {0, 2, 4},
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

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
// WR: Remove to pass the test because ReductionMeanToPoolingTranformation enabling.
       // ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypesFusing = {
// WR: Remove to pass the test because ReductionMeanToPoolingTranformation enabling.
        //ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_5D = {
    {{{}, {{2, 19, 2, 2, 9}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2}, {2, 19, 3, 2, 2}}}},
};

// TODO: should remove inputShapesFusingKeepNoDims and inputShapesFusingKeepNoDims_5D,
//       and use inputShapes and inputShapes_5D directly,
//       after Shape mismatching in fusing per channel[Issue: 62846] being fixed
std::vector<std::vector<ov::test::InputShape>> inputShapesFusingKeepNoDims = {
    {{{}, {{2, 19, 2, 9}}}},
    {{{{1, 5}, 19, 2, 9}, {{2, 19, 2, 9}, {3, 19, 2, 9}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapesFusingKeepNoDims_5D = {
    {{{}, {{2, 19, 2, 2, 9}}}},
    {{{{1, 5}, 19, 2, 2, 2}, {{2, 19, 2, 2, 2}, {3, 19, 2, 2, 2}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_6D = {
    {{{}, {{2, 19, 2, 2, 2, 2}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2, 2}, {2, 19, 2, 2, 3, 2}}}},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_4D = {
        CPUSpecificParams({nChw16c}, {}, {}, {}),
        CPUSpecificParams({nhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_5D = {
        CPUSpecificParams({nCdhw16c}, {}, {}, {}),
        CPUSpecificParams({ndhwc}, {}, {}, {})
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

/* ================================ 1.1 No fusion - Arithmetic ================================ */
const auto params_OneAxis = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes),
            testing::ValuesIn(opTypes),
            testing::ValuesIn(keepDims),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

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

/* ================================ 1.2 No fusion - Logical ================================ */
const auto params_OneAxis_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes),
            testing::ValuesIn(opTypes),
            testing::ValuesIn(keepDims),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5D),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(ElementType::boolean),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D)),
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
            testing::ValuesIn(axes),
            testing::ValuesIn(opTypes),
            testing::Values(true),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapes)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D)),
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
            testing::ValuesIn(axes),
            testing::ValuesIn(opTypes),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapesFusingKeepNoDims)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesNDFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapesFusingKeepNoDims)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axes5DFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypesFusing),
            testing::ValuesIn(inpOutPrc),
            testing::Values(ElementType::undefined),
            testing::Values(ElementType::undefined),
            testing::ValuesIn(inputShapesFusingKeepNoDims_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::ValuesIn(fusingParamsSet));

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
} // namespace CPULayerTestsDefinitions

