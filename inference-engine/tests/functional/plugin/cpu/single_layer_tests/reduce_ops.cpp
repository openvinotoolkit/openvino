// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/reduce_ops.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        reduceMeanParams,
        CPUSpecificParams,
        fusingSpecificParams> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj) {
        reduceMeanParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName(testing::TestParamInfo<reduceMeanParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }
protected:
    void SetUp() override {
        reduceMeanParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        InferenceEngine::Precision netPrecision;
        bool keepDims;
        std::vector<size_t> inputShape;
        std::vector<int> axes;
        CommonTestUtils::OpType opType;
        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
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

        selectedType = getPrimitiveType() + "_" + (inPrc == Precision::BOOL ? "I8" : inPrc.name());

        // hybrid layouts
        if (inFmts.size() != 0 && outFmts.size() == 0) {
            size_t outShapeSize = inputShape.size() - axes.size();
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

        function = makeNgraphFunction(ngPrc, params, reduce, "Reduce");
    }
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        if (ngraph::helpers::ReductionType::Prod == reductionType) {
            // We change the range of random values to avoid possible floating point overflow
            auto blob = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 10, 5);
            if (Precision::FP32 == info.getTensorDesc().getPrecision()) {
                auto *rawBlobDataPtr = blob->buffer().as<float *>();
                for (size_t i = 0; i < blob->size(); ++i) {
                    rawBlobDataPtr[i] /= 10.f;
                }
            } else if (Precision::BF16 == info.getTensorDesc().getPrecision()) {
                auto *rawBlobDataPtr = blob->buffer().as<ngraph::bfloat16 *>();
                for (size_t i = 0; i < blob->size(); ++i) {
                    rawBlobDataPtr[i] /= 10.f;
                }
            }
            return blob;
        } else {
            return LayerTestsCommon::GenerateInput(info);
        }
    }

private:
    ngraph::helpers::ReductionType reductionType;
};

TEST_P(ReduceCPULayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Reduce");
}
namespace {
std::vector<Precision> inpOutPrc = {Precision::BF16, Precision::FP32};

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

const std::vector<std::vector<int>> axes6D = {
        {0},
        {1},
        {2},
        {3},
        {4},
        {5},
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {0, 1, 2},
        {1, 2, 3},
        {2, 3, 4},
        {3, 4, 5},
        {0, 1, 2, 3},
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {0, 1, 2, 3, 4},
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

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 5, 15, 12},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<size_t>> inputShapes6D = {
        std::vector<size_t>{2, 3, 2, 2, 8, 9},
        std::vector<size_t>{2, 3, 2, 9, 8, 4},
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
        fusingRelu,
        fusingElu,
        fusingTanh,
        fusingSwish,
        /* FQ */
        fusingFakeQuantizePerChannel,
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
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::ValuesIn(inputShapes6D),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
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
            testing::Values(InferenceEngine::Precision::BOOL),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(InferenceEngine::Precision::BOOL),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(InferenceEngine::Precision::BOOL),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(InferenceEngine::Precision::BOOL),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid_Logical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(InferenceEngine::Precision::BOOL),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(InferenceEngine::Precision::BOOL),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::ValuesIn(inputShapes6D),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
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
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
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
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesNDFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesNDFusing),
            testing::Values(CommonTestUtils::OpType::VECTOR),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
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

