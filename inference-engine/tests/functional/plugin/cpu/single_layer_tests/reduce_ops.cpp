// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/reduce_ops.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {

typedef std::tuple<reduceMeanParams, CPUSpecificParams> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj) {
        reduceMeanParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ReduceOpsLayerTest::getTestCaseName(testing::TestParamInfo<reduceMeanParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
protected:
    void SetUp() override {
        reduceMeanParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

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
            case CommonTestUtils::OpType::SCALAR: {
                if (axes.size() > 1)
                    FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
                break;
            }
            case CommonTestUtils::OpType::VECTOR: {
                shapeAxes.push_back(axes.size());
                break;
            }
            default:
                FAIL() << "Reduce op doesn't support operation type: " << opType;
        }
        auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

        const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);

        selectedType = getPrimitiveType() + "_" + (inPrc == Precision::BOOL ? "I8" : inPrc.name());

        reduce->get_rt_info() = getCPUInfo();

        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reduce)};
        function = std::make_shared<ngraph::Function>(results, params, "Reduce");
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

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 5, 15, 12},
        std::vector<size_t>{3, 5, 7, 9},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const auto paramsOneAxis = ::testing::Combine(
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
        testing::Values(emptyCPUSpec));

const auto paramsOneAxisLogical = testing::Combine(
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
        testing::Values(emptyCPUSpec));

const auto params_MultiAxis = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(opTypes[1]),
            testing::Values(false),
            testing::ValuesIn(reductionTypes),
            testing::ValuesIn(inpOutPrc),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 9, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec));

const auto params_MultiAxis_4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(opTypes[1]),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)));

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(opTypes[1]),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)));

const auto params_MultiAxisLogical = testing::Combine(
        testing::Combine(
            testing::ValuesIn(axesND),
            testing::Values(opTypes[1]),
            testing::Values(false),
            testing::ValuesIn(reductionLogicalTypes),
            testing::Values(InferenceEngine::Precision::BOOL),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(std::vector<size_t>{2, 9, 2, 9}),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::Values(emptyCPUSpec));

const auto params_MultiAxisLogical4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(opTypes[1]),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(InferenceEngine::Precision::BOOL),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)));

const auto params_MultiAxisLogical5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(opTypes[1]),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(InferenceEngine::Precision::BOOL),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 19, 7, 2, 9}),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)));

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceOneAxis_CPU,
        ReduceCPULayerTest,
        paramsOneAxis,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogicalOneAxis_CPU,
        ReduceCPULayerTest,
        paramsOneAxisLogical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_Reduce_ReductionTypes_CPU,
        ReduceCPULayerTest,
        params_MultiAxis,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_Reduce_ReductionTypes4D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_Reduce_ReductionTypes5D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogical_ReductionTypes_CPU,
        ReduceCPULayerTest,
        params_MultiAxisLogical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogical4D_ReductionTypes_CPU,
        ReduceCPULayerTest,
        params_MultiAxisLogical4D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_ReduceLogical5D_ReductionTypes_CPU,
        ReduceCPULayerTest,
        params_MultiAxisLogical5D,
        ReduceCPULayerTest::getTestCaseName
);
} // namespace
} // namespace CPULayerTestsDefinitions

