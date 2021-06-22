// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

struct ReduceTestCase {
    DataShapeWithUpperBound dataShapes;
    std::vector<int64_t> axes;
    bool keepDims;
};

const DataShapeWithUpperBound defaultReduceShapes {
        DataShape{24, 81}, DataShape{100, 81}
};

const auto arithmeticCombinations = testing::Combine(
        testing::Values(
                // ReduceMean can be replaced with avg pooling and work incorrectly #-34278
                // ngraph::opset3::ReduceMean::type_info,

                // ReduceProd is not supported by myriad plugin
                // ngraph::opset3::ReduceProd::type_info,
                ngraph::opset3::ReduceSum::type_info,
                ngraph::opset3::ReduceMax::type_info,
                ngraph::opset3::ReduceMin::type_info),
        testing::Values(
                ngraph::element::f32),
        testing::Values(
                ngraph::element::i32),
        testing::Values(
                // data_shape, axes, keep_dims
                ReduceTestCase{defaultReduceShapes, {0}, true},
                ReduceTestCase{defaultReduceShapes, {1}, false},
                ReduceTestCase{defaultReduceShapes, {0, 1}, true},
                ReduceTestCase{defaultReduceShapes, {0, 1}, false}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));

const auto logicalCombinations = testing::Combine(
        testing::Values(
                // ReduceLogicalOr is not supported by Myriad plugin
                // ngraph::opset3::ReduceLogicalOr::type_info,

                ngraph::opset3::ReduceLogicalAnd::type_info),
        testing::Values(ngraph::element::boolean),
        testing::Values(
                ngraph::element::i32),
        testing::Values(
                // data_shape, axes, keep_dims
                ReduceTestCase{defaultReduceShapes, {0}, true},
                ReduceTestCase{defaultReduceShapes, {1}, false},
                ReduceTestCase{defaultReduceShapes, {0, 1}, true},
                ReduceTestCase{defaultReduceShapes, {0, 1}, false}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    ngraph::NodeTypeInfo,
    DataType,
    DataType,
    ReduceTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Reduce : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& reduceType = std::get<0>(parameters);
        const auto& dataType = std::get<1>(parameters);
        const auto& axesType = std::get<2>(parameters);
        const auto& reduceSetup = std::get<3>(parameters);
        targetDevice = std::get<4>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, reduceSetup.dataShapes);
        const auto axes = ngraph::opset3::Constant::create(axesType, {reduceSetup.axes.size()}, reduceSetup.axes);

        const auto reduce = ngraph::helpers::getNodeSharedPtr(reduceType, {inputSubgraph, axes});

        if (auto arithmetic_reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(reduce))
            arithmetic_reduce->set_keep_dims(reduceSetup.keepDims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(reduce))
            logical_reduce->set_keep_dims(reduceSetup.keepDims);
        reduce->validate_and_infer_types();

        // CNNNetworkNGraphImpl handles only I64, I32 and FP32 precisions and sets FP32 as default otherwise.
        // Set I32 explicitly.
        if (dataType == ngraph::element::boolean) {
            outPrc = InferenceEngine::Precision::I32;
        }

        return reduce;
    }
};

TEST_P(DSR_Reduce, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicArithmeticReduce, DSR_Reduce, arithmeticCombinations);
INSTANTIATE_TEST_SUITE_P(smoke_DynamicLogicalReduce, DSR_Reduce, logicalCombinations);

}  // namespace
