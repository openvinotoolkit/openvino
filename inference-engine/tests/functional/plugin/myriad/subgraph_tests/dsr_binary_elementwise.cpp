// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <ngraph/opsets/opset6.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

struct BinaryEltwiseShapes {
    DataShapeWithUpperBound lhs;
    DataShapeWithUpperBound rhs;
};

using BinaryElementwiseParameters = std::tuple<
    DataType,
    BinaryEltwiseShapes,
    ngraph::NodeTypeInfo,
    LayerTestsUtils::TargetDevice
>;

class DSR_BinaryElementwiseBothDSR : public testing::WithParamInterface<BinaryElementwiseParameters>,
                                     public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        const auto& eltwiseType = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph0 = createInputSubgraphWithDSR(inDataType, inDataShapes.lhs);
        const auto inputSubgraph1 = createInputSubgraphWithDSR(inDataType, inDataShapes.rhs);

        const auto eltwise = eltwiseType == ngraph::opset6::Select::type_info ?
            ngraph::helpers::getNodeSharedPtr(eltwiseType, {createInputSubgraphWithDSR(
                ngraph::element::boolean, inDataShapes.lhs), inputSubgraph0, inputSubgraph1}) :
            ngraph::helpers::getNodeSharedPtr(eltwiseType, {inputSubgraph0, inputSubgraph1});

        return eltwise;
    }
};

class DSR_BinaryElementwiseSingleDSR : public testing::WithParamInterface<BinaryElementwiseParameters>,
                                       public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        const auto& eltwiseType = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph0 = createInputSubgraphWithDSR(inDataType, inDataShapes.lhs);
        const auto input1 = createParameter(inDataType, inDataShapes.rhs.shape);

        const auto eltwise = eltwiseType == ngraph::opset6::Select::type_info ?
            ngraph::helpers::getNodeSharedPtr(eltwiseType, {createParameter(
                ngraph::element::boolean, inDataShapes.rhs.shape), inputSubgraph0, input1}) :
            ngraph::helpers::getNodeSharedPtr(eltwiseType, {inputSubgraph0, input1});

        return eltwise;
    }
};

static const std::vector<ngraph::NodeTypeInfo> binaryEltwiseTypeVector = {
        ngraph::opset6::Add::type_info,
        ngraph::opset6::Multiply::type_info,
        ngraph::opset6::Divide::type_info,
        ngraph::opset6::Subtract::type_info,
        ngraph::opset6::Equal::type_info,
        ngraph::opset6::Greater::type_info,
        ngraph::opset6::Power::type_info,
        ngraph::opset6::Select::type_info,
};

static const std::set<ngraph::NodeTypeInfo> doNotSupportI32 = {
        ngraph::opset6::Power::type_info,
        ngraph::opset6::Equal::type_info,
        ngraph::opset6::Greater::type_info,
};

TEST_P(DSR_BinaryElementwiseBothDSR, CompareWithReference) {
    const auto& inDataType = std::get<0>(GetParam());
    const auto& eltwiseType = std::get<2>(GetParam());

    if (doNotSupportI32.count(eltwiseType) && inDataType == ngraph::element::i32) {
        GTEST_SKIP() << eltwiseType.name << " doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBound = {
        {
            DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}},
            DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}}
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBinaryElementwise, DSR_BinaryElementwiseBothDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBound),
        ::testing::ValuesIn(binaryEltwiseTypeVector),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

TEST_P(DSR_BinaryElementwiseSingleDSR, CompareWithReference) {
    const auto& inDataType = std::get<0>(GetParam());
    const auto& eltwiseType = std::get<2>(GetParam());

    if (doNotSupportI32.count(eltwiseType) && inDataType == ngraph::element::i32) {
        GTEST_SKIP() << eltwiseType.name << " doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBoundSingleDSR = {
        {
            DataShapeWithUpperBound{DataShape{100, 100}, DataShape{200, 200}},
            DataShapeWithUpperBound{DataShape{1}, DataShape{}}
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBinaryElementwiseSingleDSR, DSR_BinaryElementwiseSingleDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBoundSingleDSR),
        ::testing::ValuesIn(binaryEltwiseTypeVector),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

class DSR_BinaryElementwiseBothDSRCheckOutputShape : public DSR_BinaryElementwiseBothDSR {
protected:
    void Validate() override {
        const auto& actualOutputs = GetOutputs();
        ASSERT_EQ(actualOutputs.size(), 1);

        const auto& output = actualOutputs.front();
        const auto& outputShape = output->getTensorDesc().getDims();

        const auto& parameters = GetParam();
        const auto& inputShapes = std::get<1>(parameters);
        const auto& lhsShape = inputShapes.lhs.shape;
        const auto& rhsShape = inputShapes.rhs.shape;

        auto broadcastedPartialShape = ngraph::PartialShape{lhsShape};
        ngraph::PartialShape::broadcast_merge_into(broadcastedPartialShape, ngraph::PartialShape{rhsShape}, ngraph::op::AutoBroadcastSpec::NUMPY);
        const auto& broadcasted = broadcastedPartialShape.to_shape();

        ASSERT_EQ(broadcasted, outputShape);

        const auto& refTotal = ngraph::shape_size(broadcasted);
        const auto& actTotal = std::accumulate(outputShape.cbegin(), outputShape.cend(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
        ASSERT_EQ(refTotal, 0);
        ASSERT_EQ(actTotal, 0);
    }
};

TEST_P(DSR_BinaryElementwiseBothDSRCheckOutputShape, CheckOutputShape) {
    const auto& inDataType = std::get<0>(GetParam());
    const auto& eltwiseType = std::get<2>(GetParam());

    if (doNotSupportI32.count(eltwiseType) && inDataType == ngraph::element::i32) {
        GTEST_SKIP() << eltwiseType.name << " doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBoundBothDSREmpty = {
    {
        DataShapeWithUpperBound{DataShape{0}, DataShape{1}},
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
    },
    {
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
        DataShapeWithUpperBound{DataShape{0}, DataShape{1}},
    },
    {
        DataShapeWithUpperBound{DataShape{0}, DataShape{1}},
        DataShapeWithUpperBound{DataShape{0}, DataShape{1}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 2}, DataShape{1, 3}},
        DataShapeWithUpperBound{DataShape{1}, DataShape{3}},
    },
    {
        DataShapeWithUpperBound{DataShape{2, 0}, DataShape{3, 2}},
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 0}, DataShape{1, 2}},
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
    },
    {
        DataShapeWithUpperBound{DataShape{1}, DataShape{3}},
        DataShapeWithUpperBound{DataShape{0, 2}, DataShape{1, 3}},
    },
    {
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
        DataShapeWithUpperBound{DataShape{2, 0}, DataShape{3, 2}},
    },
    {
        DataShapeWithUpperBound{DataShape{1}, DataShape{2}},
        DataShapeWithUpperBound{DataShape{0, 0}, DataShape{1, 2}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 2}, DataShape{1, 3}},
        DataShapeWithUpperBound{DataShape{0, 2}, DataShape{1, 3}},
    },
    {
        DataShapeWithUpperBound{DataShape{2, 0}, DataShape{3, 1}},
        DataShapeWithUpperBound{DataShape{2, 0}, DataShape{3, 1}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 0}, DataShape{1, 1}},
        DataShapeWithUpperBound{DataShape{0, 0}, DataShape{1, 1}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 2, 3}, DataShape{1, 3, 4}},
        DataShapeWithUpperBound{DataShape{2, 3}, DataShape{3, 4}},
    },
    {
        DataShapeWithUpperBound{DataShape{4, 0, 3}, DataShape{5, 2, 4}},
        DataShapeWithUpperBound{DataShape{1, 3}, DataShape{2, 4}},
    },
    {
        DataShapeWithUpperBound{DataShape{4, 5, 0}, DataShape{5, 6, 2}},
        DataShapeWithUpperBound{DataShape{5, 1}, DataShape{6, 2}},
    },
    {
        DataShapeWithUpperBound{DataShape{2, 3}, DataShape{3, 4}},
        DataShapeWithUpperBound{DataShape{0, 2, 3}, DataShape{1, 3, 4}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 3}, DataShape{2, 4}},
        DataShapeWithUpperBound{DataShape{4, 0, 3}, DataShape{5, 2, 4}},
    },
    {
        DataShapeWithUpperBound{DataShape{5, 1}, DataShape{6, 2}},
        DataShapeWithUpperBound{DataShape{4, 5, 0}, DataShape{5, 6, 2}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 0}, DataShape{1, 1}},
        DataShapeWithUpperBound{DataShape{0, 0, 0}, DataShape{1, 1, 1}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
        DataShapeWithUpperBound{DataShape{7, 5, 6}, DataShape{8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 5, 6}, DataShape{8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
        DataShapeWithUpperBound{DataShape{7, 1, 6}, DataShape{8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
        DataShapeWithUpperBound{DataShape{7, 5, 1}, DataShape{8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 1, 1}, DataShape{8, 6, 7}},
    },

    {
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 1, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 5, 6}, DataShape{1, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 1, 6}, DataShape{2, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 5, 1}, DataShape{2, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
        DataShapeWithUpperBound{DataShape{1, 1, 1}, DataShape{2, 6, 7}},
    },

    {
        DataShapeWithUpperBound{DataShape{7, 5, 6}, DataShape{8, 6, 7}},
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 5, 6}, DataShape{8, 6, 7}},
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{7, 1, 6}, DataShape{8, 6, 7}},
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{7, 5, 1}, DataShape{8, 6, 7}},
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 1, 1}, DataShape{8, 6, 7}},
        DataShapeWithUpperBound{DataShape{0, 7, 5, 6}, DataShape{1, 8, 6, 7}},
    },

    {
        DataShapeWithUpperBound{DataShape{1, 5, 6}, DataShape{2, 6, 7}},
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 1, 6}, DataShape{2, 6, 7}},
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 5, 1}, DataShape{2, 6, 7}},
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
    },
    {
        DataShapeWithUpperBound{DataShape{1, 1, 1}, DataShape{2, 6, 7}},
        DataShapeWithUpperBound{DataShape{8, 0, 5, 6}, DataShape{9, 2, 6, 7}},
    },

    {
        DataShapeWithUpperBound{DataShape{2, 3, 1, 0, 1, 0}, DataShape{3, 4, 5, 2, 2, 1}},
        DataShapeWithUpperBound{DataShape{2, 1, 4, 1, 0, 0}, DataShape{3, 4, 5, 2, 2, 1}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_BinaryElementwiseBothDSRCheckOutputShape, DSR_BinaryElementwiseBothDSRCheckOutputShape,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBoundBothDSREmpty),
        ::testing::ValuesIn(binaryEltwiseTypeVector),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
