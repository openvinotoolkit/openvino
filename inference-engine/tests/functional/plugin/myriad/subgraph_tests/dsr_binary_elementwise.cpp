// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

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

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {inputSubgraph0, inputSubgraph1});

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

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {inputSubgraph0, input1});

        return eltwise;
    }
};

static const std::vector<ngraph::NodeTypeInfo> binaryEltwiseTypeVector = {
        ngraph::opset3::Add::type_info,
        ngraph::opset3::Multiply::type_info,
        ngraph::opset3::Divide::type_info,
        ngraph::opset3::Subtract::type_info,
        ngraph::opset3::Equal::type_info,
        ngraph::opset3::Greater::type_info,
        ngraph::opset3::Power::type_info,
};

static const std::set<ngraph::NodeTypeInfo> doNotSupportI32 = {
        ngraph::opset3::Power::type_info,
        ngraph::opset3::Equal::type_info,
        ngraph::opset3::Greater::type_info,
};

TEST_P(DSR_BinaryElementwiseBothDSR, CompareWithReference) {
    const auto& inDataType = std::get<0>(GetParam());
    const auto& eltwiseType = std::get<2>(GetParam());

    if (doNotSupportI32.count(eltwiseType) && inDataType == ngraph::element::i32) {
        SKIP() << eltwiseType.name << " doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBound = {
        {
            DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}},
            DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}}
        },
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicBinaryElementwise, DSR_BinaryElementwiseBothDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBound),
        ::testing::ValuesIn(binaryEltwiseTypeVector),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

TEST_P(DSR_BinaryElementwiseSingleDSR, CompareWithReference) {
    const auto& inDataType = std::get<0>(GetParam());
    const auto& eltwiseType = std::get<2>(GetParam());

    if (doNotSupportI32.count(eltwiseType) && inDataType == ngraph::element::i32) {
        SKIP() << eltwiseType.name << " doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBoundSingleDSR = {
        {
            DataShapeWithUpperBound{DataShape{100, 100}, DataShape{200, 200}},
            DataShapeWithUpperBound{DataShape{1}, DataShape{}}
        },
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicBinaryElementwiseSingleDSR, DSR_BinaryElementwiseSingleDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBoundSingleDSR),
        ::testing::ValuesIn(binaryEltwiseTypeVector),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
