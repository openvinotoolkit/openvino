// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

struct BinaryEltwiseShapes {
    DataShapeWithUpperBound inputShape0;
    DataShapeWithUpperBound inputShape1;
};

using BinaryElementwiseParameters = std::tuple<
    DataType,
    BinaryEltwiseShapes,
    ngraph::NodeTypeInfo,
    LayerTestsUtils::TargetDevice
>;

class DSR_BinaryElementwiseBase : public testing::WithParamInterface<BinaryElementwiseParameters>,
                                  public DSR_TestsCommon {
protected:
    ngraph::NodeTypeInfo m_eltwiseType{};
    DataType m_inDataType = ngraph::element::dynamic;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        // Avoid division by zero
        const auto opForWhichAvoidingIsNecessary =
                m_eltwiseType == ngraph::opset3::Divide::type_info ||
                m_eltwiseType == ngraph::opset3::Power::type_info;
        const auto isDataInput = m_shapes.find(info.name()) == m_shapes.end();
        if (opForWhichAvoidingIsNecessary && isDataInput) {
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 5, 1, 1);
        }
        return DSR_TestsCommon::GenerateInput(info);
    }
};

class DSR_BinaryElementwise : public DSR_BinaryElementwiseBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        m_inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        m_eltwiseType = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph0 = createInputSubgraphWithDSR(m_inDataType, inDataShapes.inputShape0);
        const auto inputSubgraph1 = createInputSubgraphWithDSR(m_inDataType, inDataShapes.inputShape1);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(m_eltwiseType, {inputSubgraph0, inputSubgraph1});

        return eltwise;
    }
};

class DSR_BinaryElementwiseSingleDSR : public DSR_BinaryElementwiseBase {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        m_inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        m_eltwiseType = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph0 = createInputSubgraphWithDSR(m_inDataType, inDataShapes.inputShape0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(m_inDataType, inDataShapes.inputShape1.shape);
        m_parameterVector.push_back(input1);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(m_eltwiseType, {inputSubgraph0, input1});

        return eltwise;
    }
};

static const std::set<ngraph::NodeTypeInfo> doNotSupportI32 = {
        ngraph::opset3::Power::type_info,
        ngraph::opset3::Equal::type_info,
        ngraph::opset3::Greater::type_info,
};

TEST_P(DSR_BinaryElementwise, CompareWithReference) {
    if (doNotSupportI32.count(m_eltwiseType) && m_inDataType == ngraph::element::i32) {
        SKIP() << "Eltwise Power doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBound = {
        { DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}},
          DataShapeWithUpperBound{DataShape{800, 4}, DataShape{1000, 6}}
        },
};

INSTANTIATE_TEST_CASE_P(DynamicBinaryElementwise, DSR_BinaryElementwise,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBound),
        ::testing::Values(ngraph::opset3::Add::type_info,
                          ngraph::opset3::Multiply::type_info,
                          ngraph::opset3::Divide::type_info,
                          ngraph::opset3::Subtract::type_info,
                          ngraph::opset3::Equal::type_info,
                          ngraph::opset3::Greater::type_info,
                          ngraph::opset3::Power::type_info),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

TEST_P(DSR_BinaryElementwiseSingleDSR, CompareWithReference) {
    if (doNotSupportI32.count(m_eltwiseType) && m_inDataType == ngraph::element::i32) {
        SKIP() << "Eltwise Power doesn't support int32_t inputs" << std::endl;
    }

    Run();
}

std::vector<BinaryEltwiseShapes> dataShapesWithUpperBoundSingleDSR = {
        { DataShapeWithUpperBound{DataShape{100, 100}, DataShape{200, 200}},
                DataShapeWithUpperBound{DataShape{1}, DataShape{}}
        },
};

INSTANTIATE_TEST_CASE_P(DynamicBinaryElementwiseSingleDSR, DSR_BinaryElementwiseSingleDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::ValuesIn(dataShapesWithUpperBoundSingleDSR),
        ::testing::Values(ngraph::opset3::Add::type_info,
                          ngraph::opset3::Multiply::type_info,
                          ngraph::opset3::Divide::type_info,
                          ngraph::opset3::Subtract::type_info,
                          ngraph::opset3::Equal::type_info,
                          ngraph::opset3::Greater::type_info,
                          ngraph::opset3::Power::type_info),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
