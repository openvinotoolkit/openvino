// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using Parameters = std::tuple<
    DataType,
    DataShapeWithUpperBound,
    ngraph::NodeTypeInfo,
    LayerTestsUtils::TargetDevice
>;

class DSR_UnaryElementwise : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        const auto& type_info = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(type_info, {inputSubgraph});

        return eltwise;
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        const auto& parameters = GetParam();
        const auto& type_info = std::get<2>(parameters);

        // Log(0) = inf, therefore we have to fill data with values starts with 1
        if (type_info == ngraph::opset3::Log::type_info && m_shapes.find(info.name()) == m_shapes.end()) {
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 10, 1, 1);
        }
        return DSR_TestsCommon::GenerateInput(info);
    }
};

TEST_P(DSR_UnaryElementwise, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicUnaryElementwise, DSR_UnaryElementwise,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32),
        ::testing::Values(DataShapeWithUpperBound{ngraph::Shape{8, 800}, ngraph::Shape{10, 1000}}),
        ::testing::Values(ngraph::opset3::Floor::type_info,
                          ngraph::opset3::Exp::type_info,
                          ngraph::opset3::Log::type_info,
                          ngraph::opset3::Relu::type_info,
                          ngraph::opset3::Sigmoid::type_info,
                          ngraph::opset3::Softmax::type_info,
                          ngraph::opset3::Sqrt::type_info),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
