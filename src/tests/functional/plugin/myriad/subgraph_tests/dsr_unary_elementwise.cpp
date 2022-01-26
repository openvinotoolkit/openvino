// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
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
};

TEST_P(DSR_UnaryElementwise, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicUnaryElementwise, DSR_UnaryElementwise,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32),
        ::testing::Values(DataShapeWithUpperBound{ngraph::Shape{8, 800}, ngraph::Shape{10, 1000}}),
        ::testing::Values(ngraph::opset3::Exp::get_type_info_static(),
                          ngraph::opset3::Floor::get_type_info_static(),
                          ngraph::opset5::Ceiling::get_type_info_static(),
                          ngraph::opset5::Round::get_type_info_static(),
                          ngraph::opset3::Log::get_type_info_static(),
                          ngraph::opset3::Relu::get_type_info_static(),
                          ngraph::opset3::Sigmoid::get_type_info_static(),
                          ngraph::opset3::Softmax::get_type_info_static(),
                          ngraph::opset3::Sqrt::get_type_info_static()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
