// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

using Parameters = std::tuple<
    DataType,
    DataDims,
    LayerTestsUtils::TargetDevice
>;

class DSR_Clamp : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& dataDims = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims.size()});
        const auto dsr  = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = std::make_shared<ngraph::opset3::Clamp>(dsr, 0., 6.);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data, dims}, "DSR-Clamp");
    }
};

TEST_P(DSR_Clamp, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicClamp, DSR_Clamp,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(ngraph::Shape{1, 800}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
