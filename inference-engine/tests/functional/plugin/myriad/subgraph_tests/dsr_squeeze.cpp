// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;
using axis_vec = std::vector<int64_t>;

struct SqueezeTestCase {
    DataDims input_shape;
    axis_vec squeeze_axes;
};

using Parameters = std::tuple<
    DataType,
    SqueezeTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Squeeze : public testing::WithParamInterface<Parameters>, public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& squeeze_test_case = std::get<1>(parameters);

        const auto& input_shape = squeeze_test_case.input_shape;
        const auto& squeeze_axes = squeeze_test_case.squeeze_axes;

        targetDevice = std::get<2>(GetParam());

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, input_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{squeeze_axes.size()}, squeeze_axes);
        const auto node = std::make_shared<ngraph::opset3::Squeeze>(dsr, axes);

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data, dims}, "DSR-Squeeze");
    }
};

TEST_P(DSR_Squeeze, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicSqueeze, DSR_Squeeze,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(
                // input_shape, squeeze_axis
                SqueezeTestCase{DataDims{1, 1, 1000}, axis_vec{-2}},
                SqueezeTestCase{DataDims{1, 1000, 1}, axis_vec{0, 2}},
                SqueezeTestCase{DataDims{1, 1, 1}, axis_vec{1}},
                SqueezeTestCase{DataDims{1000, 1, 1}, axis_vec{2}}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
