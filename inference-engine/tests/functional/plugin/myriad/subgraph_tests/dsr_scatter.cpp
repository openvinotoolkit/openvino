// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;


struct ScatterTestCase {
    ngraph::NodeTypeInfo scatter_type_info;
    ngraph::Shape data_shape, indices_shape, updates_shape;
    int64_t axis;
};

using Parameters = std::tuple<
    DataType,
    DataType,
    ScatterTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Scatter : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& numeric_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& scatter_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(integer_type, scatter_setup.indices_shape);
        const auto updates = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.updates_shape);
        const auto axis = std::make_shared<ngraph::opset3::Constant>(integer_type, ngraph::Shape{1}, std::vector<int64_t>{scatter_setup.axis});


        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatter_setup.data_shape.size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = ngraph::helpers::getNodeSharedPtr(scatter_setup.scatter_type_info, {dsr, indices, updates, axis});

        const auto result = std::make_shared<ngraph::opset3::Result>(node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, indices, updates, dims}, scatter_setup.scatter_type_info.name);
    }
};

TEST_P(DSR_Scatter, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicScatter, DSR_Scatter,
    ::testing::Combine(
            testing::Values(
                    ngraph::element::f16,
                    ngraph::element::f32,
                    ngraph::element::i32,
                    ngraph::element::i64,
                    ngraph::element::u8),
            testing::Values(
                    ngraph::element::i32,
                    ngraph::element::i64,
                    ngraph::element::u8),
            testing::Values(
                    ScatterTestCase{ngraph::opset3::ScatterUpdate::type_info, {1000, 256, 10, 15}, {125, 20}, {1000, 125, 20, 10, 15}, 1}),
    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
