// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;


struct VariadicSplitTestCase {
    ngraph::Shape data_shape;
    std::vector<int64_t> split_lengths;
    int64_t axis, first_split_point, second_split_point;
};

const auto combinations = testing::Combine(
    testing::Values(
            ngraph::element::f16,
            ngraph::element::f32,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
    testing::Values(
            ngraph::element::i32,
            ngraph::element::i64),
    testing::Values(
            VariadicSplitTestCase{{6}, {2, 1, 2, 1}, 0, 0, 0},
            VariadicSplitTestCase{{6, 12, 10, 24}, {1, 1, 3, 1}, 0, 0, 1},
            VariadicSplitTestCase{{6, 12}, {7, 2, 1, 2}, 1, 1, 2},
            VariadicSplitTestCase{{6, 12, 10, 24}, {10, 14}, 3, 3, 4},
            VariadicSplitTestCase{{6, 12, 10, 24}, {14, 10}, -1, 3, 4},
            VariadicSplitTestCase{{6, 12, 10, 24}, {6}, -4, 0, 1}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    DataType,
    DataType,
    VariadicSplitTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_VariadicSplit : public testing::WithParamInterface<Parameters>,
        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& variadic_split_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, variadic_split_setup.data_shape);
        const auto axis = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{variadic_split_setup.axis});
        const auto split_lengths = ngraph::opset3::Constant::create(idx_type,
                {variadic_split_setup.split_lengths.size()}, std::vector<int64_t>{variadic_split_setup.split_lengths});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{variadic_split_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::VariadicSplit>(dsr, axis, split_lengths);

        const auto tests_wa = std::make_shared<ngraph::opset3::Concat>(node->outputs(), variadic_split_setup.axis);
        const auto result = std::make_shared<ngraph::opset3::Result>(tests_wa);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                ngraph::ParameterVector{data, dims}, "DSR-VariadicSplit");
    }
};

TEST_P(DSR_VariadicSplit, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicGatherData, DSR_VariadicSplit, combinations);

}  // namespace
