// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/type/element_type.hpp>
#include <ngraph/shape.hpp>
#include <common_test_utils/test_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <numeric>
#include <random>
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_squeeze.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;
using axis_vec = std::vector<int64_t>;

struct SqueezeTestCase {
    DataDims input_shape;
    axis_vec squeeze_axes;
    axis_vec gather_indices;
};

class DynamicToStaticShapeSqueeze : public CommonTestUtils::TestsCommon,
public testing::WithParamInterface<std::tuple<DataType, SqueezeTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& squeeze_test_case = std::get<1>(parameters);

        const auto& input_shape = squeeze_test_case.input_shape;
        const auto& squeeze_axes = squeeze_test_case.squeeze_axes;
        const auto& gather_indices = squeeze_test_case.gather_indices;

        ngraph::helpers::CompareFunctions(*transform(dataType, input_shape, squeeze_axes),
                *reference(dataType, input_shape, squeeze_axes, gather_indices));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& data_type,
        const ngraph::Shape& input_shape,
        const std::vector<std::int64_t>& squeeze_axes) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, input_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{squeeze_axes.size()}, squeeze_axes);
        const auto node = std::make_shared<ngraph::opset3::Squeeze>(dsr, axes);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(node->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeSqueeze}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::Shape& input_shape,
            const std::vector<std::int64_t>& squeeze_axes,
            const std::vector<std::int64_t>& gather_indices) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, input_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_shape.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{squeeze_axes.size()}, squeeze_axes);
        const auto squeeze = std::make_shared<ngraph::opset3::Squeeze>(dsr0, axes);

        const auto gather_axis_const = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
        const auto gather_indices_const = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{gather_indices.size()}, gather_indices);

        const auto gather = std::make_shared<ngraph::opset3::Gather>(dims, gather_indices_const, gather_axis_const);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(squeeze, gather);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeSqueeze, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeSqueeze, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        // input_shape, squeeze_axis, gather_indices
        SqueezeTestCase{DataDims{1, 1, 1000}, axis_vec{-2}, axis_vec{0, 2}},
        SqueezeTestCase{DataDims{1, 1000, 1}, axis_vec{0, 2}, axis_vec{1}},
        SqueezeTestCase{DataDims{1, 1, 1}, axis_vec{1}, axis_vec{0, 2}},
        SqueezeTestCase{DataDims{1000, 1, 1}, axis_vec{2}, axis_vec{0, 1}})));

}  // namespace
