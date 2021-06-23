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
#include <vpu/ngraph/transformations/dynamic_to_static_shape_unsqueeze.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;
using axis_vec = std::vector<int64_t>;
int64_t NEW_DIM = -100;

struct UnsqueezeTestCase {
    DataDims input_shape;
    axis_vec unsqueeze_axes;
    axis_vec concat_indices;
};

class DynamicToStaticShapeUnsqueeze : public CommonTestUtils::TestsCommon,
public testing::WithParamInterface<std::tuple<DataType, UnsqueezeTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& unsqueeze_test_case = std::get<1>(parameters);

        const auto& input_shape = unsqueeze_test_case.input_shape;
        const auto& unsqueeze_axes = unsqueeze_test_case.unsqueeze_axes;
        const auto& concat_indices = unsqueeze_test_case.concat_indices;

        ngraph::helpers::CompareFunctions(*transform(dataType, input_shape, unsqueeze_axes),
                *reference(dataType, input_shape, unsqueeze_axes, concat_indices));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& data_type,
        const ngraph::Shape& input_shape,
        const std::vector<std::int64_t>& unsqueeze_axes) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, input_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        const auto node = std::make_shared<ngraph::opset3::Unsqueeze>(dsr, axes);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{node},
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0),
                ngraph::PartialShape::dynamic(node->get_output_partial_shape(0).rank() + unsqueeze_axes.size()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeUnsqueeze}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::Shape& input_shape,
            const std::vector<std::int64_t>& unsqueeze_axes,
            const std::vector<std::int64_t>& concat_indices) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, input_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{input_shape.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        const auto unsqueeze = std::make_shared<ngraph::opset3::Unsqueeze>(dsr0, axes);

        const auto split_axis = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{0});
        const auto split = std::make_shared<ngraph::opset3::Split>(dims, split_axis, input_shape.size());

        ngraph::OutputVector new_shape;
        for (const auto & i : concat_indices) {
            if (i == NEW_DIM) {
                const auto new_dim = std::make_shared<ngraph::opset3::Constant>(
                        split->get_input_element_type(0), ngraph::Shape{1}, std::vector<int64_t>{0});
                new_shape.push_back(new_dim->output(0));
            } else {
                new_shape.push_back(split->output(i));
            }
        }

        const auto concat = std::make_shared<ngraph::opset3::Concat>(new_shape, 0);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(unsqueeze, concat);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeUnsqueeze, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeUnsqueeze, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        // input_shape, unsqueeze_axis, concat_indices
        UnsqueezeTestCase{DataDims{10, 100, 1000}, axis_vec{-1, -3}, axis_vec{0, 1, NEW_DIM, 2, NEW_DIM}},
        UnsqueezeTestCase{DataDims{10, 100, 1000}, axis_vec{0}, axis_vec{NEW_DIM, 0, 1, 2}},
        UnsqueezeTestCase{DataDims{10}, axis_vec{1}, axis_vec{0, NEW_DIM}},
        UnsqueezeTestCase{DataDims{10}, axis_vec{0}, axis_vec{NEW_DIM, 0}})));

}  // namespace
