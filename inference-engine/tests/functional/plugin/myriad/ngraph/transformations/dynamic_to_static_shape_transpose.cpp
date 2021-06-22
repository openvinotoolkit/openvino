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
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_transpose.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

ngraph::PartialShape makeDynamicShape(const ngraph::PartialShape& shape) {
    if (shape.is_dynamic()) {
        return shape;
    }

    const auto& numDimensions = shape.rank().get_length();
    if (numDimensions <= 1) {
        return ngraph::PartialShape{{ngraph::Dimension::dynamic()}};
    }

    auto dynamicShape = shape;
    for (auto i = numDimensions - 1; i > 0; --i) {
        dynamicShape[i] = ngraph::Dimension::dynamic();
    }

    return dynamicShape;
}

class DynamicToStaticShapeTranspose : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<DataType, DataDims>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& dataDims = std::get<1>(parameters);

        auto permutation = std::vector<std::int64_t>(dataDims.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), std::mt19937());

        ngraph::helpers::CompareFunctions(*transform(dataType, dataDims, permutation), *reference(dataType, dataDims, permutation));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& dataType,
        const ngraph::Shape& dataDims,
        const std::vector<std::int64_t>& permutation) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims.size()});
        const auto transposition = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()}, permutation);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto transpose = std::make_shared<ngraph::opset3::Transpose>(dsr, transposition);

        auto outputShape = transpose->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{transpose},
            ngraph::ParameterVector{data, dims},
            "Actual");
        transpose->set_output_type(0, dsr->get_input_element_type(0), makeDynamicShape(transposition->get_output_partial_shape(0)));

        const auto transformations = vpu::Transformations{{ngraph::opset3::Transpose::type_info, vpu::dynamicToStaticShapeTranspose}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
        const ngraph::element::Type_t& dataType,
        const ngraph::Shape& dataDims,
        const std::vector<std::int64_t>& permutation) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims.size()});
        const auto transposition = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{data->get_shape().size()}, permutation);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto transpose = std::make_shared<ngraph::opset3::Transpose>(dsr0, transposition);

        const auto axis = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64,
            ngraph::Shape{std::initializer_list<std::size_t>{1}},
            std::vector<std::size_t>{0});
        const auto scatterElementsUpdate = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(dims, transposition, dims, axis);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(transpose, scatterElementsUpdate);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeTranspose, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeTranspose, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        DataDims{1000},
        DataDims{4, 1000},
        DataDims{3, 128, 256},
        DataDims{2, 3, 128, 256})));

}  // namespace
