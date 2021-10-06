// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/op/parameter.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <numeric>
#include <random>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/operations/static_shape_topk.hpp>
#include <vpu/ngraph/utilities.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct TopKTestCase {
    ngraph::Shape data_shape;
    int64_t k, axis, first_split_point, second_split_point;
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
            TopKTestCase{{6}, 5, 0, 0, 0},
            TopKTestCase{{6, 12, 10, 24}, 5, 0, 0, 1},
            TopKTestCase{{6, 12}, 10, 1, 1, 2},
            TopKTestCase{{6, 12, 10, 24}, 7, 3, 3, 4},
            TopKTestCase{{6, 12, 10, 24}, 20, -1, 3, 4},
            TopKTestCase{{6, 12, 10, 24}, 3, -4, 0, 1}));


class DynamicToStaticShapeTopKConst : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, TopKTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& topk_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, topk_setup),
                *reference(data_type, idx_type, topk_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const TopKTestCase& topk_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto k = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{topk_setup.k});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::TopK>(dsr, k, topk_setup.axis, "max", "value");

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            node->outputs(),
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(topk_setup.data_shape.size()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeTopK}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const TopKTestCase& topk_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto k = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{topk_setup.k});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::TopK>(dsr, k, topk_setup.axis, "max", "value");

        ngraph::OutputVector first_shape_part, second_shape_part;
        if (topk_setup.first_split_point) {
            first_shape_part.push_back(vpu::gatherShapeElements(dims, 0, topk_setup.first_split_point));
        }
        if (topk_setup.first_split_point + 1 < topk_setup.data_shape.size()) {
            second_shape_part.push_back(vpu::gatherShapeElements(
                dims,
                topk_setup.second_split_point,
                topk_setup.data_shape.size() - topk_setup.second_split_point));
        }
        ngraph::OutputVector results, converted;
        ngraph::Output<ngraph::Node> k_0D = k;
        if (node->get_input_element_type(1)!= ngraph::element::i64) {
            k_0D = std::make_shared<ngraph::opset3::Convert>(k, ngraph::element::i64);
        }
        const auto k_1D = std::make_shared<ngraph::opset3::Unsqueeze>(k_0D, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));

        if (!first_shape_part.empty() || !second_shape_part.empty()) {
            ngraph::OutputVector output_dims{k_1D};
            output_dims.insert(output_dims.begin(), first_shape_part.begin(), first_shape_part.end());
            output_dims.insert(output_dims.end(), second_shape_part.begin(), second_shape_part.end());
            const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), output_shape));
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(1), output_shape));
        } else {
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), k_1D));
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(1), k_1D));
        }
        return std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeTopKConst, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeTopKConst, combinations);


class DynamicToStaticShapeTopK : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, TopKTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& topk_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, topk_setup),
                *reference(data_type, idx_type, topk_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const TopKTestCase& topk_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(data);
        const auto gather = std::make_shared<ngraph::opset3::Gather>(shapeOf,
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {topk_setup.axis}),
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));
        const auto upper_bound = ngraph::opset3::Constant::create(dims->get_element_type(), {1}, {100});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{upper_bound, gather}, 0);
        const auto k = std::make_shared<ngraph::opset3::ReduceMin>(concat, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}), false);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::TopK>(dsr, k, topk_setup.axis, "max", "value");

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            node->outputs(),
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(topk_setup.data_shape.size()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeTopK}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const TopKTestCase& topk_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, topk_setup.data_shape);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{topk_setup.data_shape.size()});

        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(data);
        const auto gather = std::make_shared<ngraph::opset3::Gather>(shapeOf,
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {topk_setup.axis}),
                                                                     ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));
        const auto upper_bound = ngraph::opset3::Constant::create(dims->get_element_type(), {1}, {100});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{upper_bound, gather}, 0);
        const auto k = std::make_shared<ngraph::opset3::ReduceMin>(concat, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}), false);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::vpu::op::StaticShapeTopK>(dsr, k, topk_setup.axis, "max", "value");

        ngraph::OutputVector first_shape_part, second_shape_part;
        if (topk_setup.first_split_point) {
            first_shape_part.push_back(vpu::gatherShapeElements(dims, 0, topk_setup.first_split_point));
        }
        if (topk_setup.first_split_point + 1 < topk_setup.data_shape.size()) {
            second_shape_part.push_back(vpu::gatherShapeElements(
                dims,
                topk_setup.second_split_point,
                topk_setup.data_shape.size() - topk_setup.second_split_point));
        }
        ngraph::Output<ngraph::Node> k_0D = k;
        if (node->get_input_element_type(1)!= ngraph::element::i64) {
            k_0D = std::make_shared<ngraph::opset3::Convert>(k, ngraph::element::i64);
        }
        const auto k_1D = std::make_shared<ngraph::opset3::Unsqueeze>(k_0D, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));

        ngraph::OutputVector results, converted;
        if (!first_shape_part.empty() || !second_shape_part.empty()) {
            ngraph::OutputVector output_dims{k_1D};
            output_dims.insert(output_dims.begin(), first_shape_part.begin(), first_shape_part.end());
            output_dims.insert(output_dims.end(), second_shape_part.begin(), second_shape_part.end());
            const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), output_shape));
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(1), output_shape));
        } else {
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), k_1D));
            results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(1), k_1D));
        }

        return std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeTopK, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeTopK, combinations);

}  // namespace
