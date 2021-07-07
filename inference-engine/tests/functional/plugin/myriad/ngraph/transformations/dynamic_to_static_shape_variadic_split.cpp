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
#include <vpu/ngraph/transformations/dynamic_to_static_shape_variadic_split.hpp>
#include <queue>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/utilities.hpp>

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
            VariadicSplitTestCase{{6, 12, 10, 24}, {6}, -4, 0, 1}));


class DynamicToStaticShapeVeriadicSplit : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, VariadicSplitTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type = std::get<0>(parameters);
        const auto& idx_type = std::get<1>(parameters);
        const auto& variadic_split_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, idx_type, variadic_split_setup),
                *reference(data_type, idx_type, variadic_split_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const VariadicSplitTestCase& variadic_split_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, variadic_split_setup.data_shape);
        const auto axis = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{variadic_split_setup.axis});
        const auto split_lengths = ngraph::opset3::Constant::create(idx_type,
                {variadic_split_setup.split_lengths.size()}, std::vector<int64_t>{variadic_split_setup.split_lengths});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{variadic_split_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::VariadicSplit>(dsr, axis, split_lengths);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
            node->outputs(),
            ngraph::ParameterVector{data, dims},
            "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(variadic_split_setup.data_shape.size()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeVariadicSplit}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& data_type,
            const ngraph::element::Type_t& idx_type,
            const VariadicSplitTestCase& variadic_split_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(data_type, variadic_split_setup.data_shape);
        const auto axis = ngraph::opset3::Constant::create(idx_type, {}, std::vector<int64_t>{variadic_split_setup.axis});
        const auto split_lengths = ngraph::opset3::Constant::create(idx_type,
                {variadic_split_setup.split_lengths.size()}, std::vector<int64_t>{variadic_split_setup.split_lengths});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{variadic_split_setup.data_shape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto node = std::make_shared<ngraph::opset3::VariadicSplit>(dsr, axis, split_lengths);

        ngraph::OutputVector first_shape_part, second_shape_part;
        if (variadic_split_setup.first_split_point) {
            first_shape_part.push_back(vpu::gatherShapeElements(dims, 0, variadic_split_setup.first_split_point));
        }
        if (variadic_split_setup.first_split_point + 1 < variadic_split_setup.data_shape.size()) {
            second_shape_part.push_back(vpu::gatherShapeElements(
                dims,
                variadic_split_setup.second_split_point,
                variadic_split_setup.data_shape.size() - variadic_split_setup.second_split_point));
        }
        ngraph::NodeVector results;
        for (auto i = 0; i < variadic_split_setup.split_lengths.size(); ++i) {
            const auto dim = ngraph::opset3::Constant::create(dims->get_element_type(), {1}, {variadic_split_setup.split_lengths[i]});
            if (!first_shape_part.empty() || !second_shape_part.empty()) {
                ngraph::OutputVector output_dims{dim};
                output_dims.insert(output_dims.begin(), first_shape_part.begin(), first_shape_part.end());
                output_dims.insert(output_dims.end(), second_shape_part.begin(), second_shape_part.end());
                const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dims, 0);
                results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(i), output_shape));
            } else {
                results.push_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(i), dim));
            }
        }

        return std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{data, dims},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeVeriadicSplit, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeVeriadicSplit, combinations);

}  // namespace
