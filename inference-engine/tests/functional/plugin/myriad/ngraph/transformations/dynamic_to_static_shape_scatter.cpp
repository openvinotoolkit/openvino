// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp>
#include <vpu/utils/error.hpp>
#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type_t;


struct ScatterTestCase {
    ngraph::NodeTypeInfo scatter_type_info;
    ngraph::Shape data_shape, indices_shape, updates_shape;
    int64_t axis;
};

class DynamicToStaticShapeScatter : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, ScatterTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& numeric_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& scatter_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(numeric_type, integer_type, scatter_setup),
                *reference(numeric_type, integer_type, scatter_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& numeric_type,
            const ngraph::element::Type_t& integer_type,
            const ScatterTestCase& scatter_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(integer_type, scatter_setup.indices_shape);
        const auto updates = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.updates_shape);
        const auto axis = std::make_shared<ngraph::opset3::Constant>(integer_type, ngraph::Shape{1}, std::vector<int64_t>{scatter_setup.axis});


        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatter_setup.data_shape.size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = ngraph::helpers::getNodeSharedPtr(scatter_setup.scatter_type_info, {dsr, indices, updates, axis});

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                ngraph::ParameterVector{data, indices, updates, dims},
                "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{scatter_setup.scatter_type_info, vpu::dynamicToStaticUnaryElementwise}};
        vpu::DynamicToStaticShape(transformations).transform(*function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& numeric_type,
            const ngraph::element::Type_t& integer_type,
            const ScatterTestCase& scatter_setup) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.data_shape);
        const auto indices = std::make_shared<ngraph::opset3::Parameter>(integer_type, scatter_setup.indices_shape);
        const auto updates = std::make_shared<ngraph::opset3::Parameter>(numeric_type, scatter_setup.updates_shape);
        const auto axis = std::make_shared<ngraph::opset3::Constant>(integer_type, ngraph::Shape{1}, std::vector<int64_t>{scatter_setup.axis});


        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{scatter_setup.data_shape.size()});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto node = ngraph::helpers::getNodeSharedPtr(scatter_setup.scatter_type_info, {dsr, indices, updates, axis});

        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, dims);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, indices, updates, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeScatter, CompareFunctions) {
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicToStaticShapeScatter, testing::Combine(
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
        ScatterTestCase{ngraph::opset3::ScatterUpdate::type_info, {1000, 256, 10, 15}, {125, 20}, {1000, 125, 20, 10, 15}, 1})));

}  // namespace
