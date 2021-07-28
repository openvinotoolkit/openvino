// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>
#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

class DynamicToStaticShapeConvert : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataDims>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& dataDims = std::get<1>(parameters);

        ngraph::helpers::CompareFunctions(*transform(dataType, dataDims), *reference(dataType, dataDims));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& dataType,
            const ngraph::Shape& dataDims) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);

        const auto convert = std::make_shared<ngraph::opset3::Convert>(dsr, dataType);

        auto outputShape = convert->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{convert},
                ngraph::ParameterVector{data, dims},
                "Actual");
        convert->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{ngraph::opset3::Convert::type_info, vpu::dynamicToStaticUnaryElementwise}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& dataType,
            const ngraph::Shape& dataDims) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims);
        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims);
        const auto convert = std::make_shared<ngraph::opset3::Convert>(dsr0, dataType);

        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(convert, dims);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{data, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeConvert, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeConvert, testing::Combine(
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
