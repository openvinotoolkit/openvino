// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/function.hpp"

#include "common_test_utils/test_common.hpp"
#include "gtest/gtest.h"

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <vpu/ngraph/operations/static_shape_reshape.hpp>

namespace {

using DataShape = ngraph::Shape;
using DataType  = ngraph::element::Type;
using ReshapePatternGenerator = std::function<std::shared_ptr<ngraph::op::Op>(std::shared_ptr<ngraph::vpu::op::DynamicShapeResolver>)>;

using TestParams = std::tuple<DataShape, DataType, ReshapePatternGenerator>;

class DynamicToStaticShapeReshapeTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<TestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        m_inDataShape      = std::get<0>(parameters);
        m_inDataType       = std::get<1>(parameters);
        m_patternGenerator = std::get<2>(parameters);

        const auto& actual = transform();
        const auto& expected = reference();
        ngraph::helpers::CompareFunctions(*actual, *expected);
    }

protected:
    std::shared_ptr<const ngraph::Function> transform() const {
        const auto dsr = generateInputSubgraph();

        const auto outShapeDescriptorParam = m_patternGenerator(dsr);
        const auto reshape = std::make_shared<ngraph::op::v1::Reshape>(dsr, outShapeDescriptorParam, true);

        const auto function = generateFunction(reshape, *dsr, "Actual");
        reshape->set_output_type(
            0,
            dsr->get_input_element_type(0),
            ngraph::PartialShape::dynamic(outShapeDescriptorParam->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{ngraph::op::v1::Reshape::type_info, vpu::dynamicToStaticShapeReshape}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference() const {
        const auto dsr0 = generateInputSubgraph();

        const auto outShapeDescriptorParam = m_patternGenerator(dsr0);
        const auto reshape = ngraph::as_type_ptr<ngraph::opset3::Constant>(outShapeDescriptorParam)
            ? std::make_shared<ngraph::opset3::Reshape>(dsr0, outShapeDescriptorParam, true)
            : std::make_shared<ngraph::vpu::op::StaticShapeReshape>(dsr0, outShapeDescriptorParam, true);

        const auto outShapeOfReshape = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(dsr0->input_value(1), outShapeDescriptorParam, true);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(reshape, outShapeOfReshape);
        return generateFunction(dsr1, *dsr0, "Expected");
    }

private:
    std::shared_ptr<ngraph::vpu::op::DynamicShapeResolver> generateInputSubgraph() const {
        const auto inDataParam = std::make_shared<ngraph::op::Parameter>(m_inDataType, m_inDataShape);
        const auto inDataDimsParam = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64, ngraph::Shape{m_inDataShape.size()});
        return std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(inDataParam, inDataDimsParam);
    }

    static std::shared_ptr<ngraph::Function> generateFunction(std::shared_ptr<ngraph::op::Op> result, const ngraph::op::Op& input, const std::string& name) {
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{std::move(result)},
            ngraph::ParameterVector{
                std::dynamic_pointer_cast<ngraph::opset3::Parameter>(input.get_input_node_shared_ptr(0)),
                std::dynamic_pointer_cast<ngraph::opset3::Parameter>(input.get_input_node_shared_ptr(1))
            },
            name);
    }

private:
    DataShape m_inDataShape;
    DataType m_inDataType;
    ReshapePatternGenerator m_patternGenerator;
};

TEST_P(DynamicToStaticShapeReshapeTests, compareFunctions) {}

std::shared_ptr<ngraph::op::Op> generateStaticReshapePattern(std::shared_ptr<ngraph::vpu::op::DynamicShapeResolver> dsr) {
    const auto& inDataShape = dsr->input_value(0).get_shape();
    std::vector<std::int64_t> pattern(inDataShape.rbegin(), inDataShape.rend());
    return std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{pattern.size()}, pattern);
}

std::shared_ptr<ngraph::op::Op> generateDynamicReshapePattern(std::shared_ptr<ngraph::vpu::op::DynamicShapeResolver> dsr) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(std::move(dsr));
    const auto axis    = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0});
    const auto indices = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {0});
    return std::make_shared<ngraph::opset3::Concat>(
        ngraph::OutputVector{
            std::make_shared<ngraph::opset3::Gather>(shapeOf, indices, axis),
            ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {-1})},
        0);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeReshapeTests, testing::Combine(
    testing::Values(
        DataShape{4, 1000},
        DataShape{3, 128, 256},
        DataShape{2, 3, 128, 256},
        DataShape{1000, 256, 7, 7}),
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        generateStaticReshapePattern,
        generateDynamicReshapePattern)
));

}  // namespace
