// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_concat.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type;
using DataShape = ngraph::Shape;
using DataShapes = std::vector<DataShape>;

struct ConcatParam {
    DataShapes dataShapes;
    int axis;
};
using ConcatTestParam = std::tuple<DataType, ConcatParam>;

class DynamicToStaticShapeConcatTests
        : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<ConcatTestParam> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& concatParam = std::get<1>(parameters);
        const auto& dataShapes = concatParam.dataShapes;
        const auto& axis = concatParam.axis;

        ngraph::helpers::CompareFunctions(
                *transform(dataType, dataShapes, axis),
                *reference(dataType, dataShapes, axis));
    }

protected:
    std::shared_ptr<ngraph::Node> createDSRWithParams(
            const DataShape& dataShape,
            const ngraph::element::Type& dataType,
            ngraph::ParameterVector& params) const {
        const auto param = std::make_shared<ngraph::opset3::Parameter>(
                dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::i64, ngraph::Shape{dataShape.size()});
        params.push_back(param);
        params.push_back(shape);
        return std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(param, shape);
    }

    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type& dataType,
            const DataShapes& dataShapes,
            const int axis) const {
        ngraph::NodeVector dsrVector;
        ngraph::ParameterVector params;
        for (const auto& dataShape : dataShapes) {
            dsrVector.push_back(createDSRWithParams(dataShape, dataType, params));
        }

        const auto concat = std::make_shared<ngraph::opset3::Concat>(dsrVector, axis);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{concat}, params, "Actual");
        concat->set_output_type(0, dsrVector[0]->get_input_element_type(0),
                                ngraph::PartialShape::dynamic(concat->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{
            {ngraph::opset3::Concat::type_info, vpu::dynamicToStaticShapeConcat}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type& dataType,
            const DataShapes& dataShapes,
            const int axis) const {
        ngraph::NodeVector dsrVector;
        ngraph::ParameterVector params;

        dsrVector.push_back(createDSRWithParams(dataShapes.front(), dataType, params));

        auto accumulatedShape = params.back()->output(0);
        for (size_t inputIdx = 1; inputIdx < dataShapes.size(); ++inputIdx) {
            dsrVector.push_back(createDSRWithParams(
                    dataShapes.at(inputIdx), dataType, params));
            const auto shapeAccumulatorOp = std::make_shared<ngraph::opset3::Add>(
                    accumulatedShape, params.back());
            accumulatedShape = shapeAccumulatorOp->output(0);
        }

        const size_t rank = dataShapes.front().size();
        std::vector<int64_t> dividerValues(rank, dataShapes.size());
        dividerValues[axis < 0 ? axis + rank : axis] = 1;
        const auto divider = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{rank}, dividerValues);
        const auto divide = std::make_shared<ngraph::opset3::Divide>(accumulatedShape, divider);

        const auto concat = std::make_shared<ngraph::opset3::Concat>(dsrVector, axis);
        const auto outDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(concat, divide);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{outDsr}, params, "Expected");
    }
};

TEST_P(DynamicToStaticShapeConcatTests, CompareFunctions) {
}

std::vector<ngraph::element::Type> dataTypes = {
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8,
};

std::vector<ConcatParam> concatParams = {
        {DataShapes{DataShape{128}, DataShape{256}, DataShape{512}, DataShape{1024}}, 0},
        {DataShapes{DataShape{1, 1000}, DataShape{2, 1000}, DataShape{4, 1000}, DataShape{8, 1000}}, 0},
        {DataShapes{DataShape{128, 100}, DataShape{128, 200}, DataShape{128, 400}, DataShape{128, 800}}, 1},
        {DataShapes{DataShape{3, 64, 128}, DataShape{4, 64, 128}, DataShape{5, 64, 128}}, 0},
        {DataShapes{DataShape{3, 64, 128}, DataShape{3, 64, 256}, DataShape{3, 64, 512}}, 2},
        {DataShapes{DataShape{3, 64, 128}, DataShape{3, 64, 256}, DataShape{3, 64, 512}}, -1},
};

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeConcatTests, testing::Combine(
        testing::ValuesIn(dataTypes),
        testing::ValuesIn(concatParams)));

}  // namespace
