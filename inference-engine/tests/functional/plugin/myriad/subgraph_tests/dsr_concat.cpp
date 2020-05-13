// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>

#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type;
using DataShape = ngraph::Shape;
using DataShapes = std::vector<DataShape>;

struct ConcatParam {
    DataShapes dataShapes;
    int axis;
};
using ConcatTestParam = std::tuple<DataType, ConcatParam, LayerTestsUtils::TargetDevice>;

class DSR_Concat
        : public testing::WithParamInterface<ConcatTestParam>,
          public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& concatParam = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto& dataShapes = concatParam.dataShapes;
        const auto& axis = concatParam.axis;

        ngraph::NodeVector dsrVector;
        ngraph::ParameterVector params;
        for (const auto& dataShape : dataShapes) {
            const auto param = std::make_shared<ngraph::opset3::Parameter>(
                    dataType, dataShape);
            const auto shape = std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::i64, ngraph::Shape{dataShape.size()});
            dsrVector.emplace_back(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                    param, shape));
            params.push_back(param);
            params.push_back(shape);
        }

        const auto concat = std::make_shared<ngraph::opset3::Concat>(dsrVector, axis);
        const auto result = std::make_shared<ngraph::opset3::Result>(concat);

        function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{result}, params, "DSR-Concat");
    }
};

TEST_P(DSR_Concat, CompareWithReference) {
    Run();
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
};

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicConcat, DSR_Concat, ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(concatParams),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
