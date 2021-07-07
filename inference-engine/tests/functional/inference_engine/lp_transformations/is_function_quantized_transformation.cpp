// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include "lpt_ngraph_functions/common/builders.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class IsFunctionQuantizedTransformationValues {
public:
    ngraph::Shape shape;
    ngraph::element::Type precision;
    builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize;
    bool constantSubgraphOnParameters;
    bool inputOnParameters;

    bool isQuantized;
};

class IsFunctionQuantizedTransformation : public LayerTransformation, public testing::WithParamInterface<IsFunctionQuantizedTransformationValues> {
public:
    void SetUp() override {
        const auto testValues = GetParam();

        const auto input = std::make_shared<ngraph::opset1::Parameter>(testValues.precision, ngraph::Shape(testValues.shape));
        const auto fakeQuantize = ngraph::builder::subgraph::makeFakeQuantize(
            input,
            testValues.precision,
            testValues.fakeQuantize,
            testValues.constantSubgraphOnParameters);

        if (testValues.inputOnParameters) {
            replace_node(fakeQuantize->get_input_node_shared_ptr(3), input);
        }

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IsFunctionQuantizedFunction");
        function->validate_nodes_and_infer_types();
    }

    static std::string getTestCaseName(testing::TestParamInfo<IsFunctionQuantizedTransformationValues> obj) {
        IsFunctionQuantizedTransformationValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.shape << "_" <<
            testValues.precision << "_" <<
            testValues.fakeQuantize <<
            testValues.constantSubgraphOnParameters << "_" <<
            testValues.inputOnParameters << "_" <<
            testValues.isQuantized;
        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> function;
};

TEST_P(IsFunctionQuantizedTransformation, Run) {
    const bool isQuantized = ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(function);

    const auto testValues = GetParam();
    ASSERT_EQ(testValues.isQuantized, isQuantized);
}

const std::vector<ngraph::Shape> shapes = { ngraph::Shape({ 1, 3, 72, 48 }) };

const std::vector<IsFunctionQuantizedTransformationValues> testValues = {
    {
        ngraph::Shape{1, 3, 9, 9},
        ngraph::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        false,
        false,
        true
    },
    {
        ngraph::Shape{1, 3, 9, 9},
        ngraph::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        true,
        false,
        false
    },
    {
        ngraph::Shape{1, 3, 9, 9},
        ngraph::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        false,
        true,
        false
    },
    {
        ngraph::Shape{1, 3, 9, 9},
        ngraph::element::f32,
        { 255ul, {{ 1, 1, 1, 1 }}, { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
        true,
        true,
        false
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    IsFunctionQuantizedTransformation,
    ::testing::ValuesIn(testValues),
    IsFunctionQuantizedTransformation::getTestCaseName);
