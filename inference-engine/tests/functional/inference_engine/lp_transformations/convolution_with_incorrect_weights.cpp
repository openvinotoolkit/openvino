// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

#include "simple_low_precision_transformer.hpp"


namespace {
class ConvolutionWIthIncorrectWeightsTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        ngraph::element::Type dataPrecision;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::Shape inputShape;
    ngraph::element::Type precision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool isCorrect;
    Actual actual;
    Expected expected;
};

class ConvolutionWIthIncorrectWeightsTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ConvolutionWIthIncorrectWeightsTestValues> {
public:
    void SetUp() override {
        const ConvolutionWIthIncorrectWeightsTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginalWithIncorrectWeights(
            testValues.inputShape,
            testValues.precision,
            testValues.actual.fakeQuantizeOnWeights,
            testValues.actual.fakeQuantizeOnData,
            testValues.isCorrect);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::getReferenceWithIncorrectWeights(
            testValues.inputShape,
            testValues.precision,
            testValues.expected.dataPrecision,
            testValues.expected.fakeQuantizeOnData,
            testValues.expected.dequantizationBefore,
            testValues.expected.weightsPrecision,
            testValues.expected.weightsValues,
            testValues.expected.fakeQuantizeOnWeights,
            testValues.expected.dequantizationAfter,
            testValues.isCorrect);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionWIthIncorrectWeightsTestValues> obj) {
        const ConvolutionWIthIncorrectWeightsTestValues testValues = obj.param;

        std::ostringstream result;
        result << toString(testValues.params) <<
            (testValues.isCorrect ? "_correct_weights" : "_incorrect_weights");
        return result.str();
    }
};

TEST_P(ConvolutionWIthIncorrectWeightsTransformation, CompareFunctions) {
    ngraph::pass::InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ConvolutionWIthIncorrectWeightsTestValues> testValues = {
    // incorrect weights
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        ngraph::element::f32,
        LayerTransformation::createParamsU8I8(),
        bool{ false },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
            { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            ngraph::element::u8,
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
            {{ngraph::element::f32}, {}, {0.1f}},
            ngraph::element::f32,
            {1.f},
            { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
            {}
        },
    },
    // correct weights
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        ngraph::element::f32,
        LayerTransformation::createParamsU8I8(),
        true,
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
            { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            ngraph::element::u8,
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
            {},
            ngraph::element::i8,
            {-126.f},
            {},
            {{}, {}, {0.1f}},
        },
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConvolutionWIthIncorrectWeightsTransformation,
    ::testing::ValuesIn(testValues),
    ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName);

} // namespace
