// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/convolution_function.hpp"

#include "simple_low_precision_transformer.hpp"


namespace {
class ConvolutionWIthIncorrectWeightsTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type weightsPrecision;
        std::vector<float> weightsValues;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::element::Type inputPrecision;
    ngraph::Shape inputShape;
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
            testValues.inputPrecision,
            testValues.actual.fakeQuantizeOnWeights,
            testValues.actual.dequantization,
            testValues.isCorrect);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::getReferenceWithIncorrectWeights(
            testValues.inputShape,
            testValues.inputPrecision,
            testValues.expected.dequantizationBefore,
            testValues.expected.weightsPrecision,
            testValues.expected.weightsValues,
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
        ngraph::element::u8,
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        false,
        {
            {ngraph::element::f32, {}, {0.1f}},
            { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            {ngraph::element::f32, {}, {0.1f}},
            ngraph::element::f32,
            {-126.f},
            {}
        },
    },
    // correct weights
    {
        ngraph::element::u8,
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        true,
        {
            {ngraph::element::f32, {}, {0.1f}},
            { 255ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        },
        {
            {},
            ngraph::element::i8,
            {-126.f},
            {{}, {}, {{ 0.1f }, ngraph::element::f32, { 1, 1, 1, 1 }}},
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionWIthIncorrectWeightsTransformation,
    ::testing::ValuesIn(testValues),
    ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName);

} // namespace
