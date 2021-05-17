// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvolutionQDqTransformationTestValues {
public:
    class Values {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ngraph::builder::subgraph::Constant weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Values actual;
    Values expected;
};

typedef std::tuple<
    ngraph::Shape,
    ConvolutionQDqTransformationTestValues> ConvolutionQDqTransformationParams;

class ConvolutionQDqTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionQDqTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            {},
            {},
            testValues.actual.dequantizationOnActivations,
            testValues.actual.weights,
            testValues.actual.fakeQuantizeOnWeights,
            {},
            testValues.actual.dequantizationOnWeights,
            testValues.actual.dequantizationAfter);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            {},
            {},
            testValues.expected.dequantizationOnActivations,
            testValues.expected.weights,
            testValues.expected.fakeQuantizeOnWeights,
            {},
            testValues.expected.dequantizationOnWeights,
            testValues.expected.dequantizationAfter);
    }


    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionQDqTransformationParams> obj) {
        auto inputShape = std::get<0>(obj.param);
        ConvolutionQDqTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.actual.weights.outPrecision << "_" << "{ " <<
            testValues.actual.weights.values[0] << " }_" <<
            testValues.actual.fakeQuantizeOnWeights << "_" <<
            testValues.actual.dequantizationOnWeights;
        return result.str();
    }
};

TEST_P(ConvolutionQDqTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> shapes = {
    ngraph::Shape({ 1, 3, 72, 48 }),
    ngraph::Shape({ 4, 3, 72, 48 })
};

const std::vector<ConvolutionQDqTransformationTestValues> testValues = {
    // Actual:
    //                        Constant
    //                         |FP32  Constant Constant Constant Constant
    //                         |      /FP32    /FP32    /FP32    /FP32
    // Parameter   Constant   FakeQuantize  Constant
    //  |U8         |U8        |I8          /I8
    //  |           |          |           /
    // Convert    Convert     Convert  Convert
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant
    //   \U8      /U8
    //    \      /
    //    Subtract   Constant
    //      \FP32    /I8
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 1.f }, ngraph::element::f32},
            { 255ul, Shape({ 1, 1, 1, 1 }), { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {},
                { { 127.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ngraph::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false, { "DISABLED_CONSTANT_FOLDING" } },
                {}
            },
            { std::vector<float>{ 100.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0006f }, ngraph::element::f32, {1, 1, 1, 1}}}
        }
    },

    // Actual:
    //
    // Parameter   Constant
    //  |U8         |U8
    //  |           |
    // Convert    Convert
    //   \FP32    /FP32
    //    \      /
    //    Subtract  Constant  Constant
    //      \FP32   /FP32      |FP32  Constant Constant Constant Constant
    //       \     /           |      /FP32    /FP32    /FP32    /FP32
    //       Multiply      FakeQuantize
    //         \FP32       /FP32
    //          \         /
    //          Convolution
    //
    // Transformed:
    //
    // Parameter  Constant
    //   \U8      /U8
    //    \      /
    //    Subtract   Constant
    //      \FP32    /I8
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {},
            { std::vector<float>{ 2.f }, ngraph::element::f32},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 127.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {}},
            {},
            { std::vector<float>{ -125.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },

    // Actual:
    //
    // Parameter   Constant   Constant Constant
    //  |U8         |U8        |FP32    |I8
    //  |           |          |        |
    // Convert    Convert     Convert  Convert
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter   Constant
    //  |U8         |U8
    //  |           |
    // Convert    Convert
    //   \FP32    /FP32
    //    \      /
    //    Subtract  Constant
    //      \FP32   /FP32
    //       \     /
    //       Multiply       Constant
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {{ngraph::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::i8, true }, { 0.03f }},
            { std::vector<float>{ 2.f }, ngraph::element::f32},
            {},
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {},
            { std::vector<float>{ -3.75f }, ngraph::element::f32},
            {},
            ngraph::element::f32,
            {}
        }
    },

    // Actual:
    //
    // Parameter   Constant   Constant Constant
    //  |U8         |U8        |I8      |I8
    //  |           |          |        |
    // Convert    Convert     Convert  Convert
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant  Constant  Constant
    //   \U8      /U8       |I8      /I8
    //    \      /          |       /
    //    Subtract         Subtract
    //      \FP32         /FP32
    //       \           /
    //        Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {},
                { { 127.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ngraph::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false, { "DISABLED_CONSTANT_FOLDING" } },
                {}
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0006f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },

    // Actual:
    //
    // Parameter   Constant   Constant
    //  |U8         |U8        |I8
    //  |           |          |
    // Convert    Convert     Convert  Constant
    //   \FP32    /FP32        |FP32   /FP32
    //    \      /             |      /
    //    Subtract  Constant  Subtract  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // Parameter  Constant  Constant  Constant
    //   \U8      /U8       |I8      /I8
    //    \      /          |       /
    //    Subtract         Subtract
    //      \FP32         /FP32
    //       \           /
    //        Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {},
                { { 127.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ngraph::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false, { "DISABLED_CONSTANT_FOLDING" } },
                {}
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0006f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // incorrect zero point on activations [not transformed]
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {1000.f}, element::f32, {}, false },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {1000.f}, element::f32, {}, false },
                { {0.02f}, element::f32, {}, false }
            },
            {},
            { std::vector<float>{ -3.75f }, ngraph::element::f32},
            {},
            ngraph::element::f32,
            {}
        }
    },
    // incorrect zero point on weights [not transformed, weights folded]
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {1000.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {},
            { std::vector<float>{ -29.94f }, ngraph::element::f32},
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConvolutionQDqTransformation::getTestCaseName);
