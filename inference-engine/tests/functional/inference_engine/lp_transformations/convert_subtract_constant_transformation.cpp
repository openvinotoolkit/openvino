// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <low_precision/convert_subtract_constant.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvertSubtractConstantTransformationTestValues {
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
    ConvertSubtractConstantTransformationTestValues> ConvertSubtractConstantTransformationParams;

class ConvertSubtractConstantTransformation : public LayerTransformation, public testing::WithParamInterface<ConvertSubtractConstantTransformationParams> {
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

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::low_precision::ConvertSubtractConstant>();
        manager.run_passes(actualFunction);

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


    static std::string getTestCaseName(testing::TestParamInfo<ConvertSubtractConstantTransformationParams> obj) {
        auto inputShape = std::get<0>(obj.param);
        ConvertSubtractConstantTransformationTestValues testValues = std::get<1>(obj.param);

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

TEST_P(ConvertSubtractConstantTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> shapes = {
    ngraph::Shape({ 1, 3, 72, 48 }),
    ngraph::Shape({ 4, 3, 72, 48 })
};

const std::vector<ConvertSubtractConstantTransformationTestValues> testValues = {
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
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ngraph::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true, {}, { "DISABLED_CONSTANT_FOLDING" } },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        }
    },

    // Constant Subtract values are outside INT8 range
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
                { {128.f}, element::f32, {}, false },
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
            {
                { ngraph::element::f32, false },
                { {128.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        }
    },

    // Constant Subtract values are close to zero
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
    // Parameter   Constant
    //  |U8         |U8
    //  |           |
    // Convert    Convert     Constant
    //   \FP32    /FP32        |I8
    //    \      /             |
    //    Subtract  Constant  Convert  Constant
    //      \FP32   /FP32      |FP32   /FP32
    //       \     /           |      /
    //       Multiply         Multiply
    //         \FP32         /FP32
    //          \           /
    //           Convolution
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
                { {0.000001f}, element::f32, {}, false },
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
            {
                { ngraph::element::f32, false },
                { },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ngraph::element::i8},
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    ConvertSubtractConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConvertSubtractConstantTransformation::getTestCaseName);
