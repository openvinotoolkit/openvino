// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "low_precision/convert_subtract_constant.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class ConvertSubtractConstantTransformationTestValues {
public:
    class Values {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ov::builder::subgraph::Constant weights;
        ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Values actual;
    Values expected;
};

typedef std::tuple<
    ov::Shape,
    ConvertSubtractConstantTransformationTestValues> ConvertSubtractConstantTransformationParams;

class ConvertSubtractConstantTransformation : public LayerTransformation, public testing::WithParamInterface<ConvertSubtractConstantTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::low_precision::ConvertSubtractConstant>();
        manager.run_passes(actualFunction);

        referenceFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::Shape> shapes = {
    ov::Shape({ 1, 3, 72, 48 }),
    ov::Shape({ 4, 3, 72, 48 })
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
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true, {},
                  { {ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()} } },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
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
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {128.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {128.f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
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
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {0.000001f}, element::f32, {}, false },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvertSubtractConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConvertSubtractConstantTransformation::getTestCaseName);
