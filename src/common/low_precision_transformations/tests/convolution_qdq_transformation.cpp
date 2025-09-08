// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/convolution.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class ConvolutionQDqTransformationTestValues {
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
    ov::PartialShape,
    ConvolutionQDqTransformationTestValues> ConvolutionQDqTransformationParams;

class ConvolutionQDqTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionQDqTransformationParams> {
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

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(testValues.params);
        transform.transform(actualFunction);

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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> suitablePartialShapes = {
    ov::PartialShape({ 1, 3, 72, 48 }),
    ov::PartialShape({ 4, 3, 72, 48 }),
    ov::PartialShape({ -1, 3, 72, 48 }),
    ov::PartialShape({ -1, -1, -1, -1 }),
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
            ov::element::u8,
            {
                {ov::element::f32},
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true },
                { {0.03f}, element::f32, {}, false }
            },
            { std::vector<float>{ 1.f }, ov::element::f32},
            { 255ul, Shape({ 1, 1, 1, 1 }), { -1.28f }, { 1.27f }, { -128.f }, { 127.f }, element::i8 },
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                {},
                { { 127.f }, ov::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ov::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false,
                  { {ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()} } },
                {}
            },
            { std::vector<float>{ 100.f }, ov::element::i8},
            {},
            ov::element::f32,
            {{}, {}, {{ 0.0006f }, ov::element::f32, {}}}
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
            ov::element::u8,
            {{ov::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {},
            { std::vector<float>{ 2.f }, ov::element::f32},
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, { { 127.f }, ov::element::f32, { 1, 3, 1, 1 }, false }, {}},
            {},
            { std::vector<float>{ -125.f }, ov::element::i8},
            {},
            ov::element::f32,
            {{}, {}, {{ 0.0002f }, ov::element::f32, {}}}
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
            ov::element::u8,
            {{ov::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {{ov::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::i8, true }, { 0.03f }},
            { std::vector<float>{ 2.f }, ov::element::f32},
            {},
            ov::element::f32,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, { {127.f}, element::f32, {}, false, 1ul, element::u8, true }, { 0.02f }},
            {},
            { std::vector<float>{ -3.75f }, ov::element::f32},
            {},
            ov::element::f32,
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
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::u8, true },
                { {0.02f}, element::f32, {}, false }
            },
            {
                { ov::element::f32, false },
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true },
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
                {},
                { { 127.f }, ov::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ov::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false,
                  { {ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()} } },
                {}
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {{}, {}, {{ 0.0006f }, ov::element::f32, {}}}
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
                {},
                { { 127.f }, ov::element::f32, { 1, 3, 1, 1 }, false },
                {}
            },
            {
                {},
                { { 127.f }, ov::element::f32, { 6, 1, 1, 1 }, false, 1ul, element::i8, false,
                  { {ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()} } },
                {}
            },
            { std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {{}, {}, {{ 0.0006f }, ov::element::f32, {}}}
        }
    },
    // mixed precision: f16 dequantization constants, f32 dequantization precision
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f16}, {}, {0.02f}},
            {{ov::element::f16}, {}, {0.03f}},
            {std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f16,
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, {}, {}},
            {{}, {}, {}},
            {std::vector<float>{ 2.f }, ov::element::i8},
            {},
            ov::element::f32,
            {{}, {}, {{ 0.0006f }, ov::element::f16, {}, false, 1, ov::element::f32}}
        }
    },
    // incorrect zero point on activations [not transformed]
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ov::element::u8,
            {
                { ov::element::f32, false },
                { {1000.f}, element::f32, {}, false },
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
                { {1000.f}, element::f32, {}, false },
                { {0.02f}, element::f32, {}, false }
            },
            {},
            { std::vector<float>{ -3.75f }, ov::element::f32},
            {},
            ov::element::f32,
            {}
        }
    },
    // incorrect zero point on weights [not transformed, weights folded]
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
                { {1000.f}, element::f32, {}, false },
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
            {},
            { std::vector<float>{ -29.94f }, ov::element::f32},
            {},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(suitablePartialShapes),
        ::testing::ValuesIn(testValues)),
    ConvolutionQDqTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> unsuitablePartialShapes = {
    PartialShape::dynamic(),
};

const std::vector<ConvolutionQDqTransformationTestValues> testValues = {
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
                { {127.f}, element::f32, {}, false, 1ul, element::i8, true },
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
            {},
            { std::vector<float>{ -3.75 }, ov::element::f32},
            {},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionQDqTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(unsuitablePartialShapes),
        ::testing::ValuesIn(testValues)),
    ConvolutionQDqTransformation::getTestCaseName);
} // namespace testValues2
