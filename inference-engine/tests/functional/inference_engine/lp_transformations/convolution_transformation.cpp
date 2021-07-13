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
#include "lpt_ngraph_functions/convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvolutionTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnActivations;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        std::shared_ptr<ngraph::opset1::Constant> weights;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        ngraph::element::Type precisionAfterDequantization;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    element::Type,
    ngraph::PartialShape,
    ConvolutionTransformationTestValues> ConvolutionTransformationParams;

class ConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionTransformationParams> {
public:
    void SetUp() override {
        const auto netPrecision = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());
        auto testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginal(
            netPrecision,
            testValues.actual.precisionBeforeDequantization,
            inputShape,
            testValues.actual.dequantizationOnActivations,
            testValues.actual.weights,
            testValues.actual.fakeQuantizeOnWeights);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        if (testValues.params.supportAsymmetricQuantization == false) {
            transform.set_callback<ngraph::pass::low_precision::ConvolutionTransformation>(
                [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
                    return ngraph::pass::low_precision::LayerTransformation::isAsymmetricQuantization(node);
                });
        }
        transform.transform(actualFunction);

        if (!testValues.params.updatePrecisions) {
            const auto convertOnWeights = std::make_shared<opset1::Convert>(testValues.expected.weights, netPrecision);
            OutputVector convertedOutput(1);
            convertOnWeights->constant_fold(convertedOutput, convertOnWeights->input_values());
            const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();
            testValues.expected.weights = as_type_ptr<opset1::Constant>(convertedWeights);
        }

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::getReference(
            netPrecision,
            testValues.expected.precisionBeforeDequantization,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.expected.weights,
            testValues.expected.fakeQuantizeOnWeights,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            testValues.expected.precisionAfterDequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
        const auto netPrecision = std::get<0>(obj.param);
        auto inputShape = std::get<1>(obj.param);
        ConvolutionTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            netPrecision << "_" <<
            inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantizationOnActivations << "_" << "_weights_" <<
            testValues.actual.weights->get_element_type() << "_" << "{ " <<
            testValues.actual.weights->cast_vector<float>()[0] << " }_" <<
            testValues.actual.fakeQuantizeOnWeights << "_";
        return result.str();
    }
};

TEST_P(ConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<element::Type> netPrecisions = {
    element::f32,
    element::f16
};

const std::vector<ngraph::PartialShape> suitablePartialShapes = {
    ngraph::PartialShape({ 1, 3, 72, 48 }),
    ngraph::PartialShape({ 4, 3, 72, 48 }),
    ngraph::PartialShape({ Dimension::dynamic(), 3, 72, 48 }),
    ngraph::PartialShape({ 1, 3, Dimension::dynamic(), Dimension::dynamic() }),
};

const std::vector<ConvolutionTransformationTestValues> testValues = {
    // with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ ngraph::element::f32 }, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            ngraph::element::f32,
            {}
        }
    },
    // with zero point, not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        // ActualValues
        {
            ngraph::element::f32,
            {{}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // without zero point, not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        // ActualValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // with zero point, per-channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { { 128.f }, ngraph::element::f32, {1, 3, 1, 1} }, { { 0.02f },  ngraph::element::f32, {1, 3, 1, 1} }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, { { 128.f }, ngraph::element::f32, { 1, 3, 1, 1 }, false }, {}},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // with zero point, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }, ngraph::element::f32, { 1, 3, 1, 1 }},
                {{ 0.02f, 0.01f, 0.03f }, ngraph::element::f32, {1, 3, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }, ngraph::element::f32, { 1, 3, 1, 1 }},
                {{ 0.02f, 0.01f, 0.03f }, ngraph::element::f32, {1, 3, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
    // float input
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::f32,
            {
                {ngraph::element::f32},
                {{ 128.f }, ngraph::element::f32, { 1, 1, 1, 1 }},
                {{ 0.02f }, ngraph::element::f32, {1, 1, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {
                {ngraph::element::f32},
                {{ 128.f }, ngraph::element::f32, { 1, 1, 1, 1 }},
                {{ 0.02f }, ngraph::element::f32, {1, 1, 1, 1}}
            },
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
    // without dequantization operations
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::f32,
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            ngraph::element::f32,
            {}
        }
    },
    // without zero point, without convert
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::f32,
            {{}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {{}, {}, { {0.02f}, element::f32 }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{element::f32}, {}, { {0.02f}, element::f32 }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {},
            op::Constant::create(ngraph::element::i8, ngraph::Shape{}, std::vector<float>{ -125.f }),
            {},
            ngraph::element::f32,
            {{}, {}, {{ 0.0002f }, ngraph::element::f32, { 1, 1, 1, 1 }}}
        }
    },
    // incorrect zero point on activations [not transformed]
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{element::f32}, { 1000.f }, { {0.02f}, element::f32 }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{element::f32}, { 1000.f }, { {0.02f}, element::f32 }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
    // TODO: uncomment: remove precisionsOnActivations & precisionsOnWeights
//    // incorrect zero point on weights [not transformed, weights folded]
//    {
//        LayerTransformation::createParamsU8I8(),
//        // ActualValues
//        {
//            ngraph::element::u8,
//            {{element::f32}, {}, { {0.02f}, element::f32 }},
//            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 0.f }),
//            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { 5.f }, { 6.f } }
//        },
//        // ExpectedValues
//        {
//            ngraph::element::u8,
//            {{element::f32}, {}, { {0.02f}, element::f32 }},
//            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 5.f }),
//            {},
//            ngraph::element::f32,
//            {}
//        }
//    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(suitablePartialShapes),
        ::testing::ValuesIn(testValues)),
    ConvolutionTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<element::Type> netPrecisions = {
    element::f32,
    element::f16
};

const std::vector<ngraph::PartialShape> unsuitablePartialShapes = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
    ngraph::PartialShape::dynamic()
};

const std::vector<ConvolutionTransformationTestValues> testValues = {
    // with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ ngraph::element::f32 }, { 128.f }, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ 2.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{ ngraph::element::f32 }, {}, { 0.02f }},
            op::Constant::create(ngraph::element::f32, ngraph::Shape{}, std::vector<float>{ -1.25f }),
            {},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(unsuitablePartialShapes),
        ::testing::ValuesIn(testValues)),
    ConvolutionTransformation::getTestCaseName);
} // namespace testValues2
