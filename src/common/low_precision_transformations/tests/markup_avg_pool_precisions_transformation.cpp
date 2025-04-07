// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/avg_pool.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"
#include <memory>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/markup_avg_pool_precisions.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov::pass;

class MarkupAvgPoolPrecisionsTransformationTestValues {
public:
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::element::Type,
                   ov::Shape,
                   bool,         // additional FakeQuantize After
                   std::string,  // additional layer before FQ
                   MarkupAvgPoolPrecisionsTransformationTestValues>
    MarkupAvgPoolPrecisionsTransformationParams;

class MarkupAvgPoolPrecisionsTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<MarkupAvgPoolPrecisionsTransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision;
        ov::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();

        actualFunction =
            ov::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getOriginal(precision,
                                                                                    testValues.actual.inputPrecision,
                                                                                    shape,
                                                                                    addFakeQuantize,
                                                                                    additionalLayer,
                                                                                    testValues.actual.dequantization,
                                                                                    1,
                                                                                    0);

        ov::pass::low_precision::TypeRelaxedReplacer pass;
        pass.run_on_model(actualFunction);

        auto supportedPrecisionsOnActivation = std::vector<ov::pass::low_precision::PrecisionsRestriction>(
            {ov::pass::low_precision::PrecisionsRestriction::create<ov::opset1::Convolution>(
                {{{0}, {ov::element::u8}}, {{1}, {ov::element::i8}}})});

        SimpleLowPrecisionTransformer transform(supportedPrecisionsOnActivation);
        transform.commonGraphRewrite->add_matcher<ov::pass::low_precision::AvgPoolTransformation>();
        transform.commonGraphRewrite->add_matcher<ov::pass::low_precision::ConvolutionTransformation>();
        transform.commonGraphRewrite
            ->add_matcher<ov::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        transform.commonGraphRewrite->add_matcher<ov::pass::low_precision::MaxPoolTransformation>();
        transform.cleanup->add_matcher<ov::pass::low_precision::FakeQuantizeTransformation>();
        transform.cleanup->add_matcher<ov::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
        transform.cleanup->add_matcher<ov::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MarkupAvgPoolPrecisionsTransformationParams> obj) {
        ov::element::Type precision;
        ov::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_"
               << LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision,
                                                               shape,
                                                               testValues.params)
               << "_" << testValues.actual.dequantization << "_" << testValues.expected.dequantizationBefore << "_"
               << testValues.expected.preicsionAfterOperation << "_" << testValues.expected.dequantizationAfter << "_"
               << (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(MarkupAvgPoolPrecisionsTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    const auto avgPoolOperations = LayerTransformation::get<ov::op::v1::AvgPool>(actualFunction);
    ASSERT_EQ(1ul, avgPoolOperations.size()) << "unexpected avgPoolOperations size: " << avgPoolOperations.size();

    {
        auto avgPoolPrecisioinPreservedAttribute =
            ov::pass::low_precision::getAttribute<ov::AvgPoolPrecisionPreservedAttribute>(*avgPoolOperations.begin());
        ASSERT_FALSE(avgPoolPrecisioinPreservedAttribute.empty());
        ASSERT_EQ(true, avgPoolPrecisioinPreservedAttribute.as<ov::AvgPoolPrecisionPreservedAttribute>().value());
    }

    const auto precisionPreserved = LayerTransformation::get<ov::op::v1::MaxPool>(actualFunction);
    ASSERT_TRUE(checkIfAttributesAreTheSame<ov::AvgPoolPrecisionPreservedAttribute>(precisionPreserved))
        << "AvgPoolPrecisionPreservedAttribute are not the same";

    // auto res = compare_functions(actualFunction, referenceFunction, true, true);
    // ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

const std::vector<std::string> additionalLayer = {
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    // true,
    false};

const std::vector<ov::Shape> shapes = {{1, 3, 9, 9}};

const std::vector<MarkupAvgPoolPrecisionsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // U8 without subtract
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {}, {0.02f}}}},
    // U8 per channel quantization with different values
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {{ov::element::f32}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}}},
     {
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}},
     }},
    // U8 per channel quantization with the same values
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {{ov::element::f32}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}},
     }},
    // U8 without dequantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::f32, {}},
     {ov::element::f32, {}, ov::element::f32, {}}},
    // U8 not update precisions
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     {ov::element::f32, {{}, {128.f}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // I8 per tensor quantization
    {LayerTransformation::createParamsI8I8(),
     {ov::element::f32, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // failed
    // I8 without subtract
    {LayerTransformation::createParamsI8I8(),
     {ov::element::f32, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {}, {0.02f}}}},
    // I8 per channel quantization with different values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::f32, {{ov::element::f32}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}}},
     {
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}},
     }},
    // I8 per channel quantization with the same values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::f32, {{ov::element::f32}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}},
     }},
    // I8 without dequantization
    {LayerTransformation::createParamsI8I8(),
     {ov::element::f32, {}},
     {ov::element::f32, {}, ov::element::f32, {}}},
    // I8 not update precisions
    {LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
     {ov::element::f32, {{}, {128.f}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MarkupAvgPoolPrecisionsTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(addFQ),
                                            ::testing::ValuesIn(additionalLayer),
                                            ::testing::ValuesIn(testValues)),
                         MarkupAvgPoolPrecisionsTransformation::getTestCaseName);
