// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/avg_pool.hpp"
#include "low_precision/max_pool.hpp"
#include <memory>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/avg_pool.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov;

class AvgPoolTransformationTestValues {
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
                   ov::PartialShape,
                   bool,         // additional FakeQuantize After
                   std::string,  // additional layer before FQ
                   AvgPoolTransformationTestValues>
    AvgPoolTransformationParams;

class AvgPoolTransformation : public LayerTransformation,
                              public testing::WithParamInterface<AvgPoolTransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision;
        ov::PartialShape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();
        actualFunction = ov::builder::subgraph::AvgPoolFunction::getOriginal(precision,
                                                                                 testValues.actual.inputPrecision,
                                                                                 shape,
                                                                                 addFakeQuantize,
                                                                                 {additionalLayer},
                                                                                 testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::AvgPoolTransformation, ov::op::v1::AvgPool>(testValues.params);
        transform.add<ov::pass::low_precision::MaxPoolTransformation, ov::op::v1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::AvgPoolFunction::getReference(precision,
                                                                     testValues.expected.inputPrecision,
                                                                     shape,
                                                                     addFakeQuantize,
                                                                     {additionalLayer},
                                                                     testValues.expected.dequantizationBefore,
                                                                     testValues.expected.preicsionAfterOperation,
                                                                     {},
                                                                     testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolTransformationParams> obj) {
        ov::element::Type precision;
        ov::PartialShape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolTransformationTestValues testValues;
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

TEST_P(AvgPoolTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::string> additionalLayer = {
    "",
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {true, false};

const std::vector<ov::PartialShape> shapes = {{1, 3, 72, 48}, {-1, -1, -1, -1}};

const std::vector<AvgPoolTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::u8, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // U8 without subtract
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::u8, {}, ov::element::f32, {{}, {}, {0.02f}}}},
    // U8 per channel quantization with different values
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}}},
     {
         ov::element::u8,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}},
     }},
    // U8 per channel quantization with the same values
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ov::element::u8,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}},
     }},
    // U8 without dequantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {}},
     {ov::element::u8, {}, ov::element::u8, {}}},
    // U8 not update precisions
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     {ov::element::f32, {{}, {128.f}, {0.02f}}},
     {ov::element::f32, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // I8 per tensor quantization
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::i8, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // I8 without subtract
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::i8, {}, ov::element::f32, {{}, {}, {0.02f}}}},
    // I8 per channel quantization with different values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}}},
     {
         ov::element::i8,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}},
     }},
    // I8 per channel quantization with the same values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ov::element::i8,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         AvgPoolTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(addFQ),
                                            ::testing::ValuesIn(additionalLayer),
                                            ::testing::ValuesIn(testValues)),
                         AvgPoolTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicChannel = {PartialShape::dynamic()};

const std::vector<AvgPoolTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128.f}, {0.02f}}},
     {ov::element::u8, {}, ov::element::f32, {{}, {128.f}, {0.02f}}}},
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.02f, 0.03f, 0.01f}}}},
     {ov::element::u8,
      {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.02f, 0.03f, 0.01f}}},
      ov::element::f32,
      {}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         AvgPoolTransformation,
                         ::testing::Combine(::testing::Values(element::f32),
                                            ::testing::ValuesIn(shapesWithDynamicChannel),
                                            ::testing::Values(false),
                                            ::testing::Values(""),
                                            ::testing::ValuesIn(testValues)),
                         AvgPoolTransformation::getTestCaseName);
}  // namespace testValues2
