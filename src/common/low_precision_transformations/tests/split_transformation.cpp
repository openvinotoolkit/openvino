// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/split.hpp"
#include <memory>

#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/split.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

class SplitTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        std::vector<ov::builder::subgraph::DequantizationOperations> dequantizationAfter;
    };

    ov::PartialShape inputShape;
    std::int64_t splitedAxis;
    size_t numSplits;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::element::Type, SplitTransformationTestValues> SplitTransformationParams;

class SplitTransformation : public LayerTransformation, public testing::WithParamInterface<SplitTransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision = std::get<0>(GetParam());
        SplitTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction =
            ov::builder::subgraph::SplitFunction::getOriginal(precision,
                                                                  testValues.inputShape,
                                                                  testValues.actual.precisionBeforeDequantization,
                                                                  testValues.actual.dequantization,
                                                                  testValues.splitedAxis,
                                                                  testValues.numSplits);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::SplitTransformation, ov::op::v1::Split>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::SplitFunction::getReference(precision,
                                                                   testValues.inputShape,
                                                                   testValues.expected.inputPrecision,
                                                                   testValues.expected.dequantizationBefore,
                                                                   testValues.expected.precisionAfterOperation,
                                                                   testValues.expected.dequantizationAfter,
                                                                   testValues.splitedAxis,
                                                                   testValues.numSplits);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SplitTransformationParams> obj) {
        ov::element::Type precision = std::get<0>(obj.param);
        SplitTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << precision << "_" << toString(testValues.params) << "_" << testValues.inputShape << "_"
               << testValues.actual.precisionBeforeDequantization << "_" << testValues.actual.dequantization << "_"
               << testValues.expected.dequantizationAfter << "_axis=" << testValues.splitedAxis
               << "_num_splits=" << testValues.numSplits;
        return result.str();
    }
};

TEST_P(SplitTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};
const std::vector<SplitTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {{1, 3, 16, 16},
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     // ExpectedValues
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // U8 per tensor quantization / int8 subtraction with Convert from u8 to fp32
    {{1, 3, 16, 16},
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8, {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::u8, true}, {3.f}}},
     // ExpectedValues
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::u8, true}, {3.f}},
          {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::u8, true}, {3.f}},
      }}},
    // U8 per tensor quantization / int8 subtraction with Convert from fp16 -> fp32
    {{1, 3, 16, 16},
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8,
      {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::f16, true}, {3.f}}},
     // ExpectedValues
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::f16, true}, {3.f}},
          {{ov::element::f32}, {{128.f}, element::dynamic, {}, false, 1ul, element::f16, true}, {3.f}},
      }}},
    {{-1, -1, -1, -1},
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     // ExpectedValues
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    {PartialShape::dynamic(),
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     // ExpectedValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}, ov::element::f32, {}}},
    // I8 per tensor quantization
    {{1, 3, 16, 16},
     std::int64_t{2},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::i8, {{ov::element::f32}, {128.f}, {3.f}}},
     {ov::element::i8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // U8 per channel quantization with different values
    {{1, 3, 16, 16},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // U8 per channel quantization with different values and dynamic shapes
    {{-1, 3, -1, -1},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // U8 per channel quantization with different values and dynamic shapes (dynamic channels)
    {{-1, -1, -1, -1},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // U8 per channel quantization with different values (constants without batch)
    {{1, 3, 16, 16},
     std::int64_t{-3},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // I8 per channel quantization with different values
    {{1, 3, 16, 16},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // per channel quantization with different values, split by batch
    {{2, 3, 16, 16},
     std::int64_t{0},
     size_t{2},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{2.f, 3.f}, ov::element::f32, {2, 1, 1, 1}},
       {{22.f, 33.f}, ov::element::f32, {2, 1, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {2.f}, {22.f}},
          {{ov::element::f32}, {3.f}, {33.f}},
      }}},
    // per channel quantization with different values, split by spatial dimension
    {{-1, -1, -1, -1},
     std::int64_t{2},
     size_t{3},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
       {{11.f, 22.f, 33.f, 44.f, 55.f, 66.f}, ov::element::f32, {1, 1, 6, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32},
           {{1.f, 2.f}, ov::element::f32, {1, 1, 2, 1}},
           {{11.f, 22.f}, ov::element::f32, {1, 1, 2, 1}}},
          {{ov::element::f32},
           {{3.f, 4.f}, ov::element::f32, {1, 1, 2, 1}},
           {{33.f, 44.f}, ov::element::f32, {1, 1, 2, 1}}},
          {{ov::element::f32},
           {{5.f, 6.f}, ov::element::f32, {1, 1, 2, 1}},
           {{55.f, 66.f}, ov::element::f32, {1, 1, 2, 1}}},
      }}},
    // U8 per channel quantization with the same values
    {{1, 3, 16, 16},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 1.f, 1.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 11.f, 11.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {1.f}, {11.f}},
          {{ov::element::f32}, {1.f}, {11.f}},
      }}},
    // I8 per channel quantization with the same values
    {{1, 3, 16, 16},
     std::int64_t{1},
     size_t{3},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 1.f, 1.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 11.f, 11.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32}, {1.f}, {11.f}},
       {{ov::element::f32}, {1.f}, {11.f}},
       {{ov::element::f32}, {1.f}, {11.f}}}}},
    // U8 split second dimension
    {{1, 3, 16, 16},
     std::int64_t{-1},
     size_t{2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
       {{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
       {{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}}}},
    // I8 split second dimension
    {{1, 3, 16, 16},
     std::int64_t{-1},
     size_t{2},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
       {{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
       {{ov::element::f32},
        {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
        {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}}}},
    // U8 without subtract
    {{1, 3, 16, 16},
     std::int64_t{-3},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {}, {11.f}},
          {{ov::element::f32}, {}, {22.f}},
          {{ov::element::f32}, {}, {33.f}},
      }}},
    // U8 without subtract, dynamic shape
    {{-1, 3, -1, -1},
     std::int64_t{-3},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {}, {11.f}},
          {{ov::element::f32}, {}, {22.f}},
          {{ov::element::f32}, {}, {33.f}},
      }}},
    // U8 without subtract, dynamic shape (dynamic channels)
    {{-1, -1, -1, -1},
     std::int64_t{-3},
     size_t{3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {}, {11.f}},
          {{ov::element::f32}, {}, {22.f}},
          {{ov::element::f32}, {}, {33.f}},
      }}},
    // I8 without subtract
    {{1, 3, 16, 16},
     std::int64_t{-3},
     size_t{3},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {}, {11.f}},
          {{ov::element::f32}, {}, {22.f}},
          {{ov::element::f32}, {}, {33.f}},
      }}},
    // I8 dequantization in second dimension
    {{1, 4, 3, 3},
     std::int64_t{1},
     size_t{2},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4, 1, 1}},
       {{11.f, 22.f, 33.f, 44.f}, ov::element::f32, {1, 4, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32},
        {{3.f, 4.f}, ov::element::f32, {1, 2, 1, 1}},
        {{33.f, 44.f}, ov::element::f32, {1, 2, 1, 1}}}}}},
    // without Convert
    {{1, 4, 3, 3},
     std::int64_t{1},
     size_t{2},
     LayerTransformation::createParamsI8I8(),
     {ov::element::f32,
      {{},
       {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4, 1, 1}},
       {{11.f, 22.f, 33.f, 44.f}, ov::element::f32, {1, 4, 1, 1}}}},
     {ov::element::f32,
      {},
      ov::element::f32,
      {{{}, {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}}, {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{}, {{3.f, 4.f}, ov::element::f32, {1, 2, 1, 1}}, {{33.f, 44.f}, ov::element::f32, {1, 2, 1, 1}}}}}},
    // no dequantization
    {ov::Shape({1, 3, 4, 4}), std::int64_t{2}, size_t{2}, LayerTransformation::createParamsI8I8(), {}, {}},
};
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         SplitTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions), ::testing::ValuesIn(testValues)),
                         SplitTransformation::getTestCaseName);
}  // namespace
