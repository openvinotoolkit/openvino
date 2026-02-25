// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/variadic_split.hpp"
#include <memory>

#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/variadic_split.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

class VariadicSplitTransformationTestValues {
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
    std::int64_t axis;
    std::vector<size_t> splitLengths;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class VariadicSplitTransformation : public LayerTransformation,
                                    public testing::WithParamInterface<VariadicSplitTransformationTestValues> {
public:
    void SetUp() override {
        const VariadicSplitTransformationTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::VariadicSplitFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.axis,
            testValues.splitLengths);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::VariadicSplitTransformation, ov::op::v1::VariadicSplit>(
            testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::VariadicSplitFunction::getReference(testValues.inputShape,
                                                                           testValues.expected.inputPrecision,
                                                                           testValues.expected.dequantizationBefore,
                                                                           testValues.expected.precisionAfterOperation,
                                                                           testValues.expected.dequantizationAfter,
                                                                           testValues.axis,
                                                                           testValues.splitLengths);
    }

    static std::string getTestCaseName(testing::TestParamInfo<VariadicSplitTransformationTestValues> obj) {
        const VariadicSplitTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << toString(testValues.params) << "_" << testValues.inputShape << "_"
               << testValues.actual.precisionBeforeDequantization << "_" << testValues.actual.dequantization << "_"
               << testValues.expected.dequantizationAfter << "_splitLengths=" << testValues.splitLengths;
        return result.str();
    }
};

TEST_P(VariadicSplitTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<VariadicSplitTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {{1, 3, 16, 16},
     std::int64_t{2},
     std::vector<size_t>{10, 6},
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
    // U8 per tensor quantization
    {{-1, -1, -1, -1},
     std::int64_t{2},
     std::vector<size_t>{10, 6},
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
     std::vector<size_t>{10, 6},
     LayerTransformation::createParamsU8I8(),
     // ActualValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     // ExpectedValues
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}, ov::element::f32, {}}},
    // I8 per tensor quantization
    {{1, 3, 16, 16},
     std::int64_t{2},
     std::vector<size_t>{10, 6},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {128.f}, {3.f}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // U8 per channel quantization with different values
    {{1, 3, 16, 16},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {3.f}, {33.f}}}}},
    // U8 per channel quantization with different values, split by batch
    {{2, 3, 16, 16},
     std::int64_t{0},
     std::vector<size_t>{1, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{2.f, 3.f}, ov::element::f32, {2, 1, 1, 1}},
       {{22.f, 33.f}, ov::element::f32, {2, 1, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32}, {2.f}, {22.f}}, {{ov::element::f32}, {3.f}, {33.f}}}}},
    // U8 per channel quantization with different values, split by spatial dimension
    {{-1, -1, -1, -1},
     std::int64_t{3},
     std::vector<size_t>{4, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 1, 6}},
       {{11.f, 22.f, 33.f, 44.f, 55.f, 66.f}, ov::element::f32, {1, 1, 1, 6}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32},
           {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 1, 1, 4}},
           {{11.f, 22.f, 33.f, 44.f}, ov::element::f32, {1, 1, 1, 4}}},
          {{ov::element::f32},
           {{5.f, 6.f}, ov::element::f32, {1, 1, 1, 2}},
           {{55.f, 66.f}, ov::element::f32, {1, 1, 1, 2}}},
      }}},
    // U8 per channel quantization with different values, dynamic shape
    {{-1, 3, -1, -1},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {3.f}, {33.f}}}}},
    // U8 per channel quantization with different values, dynamic shape (dynamic channels)
    {{-1, -1, -1, -1},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {3.f}, {33.f}}}}},
    // U8 per channel quantization with different values (constants without batch)
    {{1, 3, 16, 16},
     std::int64_t{-3},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {3.f}, {33.f}}}}},
    // I8 per channel quantization with different values
    {{1, 3, 16, 16},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32},
        {{1.f, 2.f}, ov::element::f32, {1, 2, 1, 1}},
        {{11.f, 22.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {3.f}, {33.f}}}}},
    // U8 per channel quantization with the same values
    {{1, 3, 16, 16},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{1.f, 1.f, 1.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 11.f, 11.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {{{ov::element::f32}, {1.f}, {11.f}}, {{ov::element::f32}, {1.f}, {11.f}}}}},
    // I8 per channel quantization with the same values
    {{1, 3, 16, 16},
     std::int64_t{1},
     std::vector<size_t>{2, 1},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 1.f, 1.f}, ov::element::f32, {1, 3, 1, 1}},
       {{11.f, 11.f, 11.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32}, {1.f}, {11.f}}, {{ov::element::f32}, {1.f}, {11.f}}}}},
    // U8 split second dimension
    {{1, 3, 16, 16},
     std::int64_t{-1},
     std::vector<size_t>{10, 4, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // U8 split second dimension, dynamic shape
    {{-1, -1, -1, -1},
     std::int64_t{-1},
     std::vector<size_t>{10, 4, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128.f}, {3.f}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // I8 split second dimension
    {{1, 3, 16, 16},
     std::int64_t{-1},
     std::vector<size_t>{10, 4, 2},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {128.f}, {3.f}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
          {{ov::element::f32}, {128.f}, {3.f}},
      }}},
    // U8 per channel split
    {{1, 4, 224, 224},
     std::int64_t{-3},
     std::vector<size_t>{1, 2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::i8,
      {{ov::element::f32},
       {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4, 1, 1}},
       {{11.f, 22.f, 33.f, 44.f}, ov::element::f32, {1, 4, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {{{ov::element::f32}, {1.f}, {11.f}},
       {{ov::element::f32},
        {{2.f, 3.f}, ov::element::f32, {1, 2, 1, 1}},
        {{22.f, 33.f}, ov::element::f32, {1, 2, 1, 1}}},
       {{ov::element::f32}, {4.f}, {44.f}}}}},
    // U8 without subtract
    {{1, 3, 16, 16},
     std::int64_t{3},
     std::vector<size_t>{1, 1, 14},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {},
      ov::element::u8,
      {
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
      }}},
    // I8 without subtract
    {{1, 3, 16, 16},
     std::int64_t{3},
     std::vector<size_t>{1, 1, 14},
     LayerTransformation::createParamsI8I8(),
     {ov::element::i8, {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::i8,
      {},
      ov::element::i8,
      {
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
          {{ov::element::f32}, {}, {{11.f, 22.f, 33.f}, ov::element::f32, {1, 3, 1, 1}}},
      }}},
    // I8 split second dimension
    {{1, 4, 3, 3},
     std::int64_t{1},
     std::vector<size_t>{2, 2},
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
     std::vector<size_t>{2, 2},
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
    {{1, 3, 4, 4},
     std::int64_t{2},
     std::vector<size_t>{2, 2},
     LayerTransformation::createParamsI8I8(),
     // ActualValues
     {},
     // ExpectedValues
     {}},
};
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         VariadicSplitTransformation,
                         ::testing::ValuesIn(testValues),
                         VariadicSplitTransformation::getTestCaseName);
}  // namespace
