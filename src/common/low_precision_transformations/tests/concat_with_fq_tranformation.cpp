// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/common/precisions_restriction.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/concat.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

namespace {

class ConcatTransformationActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ov::builder::subgraph::DequantizationOperations::Convert convert1;
    ov::builder::subgraph::DequantizationOperations dequantization1;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ov::builder::subgraph::DequantizationOperations::Convert convert2;
    ov::builder::subgraph::DequantizationOperations dequantization2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.convert1.outPrecision << "_" << values.dequantization1
               << "_" << values.fakeQuantize2 << "_" << values.convert2.outPrecision << "_" << values.dequantization2;
}

class ConcatTransformationResultValues {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ov::builder::subgraph::DequantizationOperations::Convert convert1;
    ov::builder::subgraph::DequantizationOperations dequantization1;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ov::builder::subgraph::DequantizationOperations::Convert convert2;
    ov::builder::subgraph::DequantizationOperations dequantization2;
    ov::element::Type precisionAfterOperation;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.convert1.outPrecision << "_" << values.dequantization1
               << "_" << values.fakeQuantize2 << "_" << values.convert2.outPrecision << "_" << values.dequantization2
               << "_" << values.dequantizationAfter;
}

class ConcatTransformationTestValues {
public:
    ConcatTransformationTestValues() = default;
    ConcatTransformationTestValues(const TestTransformationParams& params,
                                   const bool multiChannels,
                                   const std::int64_t axis,
                                   const ConcatTransformationActualValues& actual,
                                   const ConcatTransformationResultValues& result,
                                   const bool addNotPrecisionPreservedOperation = false,
                                   const bool checkIntervalsAlignmentAttributes = true)
        : params(params),
          multiChannels(multiChannels),
          axis(axis),
          actual(actual),
          result(result),
          addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
          checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

    TestTransformationParams params;
    bool multiChannels;
    std::int64_t axis;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
    bool checkIntervalsAlignmentAttributes;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple<ov::element::Type, ov::PartialShape, ConcatTransformationTestValues>
    ConcatTransformationParams;

class ConcatWithFQTransformation : public LayerTransformation,
                                   public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }
        actualFunction = ov::builder::subgraph::ConcatFunction::get(precision,
                                                                        shape,
                                                                        testValues.actual.fakeQuantize1,
                                                                        testValues.actual.convert1,
                                                                        testValues.actual.dequantization1,
                                                                        testValues.actual.fakeQuantize2,
                                                                        testValues.actual.convert2,
                                                                        testValues.actual.dequantization2,
                                                                        {},
                                                                        ov::element::dynamic,
                                                                        {},
                                                                        testValues.axis,
                                                                        testValues.addNotPrecisionPreservedOperation);
        auto supportedPrecisionsOnActivation = std::vector<ov::pass::low_precision::PrecisionsRestriction>(
            {ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::AvgPool>(
                {{{0}, testValues.params.precisionsOnActivations}})});

        auto quantizationRestrictions =
            testValues.multiChannels
                ? std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>()
                : std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>(
                      {ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::AvgPool>()});

        const auto params = TestTransformationParams::toParams(testValues.params);
        SimpleLowPrecisionTransformer transformer(supportedPrecisionsOnActivation, quantizationRestrictions);
        transformer.commonGraphRewrite
            ->add_matcher<ov::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
        transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::ConcatTransformation>(params);
        transformer.transform(actualFunction);

        {
            ov::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager
                .register_pass<ov::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        {
            ov::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager
                .register_pass<ov::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions && (precision == ov::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        ov::IntervalsAlignmentSharedValue::Interval interval{-1.28f, 2.55f};

        referenceFunction =
            ov::builder::subgraph::ConcatFunction::get(precision,
                                                           shape,
                                                           testValues.result.fakeQuantize1,
                                                           testValues.result.convert1,
                                                           testValues.result.dequantization1,
                                                           testValues.result.fakeQuantize2,
                                                           testValues.result.convert2,
                                                           testValues.result.dequantization2,
                                                           {ov::PrecisionPreservedAttribute(true),
                                                            ov::IntervalsAlignmentAttribute(interval, 256),
                                                            ov::QuantizationAlignmentAttribute(false)},
                                                           testValues.result.precisionAfterOperation,
                                                           testValues.result.dequantizationAfter,
                                                           testValues.axis,
                                                           testValues.addNotPrecisionPreservedOperation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const ov::PartialShape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_"
               << (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") << "axis_" << testValues.axis
               << "_" << testValues.actual << "_" << testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithFQTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";

    ConcatTransformationTestValues testValues = std::get<2>(GetParam());
    const auto actualFakeQuantizes = LayerTransformation::get<ov::op::v0::FakeQuantize>(actualFunction);
    if (testValues.axis == 1) {
        ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<ov::PrecisionsAttribute>(actualFakeQuantizes))
            << "ov::PrecisionsAttribute are not the same";

        if (testValues.checkIntervalsAlignmentAttributes) {
            auto operations = LayerTransformation::get<ov::op::v0::Concat>(actualFunction);
            operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
            ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<ov::IntervalsAlignmentAttribute>(operations))
                << "ov::IntervalsAlignmentAttribute are not the same";
        }
    }
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {{1, 3, 9, 9}, {4, 3, 9, 9}, {-1, 3, 9, -1}};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8: concat: levels less then threshold is ignored, function is not transformed
    // U8: concat: per-channel quantization: function is transformed
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{256ul, {}, {0.f}, {2550.f}, {0.f}, {2550.f}}, {}, {}, {256ul, {}, {0.f}, {0.1f}, {0.f}, {0.1f}}},
     {
         {256ul,
          {},
          {0.f},
          {2550.f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {256ul,
          {},
          {0.f},
          {0.1f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::u8,
         {ov::element::f32, {}, {{10.f, 10.f, 10.f, 0.000392157f, 0.000392157f, 0.000392157f}}},
     },
     true},
    // right branch is not quantized
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {}},
     {
         {256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {2.55f},
          ov::element::f32,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {},
         {},
         {},
         ov::element::f32,
     }},
    // left branch is not quantized
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
     {
         {},
         {},
         {},
         {256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {2.55f},
          ov::element::f32,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::f32,
     }},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
     {
         {256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::u8,
         {ov::element::f32, {}, {0.01f}},
     }},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f}},
      {},
      {},
      {256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f}},
      {},
      {}},
     {{256ul,
       {{1}, {1}, {}, {}},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1}, {1}, {}, {}},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f}},
      {},
      {},
      {256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {0.f}, {2.55f}, {0.f}, {2.55f}},
      {},
      {}},
     {{256ul,
       {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {0.01f}}}},
    // U8: concat multi channels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {0.f}, {1.275f}, {0.f}, {1.275f}}, {}, {}},
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {1.275f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f}}}}},
    // U8: concat multi channels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{256ul, {{1}, {1}, {1}, {1}}, {0.f}, {2.55f}, {0.f}, {2.55f}},
      {},
      {},
      {256ul, {{1}, {1}, {1}, {1}}, {0.f}, {1.275f}, {0.f}, {1.275f}},
      {},
      {}},
     {{256ul,
       {{1}, {1}, {}, {}},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1}, {1}, {}, {}},
       {0.f},
       {1.275f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f}}}}},
    // U8: concat multi channels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f, 2.55f, 2.55f},
       {0.f, 0.f, 0.f},
       {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}},
      {},
      {},
      {256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {1.275f, 1.275f, 1.275f},
       {0.f, 0.f, 0.f},
       {1.275f / 1.f, 1.275f / 2.f, 1.275f / 3.f}},
      {},
      {}},
     {{256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
       {0.f, 0.f, 0.f},
       {2.55f, 2.55f, 2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
       {0.f, 0.f, 0.f},
       {1.275f, 1.275f, 1.275f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {{0.01f / 1.f, 0.01f / 2.f, 0.01f / 3.f, 0.005f / 1.f, 0.005f / 2.f, 0.005f / 3.f}}}},
     false,
     false},
    // I8
    {LayerTransformation::createParamsI8I8(),
     false,
     1,
     {{256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
      {},
      {},
      {256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
      {},
      {}},
     {{256ul,
       {},
       {-1.28f},
       {1.27f},
       {-128.f},
       {127.f},
       ov::element::i8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {-1.28f},
       {1.27f},
       {-128.f},
       {127.f},
       ov::element::i8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::i8,
      {ov::element::f32, {}, {0.01f}}}},
    // mixed: U8 + I8: concat (check constant values here)
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}}, {}, {}},
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{-1.28f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {-1.28f},
       {1.27f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{-1.28f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {{0.f, 0.f, 0.f, 128.f, 128.f, 128.f}}, {0.01f}}}},
    // mixed: U8 + I8: concat multi channels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}}, {}, {}},
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{-1.28f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {-1.28f},
       {1.27f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{-1.28f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {{0.f, 0.f, 0.f, 128.f, 128.f, 128.f}}, {0.01f}}}},
    // mixed: I8 + U8: concat (check constant values here)
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {{256ul, {}, {-1.28f}, {1.27f}, {0.f}, {170.f}, ov::element::u8},
      {},
      {},
      {256ul, {}, {0.f}, {2.55f}, {85.f}, {255.f}, ov::element::u8},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {85}, {0.015f}}},
     true},
    // real case from ctdet_coco_dlav0_384 model, coverage bad rounding
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {{256ul, {}, {-1.28f}, {1.27f}, {0.f}, {2.3007815f}},
      {},
      {},
      {256ul, {}, {0.f}, {2.55f}, {-3.873046875f}, {3.84375}},
      {},
      {}},
     {{256ul, {}, {-1.28f}, {1.27f}, {128.f}, {204.f}, ov::element::u8},
      {},
      {},
      {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {128}, {0.0302619f}}},
     true},
    // U8: concat multi channels with subtract, negative axis
    {LayerTransformation::createParamsU8I8(),
     true,
     -3,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f}}, {}, {}},
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8},
      {},
      {},
      {256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f}, ov::element::u8},
      {},
      {},
      ov::element::u8,
      {ov::element::f32,
       {{0.f, 0.f, 0.f, -255.f, -255.f, -255.f}},
       {{0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f}}}}},
    // U8: concat multi channels, concatenation by spatial dimension
    {
        LayerTransformation::createParamsU8I8(),
        true,
        2,
        {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {0.f}, {1.275f}, {0.f}, {1.275f}}, {}, {}},
        {{256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {256ul,
          {},
          {0.f},
          {1.275f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::u8,
         {ov::element::f32,
          {},
          {
              // Multiply
              {0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.01f,
               0.005f,
               0.005f,
               0.005f,
               0.005f,
               0.005f,
               0.005f,
               0.005f,
               0.005f,
               0.005f},          // Values
              ov::element::f32,  // Precision
              {1, 1, 18, 1}      // Shape
          }}},
    },
    // U8: concat with subtract convert and subtract without convert
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {127.f}, {0.01f}},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {{127}, element::f32, {}, false, 1ul, ov::element::u8, true, {}, {}}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {127}, {0.01f}}}},
    // U8: concat multi channels with subtract convert and subtract without convert
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
          {0.f, 0.f, 0.f},
          {255.f, 255.f, 255.f}},
         {ov::element::u8},
         {{element::f32}, {{127, 127, 127}}, {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}},
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
          {0.f, 0.f, 0.f},
          {255.f, 255.f, 255.f}},
         {ov::element::u8},
         {{element::f32}, {{127, 127, 127}}, {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}},
     },
     {{256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
       {0.f, 0.f, 0.f},
       {255.f, 255.f, 255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
       {0.f, 0.f, 0.f},
       {255.f, 255.f, 255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {127}, {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f, 0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}}}},
    // U8: concat with subtract convert on both branches
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {{128}, element::f32, {}, false, 1ul, ov::element::u8, true, {}, {}}, {0.01f}},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {{127}, element::f32, {}, false, 1ul, ov::element::u8, true, {}, {}}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32,
       {{128, 128, 128, 127, 127, 127}, element::f32, {1, 6, 1, 1}, false, 1ul, ov::element::u8, true, {}, {}},
       {0.01f}}}},
    // U8: concat with subtract convert
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {ov::element::u8},
         {{element::f32}, {{127}, element::f32, {}, false, 1ul, ov::element::u8, true, {}, {}}, {0.01f}},
     },
     {{256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {},
       {0.f},
       {2.55f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32,
       {{0, 0, 0, 127, 127, 127}, element::f32, {1, 6, 1, 1}, false, 1ul, ov::element::u8, true, {}, {}},
       {0.01f}}}},
    // U8: concat multi channels with subtract convert
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f},
          {0.f, 0.f, 0.f},
          {2.55f, 2.55f, 2.55f}},
         {},
         {},
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
          {0.f, 0.f, 0.f},
          {255.f, 255.f, 255.f}},
         {ov::element::u8},
         {{element::f32},
          {{128, 128, 128}, element::f32, {1, 3, 1, 1}, false, 1ul, ov::element::u8, true},
          {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}},
     },
     {{256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
       {0.f, 0.f, 0.f},
       {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f},
       {0.f},
       {255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
       {0.f, 0.f, 0.f},
       {255.f, 255.f, 255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32,
       {{0.f, 0.f, 0.f, 128.f, 128.f, 128.f}, element::f32, {1, 6, 1, 1}, false, 1ul, ov::element::u8, true},
       {{0.01f, 0.01f, 0.01f, 0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}}}},
    // U8: concat multi channels with subtract convert on both branches
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
          {0.f, 0.f, 0.f},
          {255.f, 255.f, 255.f}},
         {ov::element::u8},
         {{element::f32},
          {{128, 128, 128}, element::f32, {1, 3, 1, 1}, false, 1ul, ov::element::u8, true},
          {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}},
         {256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
          {0.f, 0.f, 0.f},
          {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
          {0.f, 0.f, 0.f},
          {255.f, 255.f, 255.f}},
         {ov::element::u8},
         {{element::f32},
          {{127, 127, 127}, element::f32, {1, 3, 1, 1}, false, 1ul, ov::element::u8, true},
          {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}},
     },
     {{256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
       {0.f, 0.f, 0.f},
       {255.f, 255.f, 255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      {256ul,
       {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {0.f, 0.f, 0.f},
       {2.55f / 3.f, 2.55f / 2.f, 2.55f / 1.f},
       {0.f, 0.f, 0.f},
       {255.f, 255.f, 255.f},
       ov::element::u8,
       {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
      {},
      {},
      ov::element::u8,
      {ov::element::f32,
       {{128, 128, 128, 127, 127, 127}, element::f32, {1, 6, 1, 1}, false, 1ul, ov::element::u8, true, {}, {}},
       {{0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f, 0.01f / 3.f, 0.01f / 2.f, 0.01f / 1.f}}}}},
    // U8: concat multi channels with subtract
    // Features:
    //  1. fakeQuantize1 defines precision
    //  2. fakeQuantize2 has zero point (doesn't define precision)
    //  3. FakeQuantize operations order is not important.
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f}},
         {},
         {}},
        {{256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {256ul,
          {},
          {1.275f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::u8,
         {ov::element::f32,
          {{0.f, 0.f, 0.f, -255.f, -255.f, -255.f}},
          {{0.01f, 0.01f, 0.01f, 0.005f, 0.005f, 0.005f}}}},
    },
    // U8: concat multi channels with subtract
    // Features:
    //  1. fakeQuantize2 has zero point (doesn't define precision)
    //  2. fakeQuantize1 defines precision
    //  3. FakeQuantize operations order is not important.
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {{256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f}},
         {},
         {},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {}},
        {{256ul,
          {},
          {1.275f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         {256ul,
          {},
          {0.f},
          {2.55f},
          {0.f},
          {255.f},
          ov::element::u8,
          {ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{0.f, 2.55f}, 256ul)}},
         {},
         {},
         ov::element::u8,
         {ov::element::f32,
          {{-255.f, -255.f, -255.f, 0.f, 0.f, 0.f}},
          {{0.005f, 0.005f, 0.005f, 0.01f, 0.01f, 0.01f}}}},
    },
    // not update precisions
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     false,
     1,
     {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {},
         {},
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
         {},
         {},
         ov::element::f32,
         {{element::f32}, {}, {0.01f}},
     }},
    // INT4+INT8 quantization levels, concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {{16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f}}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
        {
            {16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f}},
            {},
            {},
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {},
            {},
            ov::element::f32,
            {},
        },
        true,
        false,
    },
    // INT4+INT8 quantization levels, concat multi channels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {{16ul, {}, {0.f}, {1.5f}, {0.f}, {1.5f}}, {}, {}, {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {{16ul, {}, {0.f}, {1.5f}, {0.f}, {15.f}, ov::element::u8},
      {},
      {},
      {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8},
      {},
      {},
      ov::element::u8,
      {ov::element::f32, {}, {{0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.01f}}}},
     true,
     false}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         ConcatWithFQTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues)),
                         ConcatWithFQTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicChannels = {
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
};

const std::vector<ConcatTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f}},
         {},
         {}},
        {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {},
         {},
         {256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f}},
         {},
         {}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         ConcatWithFQTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapesWithDynamicChannels),
                                            ::testing::ValuesIn(testValues)),
                         ConcatWithFQTransformation::getTestCaseName);
}  // namespace testValues2

namespace testValues3 {
const std::vector<ov::PartialShape> shapesWithDynamicChannels = {PartialShape::dynamic()};

const std::vector<ConcatTransformationTestValues> testValues = {
    // issue #58915
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    true,
    //    1,
    //    {
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        {},
    //        {},
    //        { 256ul, {}, {1.275f}, {2.55f}, {1.275f}, {2.55f} },
    //        {},
    //        {}
    //    },
    //    {
    //        { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8, },
    //        {},
    //        {{ov::element::f32}, {}, {0.01f}},
    //        { 256ul, {}, {1.275f}, {2.55f}, {0.f}, {255.f}, ov::element::u8 },
    //        {},
    //        {{ov::element::f32}, {-255.f}, {0.005f}},
    //        ov::element::f32,
    //        {},
    //    },
    //},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         ConcatWithFQTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapesWithDynamicChannels),
                                            ::testing::ValuesIn(testValues)),
                         ConcatWithFQTransformation::getTestCaseName);
}  // namespace testValues3
}  // namespace
