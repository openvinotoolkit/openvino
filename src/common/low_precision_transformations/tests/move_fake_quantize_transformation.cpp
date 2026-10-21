// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/concat.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/relu.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "low_precision/move_fake_quantize.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/move_fake_quantize.hpp"
#include "ov_lpt_models/relu.hpp"
#include "simple_low_precision_transformer.hpp"
#include "openvino/op/avg_pool.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/core/model.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

namespace {

class MoveFakeQuantizeTransformationActualValues {
public:
    size_t number_of_operations;
    std::vector<ov::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizeBefore;
    ov::builder::subgraph::DequantizationOperations::Convert convertBefore;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    std::string operation;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
    ov::builder::subgraph::DequantizationOperations::Convert convertAfter;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationActualValues& values) {
    return out << "_" << values.number_of_operations << "_" << values.convertBefore.outPrecision << "_"
               << values.dequantizationBefore << "_" << values.operation << "_" << values.fakeQuantizeAfter << "_"
               << values.convertAfter.outPrecision << "_" << values.dequantizationAfter;
}

class MoveFakeQuantizeTransformationResultValues {
public:
    size_t number_of_operations;
    std::vector<ov::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizeBefore;
    ov::builder::subgraph::DequantizationOperations::Convert convertBefore;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    std::string operation;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
    ov::builder::subgraph::DequantizationOperations::Convert convertAfter;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    ov::element::Type precisionAfterOperation;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationResultValues& values) {
    return out << "_" << values.convertBefore.outPrecision << "_" << values.dequantizationBefore << "_"
               << values.operation << "_" << values.fakeQuantizeAfter << "_" << values.convertAfter << "_"
               << values.dequantizationAfter;
}

class MoveFakeQuantizeTransformationTestValues {
public:
    MoveFakeQuantizeTransformationTestValues() = default;
    MoveFakeQuantizeTransformationTestValues(const TestTransformationParams& params,
                                             const bool multiChannels,
                                             const std::int64_t axis,
                                             const MoveFakeQuantizeTransformationActualValues& actual,
                                             const MoveFakeQuantizeTransformationResultValues& result,
                                             const bool addNotPrecisionPreservedOperation = false,
                                             const bool checkIntervalsAlignmentAttributes = true)
        : params(params),
          multiChannels(multiChannels),
          axis(axis),
          actual(actual),
          result(result) {}

    TestTransformationParams params;
    bool multiChannels;
    std::int64_t axis;
    MoveFakeQuantizeTransformationActualValues actual;
    MoveFakeQuantizeTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::
    tuple<ov::element::Type, std::vector<ov::PartialShape>, MoveFakeQuantizeTransformationTestValues, bool>
        MoveFakeQuantizeTransformationParams;

class MoveFakeQuantizeTransformation : public LayerTransformation,
                                       public testing::WithParamInterface<MoveFakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        std::vector<ov::PartialShape> inputShapes = std::get<1>(GetParam());
        // const auto shape = std::get<1>(GetParam());
        MoveFakeQuantizeTransformationTestValues testValues = std::get<2>(GetParam());
        const bool oneInputWithSplit = std::get<3>(GetParam());
        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantizationBefore.multiply.empty()) {
            testValues.actual.dequantizationBefore.multiply.outPrecision = precision;
        }

        ov::IntervalsAlignmentSharedValue::Interval interval{-1.28f, 2.55f};

        actualFunction = ov::builder::subgraph::MoveFakeQuantize::get(precision,
                                                                          inputShapes,
                                                                          testValues.actual.number_of_operations,
                                                                          testValues.actual.fakeQuantizeBefore,
                                                                          testValues.actual.convertBefore,
                                                                          testValues.actual.dequantizationBefore,
                                                                          testValues.actual.operation,
                                                                          testValues.actual.fakeQuantizeAfter,
                                                                          testValues.actual.convertAfter,
                                                                          testValues.actual.dequantizationAfter,
                                                                          {ov::PrecisionPreservedAttribute(true),
                                                                           ov::IntervalsAlignmentAttribute(interval, 256),
                                                                           ov::QuantizationAlignmentAttribute(false)},
                                                                          testValues.axis,
                                                                          oneInputWithSplit);

        auto supportedPrecisionsOnActivation = std::vector<ov::pass::low_precision::PrecisionsRestriction>(
            {ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::AvgPool>(
                {{{0}, testValues.params.precisionsOnActivations}})});

        auto quantizationRestrictions =
            testValues.multiChannels
                ? std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>()
                : std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>(
                      {ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::AvgPool>()});

        const auto params = TestTransformationParams::toParams(testValues.params);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::low_precision::MoveFakeQuantize>(params);
        manager.run_passes(actualFunction);

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions && (precision == ov::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        referenceFunction =
            ov::builder::subgraph::MoveFakeQuantize::get(precision,
                                                             inputShapes,
                                                             testValues.result.number_of_operations,
                                                             testValues.result.fakeQuantizeBefore,
                                                             testValues.result.convertBefore,
                                                             testValues.result.dequantizationBefore,
                                                             testValues.result.operation,
                                                             testValues.result.fakeQuantizeAfter,
                                                             testValues.result.convertAfter,
                                                             testValues.result.dequantizationAfter,
                                                             {ov::PrecisionPreservedAttribute(true),
                                                              ov::IntervalsAlignmentAttribute(interval, 256),
                                                              ov::QuantizationAlignmentAttribute(false)},
                                                             testValues.axis,
                                                             oneInputWithSplit);
    }
    static std::string getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const std::vector<ov::PartialShape> shape = std::get<1>(obj.param);
        const MoveFakeQuantizeTransformationTestValues testValues = std::get<2>(obj.param);
        const bool oneInputWithSplit = std::get<3>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape[0], testValues.params) << "_"
               << (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") << "axis_" << testValues.axis
               << "_" << testValues.actual << "_" << testValues.result << "_" << oneInputWithSplit;
        return result.str();
    }
};

TEST_P(MoveFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";

    const auto actualFakeQuantizes = LayerTransformation::get<ov::op::v0::FakeQuantize>(actualFunction);
    ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<ov::PrecisionsAttribute>(actualFakeQuantizes))
        << "ov::PrecisionsAttribute are not the same";
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

namespace perTensorValues {
const std::vector<std::vector<ov::PartialShape>> shapes = {{{1, 1, 9, 9}, {1, 1, 9, 9}},
                                                               {{4, 3, 9, 9}, {4, 3, 9, 9}},
                                                               {{-1, -1, -1, -1}, {-1, -1, -1, -1}}};

const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    // without operation
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {2, {}, {}, {}, "", {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {},
         {},
         "",
         {},
         {},
         {},
     }},
    // with ReLU
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {2, {}, {}, {}, "relu", {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {},
         {},
         "relu",
         {},
         {},
         {},
     }},
    // concat by batch
    {LayerTransformation::createParamsU8I8(),
     false,
     0,
     {2, {}, {}, {}, "", {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}, {}, {}},
     {2, {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}}, {}, {}, "", {}, {}, {}}},
    // Q/DQ
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         2,
         {},
         {},
         {},
         "",
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
         "",
         {},
         {},
         {},
     }},
    // Q/DQ with ReLU
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         2,
         {},
         {},
         {},
         "relu",
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
     },
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {ov::element::u8},
         {{element::f32}, {}, {0.01f}},
         "relu",
         {},
         {},
         {},
     }},
    // Q/DQ with subtract
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         2,
         {},
         {},
         {},
         "",
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {ov::element::u8},
         {{element::f32}, {0.01f}, {0.01f}},
     },
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {ov::element::u8},
         {{element::f32}, {0.01f}, {0.01f}},
         "",
         {},
         {},
         {},
     }},
    // F16 Q/DQ with subtract
    {LayerTransformation::createParamsU8I8(),
     false,
     1,
     {
         2,
         {},
         {},
         {},
         "",
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {ov::element::u8},
         {{element::f16}, {0.01f}, {0.01f}},
     },
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {ov::element::u8},
         {{element::f16}, {0.01f}, {0.01f}},
         "",
         {},
         {},
         {},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveFakeQuantizeTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn({false, true})),
                         MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace perTensorValues

namespace perChannelValues {
const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<ov::PartialShape>> shapes = {
    {{1, 1, 224, 224}, {1, 2, 224, 224}},
    {{4, 1, 9, 9}, {4, 2, 9, 9}},
    {{-1, 1, -1, -1}, {-1, 2, -1, -1}},
};

const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    // multi-chanels
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {2,
      {},
      {},
      {},
      "",
      {256ul,
       {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
       {-2.66068696975708f},
       {2.6399004459381104f},
       {-31.695816040039062f, -35.69844055175781f, -49.126914978027344f},
       {277.8320007324219f, 267.07110595703125f, 254.99429321289062f}},
      {},
      {}},
     {
         2,
         {{256ul,
           {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
           {-2.66068696975708f},
           {2.6399004459381104f},
           {-31.695816040039062f},
           {277.8320007324219f}},
          {256ul,
           {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 2, 1, 1}, {1, 2, 1, 1}},
           {-2.66068696975708f},
           {2.6399004459381104f},
           {-35.69844055175781f, -49.126914978027344f},
           {267.07110595703125f, 254.99429321289062f}}},
         {},
         {},
         "",
         {},
         {},
         {},
     }},
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {2,
      {},
      {},
      {},
      "",
      {256ul, {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}, {-2.6f}, {2.6f}, {-31.7f, -35.7f, -49.1f}, {277.8f, 267.f, 254.9f}},
      {},
      {}},
     {
         2,
         {{256ul, {{}, {}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {-2.6f}, {2.6f}, {-31.7f}, {277.8f}},
          {256ul, {{}, {}, {1, 2, 1, 1}, {1, 2, 1, 1}}, {-2.6f}, {2.6f}, {-35.7f, -49.1f}, {267.f, 254.9f}}},
         {},
         {},
         "",
         {},
         {},
         {},
     }},
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {2,
      {},
      {},
      {},
      "",
      {
          256ul,
          {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
          {-31.7f, -35.7f, -49.1f},
          {277.8f, 267.f, 254.9f},
          {-2.6f},
          {2.6f},
      },
      {},
      {}},
     {
         2,
         {{256ul, {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}}, {-31.7f}, {277.8f}, {-2.6f}, {2.6f}},
          {256ul, {{1, 2, 1, 1}, {1, 2, 1, 1}, {}, {}}, {-35.7f, -49.1f}, {267.f, 254.9f}, {-2.6f}, {2.6f}}},
         {},
         {},
         "",
         {},
         {},
         {},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveFakeQuantizeTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn({false})),
                         MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace perChannelValues

namespace testValues3 {
const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<ov::PartialShape>> shapes = {{{1, 1}, {1, 2}}, {{4, 1}, {4, 2}}};
const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    // 2D shape
    {LayerTransformation::createParamsU8I8(),
     true,
     1,
     {2,
      {},
      {},
      {},
      "",
      {
          256ul,
          {{1, 3}, {1, 3}, {}, {}},
          {-31.7f, -35.7f, -49.1f},
          {277.8f, 267.f, 254.9f},
          {-2.6f},
          {2.6f},
      },
      {},
      {}},
     {
         2,
         {{256ul, {{1, 1}, {1, 1}, {}, {}}, {-31.7f}, {277.8f}, {-2.6f}, {2.6f}},
          {256ul, {{1, 2}, {1, 2}, {}, {}}, {-35.7f, -49.1f}, {267.f, 254.9f}, {-2.6f}, {2.6f}}},
         {},
         {},
         "",
         {},
         {},
         {},
     }},
};
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveFakeQuantizeTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn({false})),
                         MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace testValues3

namespace NegativeTestValues {
const std::vector<ov::element::Type> precisions = {ov::element::f32};

const std::vector<std::vector<ov::PartialShape>> shapes = {
    {{-1, -1, -1, -1}, {-1, -1, -1, -1}},
};
const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {2,
         {},
         {},
         {},
         "",
         {
             256ul,
             {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
             {-31.7f, -35.7f, -49.1f},
             {277.8f, 267.f, 254.9f},
             {-2.6f},
             {2.6f},
         },
         {},
         {}},
        {2,
         {},
         {},
         {},
         "",
         {
             256ul,
             {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
             {-31.7f, -35.7f, -49.1f},
             {277.8f, 267.f, 254.9f},
             {-2.6f},
             {2.6f},
         },
         {},
         {}},
    },
};
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveFakeQuantizeTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn({false})),
                         MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace NegativeTestValues

namespace axisBeyondDequantizationRank {
// Concat on the last axis (3) whose dequantization constant is only rank-2. The split decision in
// moveDequantizationBefore normalizes the axis against the Concat rank (4) and then indexes the
// dequantization constant's own shape with it, so it must bounds-check that shape first. Covers
// both the subtract and the multiply branch of that decision.
const std::vector<ov::element::Type> precisions = {ov::element::f32};

const std::vector<std::vector<ov::PartialShape>> shapes = {{{1, 1, 9, 9}, {1, 1, 9, 9}}};
const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    {LayerTransformation::createParamsU8I8(),
     false,
     3,  // concat axis (last) exceeds the rank-2 dequantization constant
     {
         2,
         {},
         {},
         {},
         "",
         {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
         {ov::element::u8},
         {{element::f32}, {{0.01f}, element::f32, {1, 1}}, {{0.01f}, element::f32, {1, 1}}},
     },
     {
         2,
         {{256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
         {ov::element::u8},
         {{element::f32}, {{0.01f}, element::f32, {1, 1}}, {{0.01f}, element::f32, {1, 1}}},
         "",
         {},
         {},
         {},
     }},
};
INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MoveFakeQuantizeTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn({false})),
                         MoveFakeQuantizeTransformation::getTestCaseName);
}  // namespace axisBeyondDequantizationRank
}  // namespace

// Direct moveDequantizationBefore tests for Concat shapes that the only pipeline caller
// (MoveFakeQuantize) doesn't produce today: a dynamic-rank Concat, and dequantization constants of
// lower rank than the Concat, which broadcast against it aligned to the trailing dimensions.
namespace {
struct ConcatDequantization {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<opset1::Concat> concat;
    std::shared_ptr<opset1::Result> result;
};

ConcatDequantization makeConcatDequantization(const PartialShape& inputShape,
                                              const int64_t axis,
                                              const Shape& constantShape,
                                              const bool scalarSubtract,
                                              const bool scalarMultiply) {
    const auto makeConstant = [](const Shape& shape, const float step) {
        std::vector<float> values(shape_size(shape));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = step * static_cast<float>(i + 1);
        }
        return opset1::Constant::create(element::f32, shape, values);
    };
    const auto input1 = std::make_shared<opset1::Parameter>(element::u8, inputShape);
    const auto input2 = std::make_shared<opset1::Parameter>(element::u8, inputShape);
    const auto concat = std::make_shared<opset1::Concat>(OutputVector{input1, input2}, axis);
    const auto convert = std::make_shared<opset1::Convert>(concat, element::f32);
    const auto subtract =
        std::make_shared<opset1::Subtract>(convert, makeConstant(scalarSubtract ? Shape{} : constantShape, 1.f));
    const auto multiply =
        std::make_shared<opset1::Multiply>(subtract, makeConstant(scalarMultiply ? Shape{} : constantShape, 0.1f));
    const auto result = std::make_shared<opset1::Result>(multiply);
    const auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input1, input2});
    return {model, concat, result};
}

// Moves the dequantization in front of the Concat inputs and returns the Subtract and Multiply
// constants that ended up in front of each input.
std::vector<std::pair<std::shared_ptr<opset1::Constant>, std::shared_ptr<opset1::Constant>>> moveAndGetConstants(
    const ConcatDequantization& graph) {
    const auto dequantization = low_precision::NetworkHelper::getDequantizationBelow(graph.concat);
    low_precision::NetworkHelper::moveDequantizationBefore(graph.concat, dequantization, true);
    graph.model->validate_nodes_and_infer_types();

    std::vector<std::pair<std::shared_ptr<opset1::Constant>, std::shared_ptr<opset1::Constant>>> constants;
    const auto newConcat = ov::as_type_ptr<opset1::Concat>(graph.result->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(newConcat != nullptr, "Concat is expected in front of the Result");
    for (size_t i = 0; i < newConcat->get_input_size(); ++i) {
        const auto multiply = ov::as_type_ptr<opset1::Multiply>(newConcat->get_input_node_shared_ptr(i));
        OPENVINO_ASSERT(multiply != nullptr, "Multiply is expected in front of Concat input ", i);
        const auto subtract = ov::as_type_ptr<opset1::Subtract>(multiply->get_input_node_shared_ptr(0));
        OPENVINO_ASSERT(subtract != nullptr, "Subtract is expected in front of Concat input ", i);
        constants.emplace_back(ov::as_type_ptr<opset1::Constant>(subtract->get_input_node_shared_ptr(1)),
                               ov::as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(1)));
    }
    return constants;
}
}  // namespace

// With a dynamic rank the Concat axis can't be mapped onto a non-scalar constant, so whether it
// must be split per input is unknown: reject instead of leaving it unsplit.
TEST(LPT, MoveDequantizationBeforeDynamicRankConcatNonScalarMultiply) {
    const auto graph = makeConcatDequantization(PartialShape::dynamic(), 1, Shape{2, 1}, true, false);
    EXPECT_THROW(moveAndGetConstants(graph), ov::Exception);
}

TEST(LPT, MoveDequantizationBeforeDynamicRankConcatNonScalarSubtract) {
    const auto graph = makeConcatDequantization(PartialShape::dynamic(), 1, Shape{2, 1}, false, true);
    EXPECT_THROW(moveAndGetConstants(graph), ov::Exception);
}

TEST(LPT, MoveDequantizationBeforeDynamicRankConcatScalar) {
    const auto graph = makeConcatDequantization(PartialShape::dynamic(), 1, Shape{}, true, true);
    std::vector<std::pair<std::shared_ptr<opset1::Constant>, std::shared_ptr<opset1::Constant>>> constants;
    ASSERT_NO_THROW(constants = moveAndGetConstants(graph));
    ASSERT_EQ(constants.size(), 2);
    for (const auto& constant : constants) {
        ASSERT_NE(constant.first, nullptr);
        ASSERT_NE(constant.second, nullptr);
        EXPECT_EQ(constant.first->get_shape(), Shape{});
        EXPECT_EQ(constant.second->get_shape(), Shape{});
    }
}

// Rank-2 {1, 6} constants under a rank-4 Concat on axis 3: aligned to the trailing dims, the
// constant's last dim is the Concat axis, so each input gets its own 3-element slice.
TEST(LPT, MoveDequantizationBeforeLowRankConstantSplitOnAlignedAxis) {
    const auto graph = makeConcatDequantization(PartialShape{1, 1, 2, 3}, 3, Shape{1, 6}, false, false);
    std::vector<std::pair<std::shared_ptr<opset1::Constant>, std::shared_ptr<opset1::Constant>>> constants;
    ASSERT_NO_THROW(constants = moveAndGetConstants(graph));
    ASSERT_EQ(constants.size(), 2);
    const std::vector<std::vector<float>> subtractValues = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    const std::vector<std::vector<float>> multiplyValues = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};
    for (size_t i = 0; i < constants.size(); ++i) {
        ASSERT_NE(constants[i].first, nullptr);
        ASSERT_NE(constants[i].second, nullptr);
        EXPECT_EQ(constants[i].first->get_shape(), (Shape{1, 1, 1, 3}));
        EXPECT_EQ(constants[i].second->get_shape(), (Shape{1, 1, 1, 3}));
        EXPECT_EQ(constants[i].first->cast_vector<float>(), subtractValues[i]);
        EXPECT_EQ(constants[i].second->cast_vector<float>(), multiplyValues[i]);
    }
}

// The same {1, 6} constants under a rank-4 Concat on axis 1: aligned to the trailing dims, the
// constant broadcasts along the Concat axis, so every input gets it unsplit.
TEST(LPT, MoveDequantizationBeforeLowRankConstantBroadcastOnConcatAxis) {
    const auto graph = makeConcatDequantization(PartialShape{1, 1, 2, 6}, 1, Shape{1, 6}, false, false);
    std::vector<std::pair<std::shared_ptr<opset1::Constant>, std::shared_ptr<opset1::Constant>>> constants;
    ASSERT_NO_THROW(constants = moveAndGetConstants(graph));
    ASSERT_EQ(constants.size(), 2);
    for (const auto& constant : constants) {
        ASSERT_NE(constant.first, nullptr);
        ASSERT_NE(constant.second, nullptr);
        EXPECT_EQ(constant.first->get_shape(), (Shape{1, 6}));
        EXPECT_EQ(constant.second->get_shape(), (Shape{1, 6}));
    }
}
