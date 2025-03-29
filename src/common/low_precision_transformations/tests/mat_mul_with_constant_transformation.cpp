// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/mat_mul.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/mat_mul.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

class MatMullTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnData;

        ov::builder::subgraph::Constant weights;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationOnData;
        ov::builder::subgraph::Constant weights;

        ov::element::Type precisionBeforeOperation;
        ov::builder::subgraph::DequantizationOperations resultDequantization;

        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.precisionBeforeDequantization << "_" << actual.dequantizationOnData << "_"
               << actual.weights.shape << "_" << actual.fqOnWeights << "_" << actual.dequantizationOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" << expected.weights.shape << "_" << expected.dequantizationOnData << "_"
               << expected.precisionBeforeOperation << "_" << expected.resultDequantization << "_"
               << expected.fqOnWeights << "_" << expected.dequantizationOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<ov::element::Type, ov::PartialShape, MatMullTransformationTestValues>
    MatMulTransformationParams;

class MatMulWithConstantTransformation : public LayerTransformation,
                                         public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape inputShape = std::get<1>(GetParam());
        MatMullTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction =
            ov::builder::subgraph::MatMulFunction::getOriginal(precision,
                                                                   inputShape,
                                                                   testValues.actual.precisionBeforeDequantization,
                                                                   testValues.actual.dequantizationOnData,
                                                                   testValues.actual.weights,
                                                                   testValues.actual.fqOnWeights,
                                                                   testValues.actual.dequantizationOnWeights);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::MatMulTransformation, ov::op::v0::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            (testValues.expected.fqOnWeights.empty() && testValues.expected.dequantizationOnWeights.empty())
                ? ov::builder::subgraph::MatMulFunction::getReference(
                      precision,
                      inputShape,
                      testValues.expected.precisionBeforeDequantization,
                      testValues.expected.dequantizationOnData,
                      testValues.expected.weights,
                      testValues.expected.resultDequantization)
                : ov::builder::subgraph::MatMulFunction::getOriginal(
                      precision,
                      inputShape,
                      testValues.expected.precisionBeforeDequantization,
                      testValues.expected.dequantizationOnData,
                      testValues.expected.weights,
                      testValues.expected.fqOnWeights,
                      testValues.expected.dequantizationOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ov::element::Type precision;
        ov::PartialShape inputShape;
        MatMullTransformationTestValues testValues;
        std::tie(precision, inputShape, testValues) = obj.param;

        std::stringstream ss;
        ss << precision << "_" << inputShape << "_" << testValues;
        return ss.str();
    }
};

TEST_P(MatMulWithConstantTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    {1, 384, 1024},
    {4, 384, 1024},
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}};

std::vector<MatMullTransformationTestValues> testValues = {
    // supported 3D: U8 & I8
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::f32, ov::Shape{1024, 1024}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {},
      {std::vector<float>(1024 * 1024, -126.f), ov::element::i8, ov::Shape{1024, 1024}},
      ov::element::u8,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}},

    // test: multiply with f16 constant
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, ov::builder::subgraph::DequantizationOperations::Multiply{0.02f}.setConstantPrecision(ov::element::f16)},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::i8, ov::Shape{1024, 1024}},
      {},
      {ov::element::f32, {}, {0.1f}},
     },
     {ov::element::u8,
      {},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::i8, ov::Shape{1024, 1024}},
      ov::element::u8,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}},

    // supported 3D: U8 & I8 with Dq on weights
    {LayerTransformation::createParamsU8I8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         {std::vector<float>(1024 * 1024, 1.f), ov::element::i8, ov::Shape{1024, 1024}},
         {},
         {ov::element::f32, {}, {0.1f}},
     },
     {ov::element::u8,
      {},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::i8, ov::Shape{1024, 1024}},
      ov::element::u8,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulWithConstantTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulWithConstantTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapes = {{1, 3, 4},
                                                       {4, 3, 4},
                                                       {Dimension::dynamic(), 3, Dimension::dynamic()}};

std::vector<MatMullTransformationTestValues> testValues = {
    // 3D: U8 & I8
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {},
      {std::vector<float>(4 * 4, -126.f), ov::element::i8, ov::Shape{4, 4}},
      ov::element::u8,
      {{}, {}, {{0.001f, 0.002f, 0.003f}}},
      {},
      {}}},

    // 3D: U8 & I8
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {128.f}, {0.01f}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {},
      {std::vector<float>(4 * 4, -126.f), ov::element::i8, ov::Shape{4, 4}},
      ov::element::u8,
      {{}, {-64512.f}, {0.001f}},
      {},
      {}}},

    // 3D: U8 & I8 with Dq on weights
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::i8, ov::Shape{4, 4}},
      {},
      {ov::element::f32, {}, {0.1f}}},
     {ov::element::u8,
      {{}, {}, {}},
      {std::vector<float>(4 * 4, 1.f), ov::element::i8, ov::Shape{4, 4}},
      ov::element::u8,
      {{}, {}, {{0.001f, 0.002f, 0.003f}}},
      {},
      {}}},

    // 3D: U8 & I8
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      {
          255,
          {1, 4},
          {0.f, 0.f, 0.f, 0.f},
          {254.f, 254.f, 254.f, 254.f},
          {-127.f, -12.7f, -1.27f, -0.127f},
          {127.f, 12.7f, 1.27f, 0.127f},
      },
      {}},
     {ov::element::u8,
      {{}, {}, {}},
      {std::vector<float>(4 * 4, -126.f), ov::element::i8, ov::Shape{4, 4}},
      ov::element::u8,
      {{}, {}, {{0.02f, 0.002f, 0.0002f, 0.00002f}, ov::element::f32, ov::Shape{1, 1, 4}}},
      {},
      {}}},

    // 3D: U8 & I8 with Dq on weights with different values
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(4 * 4, 1.f), ov::element::i8, ov::Shape{4, 4}},
      {},
      {ov::element::f32, {}, {{1.f, 0.1f, 0.01f, 0.001f}}}},
     {ov::element::u8,
      {{}, {}, {}},
      {std::vector<float>(4 * 4, 1.f), ov::element::i8, ov::Shape{4, 4}},
      ov::element::u8,
      {{}, {}, {{0.02f, 0.002f, 0.0002f, 0.00002f}, ov::element::f32, ov::Shape{1, 1, 4}}},
      {},
      {}}},

    // 3D: U8 & I8: dequantization by columns in first input: can't be transformed
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f, 0.01f}, ov::element::f32, ov::Shape{1, 1, 4}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f, 0.01f}, ov::element::f32, ov::Shape{1, 1, 4}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      ov::element::f32,
      {},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}}},

    // U8 & I8: dequantization by rows in second input: can't be transformed
    {
        LayerTransformation::createParamsU8I8(),
        {ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
         {
             255,
             {4, 1},
             {0.f, 0.f, 0.f, 0.f},
             {254.f, 254.f, 254.f, 254.f},
             {-127.f, -12.7f, -1.27f, -0.127f},
             {127.f, 12.7f, 1.27f, 0.127f},
         },
         {}},
        {ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
         ov::element::f32,
         {},
         {
             255,
             {4, 1},
             {0.f, 0.f, 0.f, 0.f},
             {254.f, 254.f, 254.f, 254.f},
             {-127.f, -12.7f, -1.27f, -0.127f},
             {127.f, 12.7f, 1.27f, 0.127f},
         },
         {}},
    },

    // U8 & I8: dequantization by rows in second input: can't be transformed (Dq on weights)
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {ov::element::f32, {}, {0.02f}},
            {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
            {},
            {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f, 0.01f}, ov::element::f32, ov::Shape{4, 1}}},
        },
        {
            ov::element::u8,
            {ov::element::f32, {}, {0.02f}},
            {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
            ov::element::u8,
            {},
            {},
            {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f, 0.01f}, ov::element::f32, ov::Shape{4, 1}}},
        },
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulWithConstantTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulWithConstantTransformation::getTestCaseName);
}  // namespace testValues2

namespace testValues3 {
const std::vector<ov::PartialShape> inputShapes = {{1, 2048},
                                                       {4, 2048},
                                                       {Dimension::dynamic(), Dimension::dynamic()}};

std::vector<MatMullTransformationTestValues> testValues = {
    // 2D: U8 & I8
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::f32, ov::Shape{2048, 1000}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, -126.f), ov::element::i8, ov::Shape{2048, 1000}},
      ov::element::u8,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}},

    // 2D: U8 & I8 with Dq on weights
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.2f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      {},
      {ov::element::f32, {}, {0.2f}}},
     {ov::element::u8,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      ov::element::u8,
      {{}, {}, {0.2f * 0.2f}},
      {},
      {}}},

    // 2D: I8 & I8
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::f32, ov::Shape{2048, 1000}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::i8,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, -126.f), ov::element::i8, ov::Shape{2048, 1000}},
      ov::element::i8,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}},

    // 2D: I8 & I8 with Dq on weights with small subtract values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      {},
      {ov::element::f32, {1e-7f}, {0.02f}}},
     {ov::element::i8,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      ov::element::i8,
      {{}, {}, {0.02f * 0.02f}},
      {},
      {}}},

    // 2D: I8 & I8 with Dq on weights with zero subtract values
    {LayerTransformation::createParamsI8I8(),
     {ov::element::i8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      {},
      {ov::element::f32, {0.f}, {0.02f}}},
     {ov::element::i8,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::i8, ov::Shape{2048, 1000}},
      ov::element::i8,
      {{}, {}, {0.02f * 0.02f}},
      {},
      {}}},

    // 2D: FP32 & FP32
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     {ov::element::f32,
      {{}, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::f32, ov::Shape{2048, 1000}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::f32,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, -126.f), ov::element::f32, ov::Shape{2048, 1000}},
      ov::element::f32,
      {{}, {}, {0.02f * 0.1f}},
      {},
      {}}},

    // 2D: FP32 & FP32 with Dq on weights
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     {ov::element::f32,
      {{}, {}, {0.02f}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::f32, ov::Shape{2048, 1000}},
      {},
      {ov::element::f32, {}, {0.02f}}},
     {ov::element::f32,
      {{}, {}, {}},
      {std::vector<float>(2048 * 1000, 1.f), ov::element::f32, ov::Shape{2048, 1000}},
      ov::element::f32,
      {{}, {}, {0.02f * 0.02f}},
      {},
      {}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulWithConstantTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulWithConstantTransformation::getTestCaseName);
}  // namespace testValues3

namespace testValues4 {
const std::vector<ov::PartialShape> inputShapes = {PartialShape::dynamic()};

std::vector<MatMullTransformationTestValues> testValues = {
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::f32, ov::Shape{1024, 1024}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {ov::element::f32, {}, {0.02f}},
      {std::vector<float>(1024 * 1024, 1.f), ov::element::f32, ov::Shape{1024, 1024}},
      ov::element::f32,
      {},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}}},
    {LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}},
     {ov::element::u8,
      {ov::element::f32, {}, {{0.01f, 0.02f, 0.03f}}},
      {std::vector<float>(4 * 4, 1.f), ov::element::f32, ov::Shape{4, 4}},
      ov::element::f32,
      {},
      {255, {1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
      {}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulWithConstantTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulWithConstantTransformation::getTestCaseName);
}  // namespace testValues4
}  // namespace
