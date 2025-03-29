// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/mat_mul.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
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
        ov::element::Type precisionBeforeDequantization1;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type precisionBeforeDequantization2;
        ov::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization1;
        ov::builder::subgraph::DequantizationOperations dequantization1;
        ov::element::Type precisionBeforeDequantization2;
        ov::builder::subgraph::DequantizationOperations dequantization2;
        ov::element::Type precisionBeforeOperation1;
        ov::element::Type precisionBeforeOperation2;
        ov::builder::subgraph::DequantizationOperations result;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.dequantization1 << "_" << actual.dequantization2;
}

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" << expected.precisionBeforeDequantization1 << "_" << expected.dequantization1 << "_"
               << expected.precisionBeforeDequantization2 << "_" << expected.dequantization2 << "_"
               << expected.precisionBeforeOperation1 << "_" << expected.precisionBeforeOperation2 << "_"
               << expected.result;
}

inline std::ostream& operator<<(std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.params.supportAsymmetricQuantization << "_" << values.params.updatePrecisions << "_"
               << values.actual << "_" << values.expected;
}

typedef std::
    tuple<ov::element::Type, std::pair<ov::PartialShape, ov::PartialShape>, MatMullTransformationTestValues>
        MatMulTransformationParams;

class MatMulTransformation : public LayerTransformation,
                             public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const std::pair<ov::PartialShape, ov::PartialShape> shapes = std::get<1>(GetParam());
        const MatMullTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction =
            ov::builder::subgraph::MatMulFunction::getOriginal(precision,
                                                                   shapes.first,
                                                                   testValues.actual.precisionBeforeDequantization1,
                                                                   testValues.actual.dequantization1,
                                                                   shapes.second,
                                                                   testValues.actual.precisionBeforeDequantization2,
                                                                   testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::MatMulTransformation, ov::op::v0::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            (testValues.expected.precisionBeforeOperation1 == ov::element::f32) && testValues.expected.result.empty()
                ? ov::builder::subgraph::MatMulFunction::getOriginal(
                      precision,
                      shapes.first,
                      testValues.actual.precisionBeforeDequantization1,
                      testValues.actual.dequantization1,
                      shapes.second,
                      testValues.actual.precisionBeforeDequantization2,
                      testValues.actual.dequantization2)
                : ov::builder::subgraph::MatMulFunction::getReference(
                      precision,
                      shapes.first,
                      testValues.expected.precisionBeforeDequantization1,
                      testValues.expected.dequantization1,
                      shapes.second,
                      testValues.expected.precisionBeforeDequantization2,
                      testValues.expected.dequantization2,
                      testValues.expected.result);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ov::element::Type precision;
        std::pair<ov::PartialShape, ov::PartialShape> shapes;
        MatMullTransformationTestValues testValues;
        std::tie(precision, shapes, testValues) = obj.param;

        std::stringstream ss;
        ss << precision << "_" << shapes.first << "_" << shapes.second << "_" << testValues;
        return ss.str();
    }
};

TEST_P(MatMulTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

namespace testValues1 {
const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapes = {
    {{-1, -1, -1, -1}, {-1, -1, -1, -1}},
    {{1, 16, 384, 64}, {1, 16, 64, 384}},
    {{1, 1, 4, 16, 384, 64}, {1, 1, 4, 16, 64, 384}},
    {{64}, {64}}};

std::vector<MatMullTransformationTestValues> testValues = {
    // U8 + I8: Constant on dequantization operations on 0 branch
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(true),
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f}, ov::element::f32, {}, true, 0}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {},
         ov::element::i8,
         {},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {0.0006f}},
     }},
    // U8 + I8
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::u8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
    // I8 + I8
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::i8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::i8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
    // U8 + I8, Subtract with not int
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::u8,
         {ov::element::f32, {127.5f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {ov::element::f32, {127.5f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
    // U8 + FP32
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::u8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::f32,
         {{}, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {ov::element::f32, {127.f}, {0.02f}},
         ov::element::f32,
         {{}, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {},
     }},
    // FP32 + I8
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::f32,
         {{}, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::f32,
         {{}, {127.f}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {},
     }},
    {LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(false),
     {
         ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {{}, {}, {}},
         ov::element::i8,
         {{}, {}, {}},
         ov::element::u8,
         ov::element::i8,
         {{}, {}, {0.02f * 0.03f}},
     }},
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {{}, {}, {}},
         ov::element::i8,
         {{}, {}, {}},
         ov::element::u8,
         ov::element::i8,
         {{}, {}, {0.02f * 0.03f}},
     }},
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {0.02f}},
         ov::element::u8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {{}, {}, {}},
         ov::element::u8,
         {{}, {}, {}},
         ov::element::u8,
         ov::element::u8,
         {{}, {}, {0.02f * 0.03f}},
     }},
    {LayerTransformation::createParamsI8I8().setUpdatePrecisions(true),
     {
         ov::element::i8,
         {ov::element::f32, {}, {0.02f}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::i8,
         {{}, {}, {}},
         ov::element::i8,
         {{}, {}, {}},
         ov::element::i8,
         ov::element::i8,
         {{}, {}, {0.02f * 0.03f}},
     }},
    {LayerTransformation::createParamsI8I8().setUpdatePrecisions(true),
     {
         ov::element::f32,
         {{}, {}, {0.02f}},
         ov::element::f32,
         {{}, {}, {0.03f}},
     },
     {
         ov::element::f32,
         {{}, {}, {0.02f}},
         ov::element::f32,
         {{}, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
    {LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
     {
         ov::element::f32,
         {{}, {}, {0.02f}},
         ov::element::f32,
         {{}, {}, {0.03f}},
     },
     {
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         {{}, {}, {}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {0.02f * 0.03f}},
     }}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapes = {
    {{-1, -1, -1, -1}, {-1, -1, -1, -1}},
    {{1, 3, 384, 64}, {1, 3, 64, 384}},
};

std::vector<MatMullTransformationTestValues> testValues = {
    // U8 + I8: Constant on dequantization operations on 0 branch
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {},
         ov::element::i8,
         {},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {{0.0006f, 0.0009f, 0.0003f}}},
     }},
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
     },
     {
         ov::element::u8,
         {},
         ov::element::i8,
         {},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {{0.0004f, 0.0009f, 0.0001f}}},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues)),
                         MatMulTransformation::getTestCaseName);
}  // namespace testValues2

namespace testValues3 {
const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapesWithDynamicChannels = {
    {PartialShape::dynamic(), PartialShape::dynamic()}};

std::vector<MatMullTransformationTestValues> testValuesWithPerChannelDq = {
    // U8 + I8: Constant on dequantization operations on 0 branch
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
     },
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {0.03f}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
    {LayerTransformation::createParamsU8U8(),
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
     },
     {
         ov::element::u8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::i8,
         {ov::element::f32, {}, {{0.02f, 0.03f, 0.01f}}},
         ov::element::f32,
         ov::element::f32,
         {{}, {}, {}},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MatMulTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapesWithDynamicChannels),
                                            ::testing::ValuesIn(testValuesWithPerChannelDq)),
                         MatMulTransformation::getTestCaseName);
}  // namespace testValues3
}  // namespace
