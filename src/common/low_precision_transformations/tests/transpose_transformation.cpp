// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/transpose.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/transpose.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov::pass;
using namespace ov;

class TransposeTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    std::vector<int> transposeConstValues;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::PartialShape, TransposeTransformationTestValues> TransposeTransformationParams;

class TransposeTransformation : public LayerTransformation,
                                public testing::WithParamInterface<TransposeTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const TransposeTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction =
            ov::builder::subgraph::TransposeFunction::getOriginal(inputShape,
                                                                      testValues.transposeConstValues,
                                                                      testValues.actual.precisionBeforeDequantization,
                                                                      testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::TransposeTransformation, ov::op::v1::Transpose>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::TransposeFunction::getReference(
            inputShape,
            testValues.transposeConstValues,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<TransposeTransformationParams> obj) {
        const ov::PartialShape inputShape = std::get<0>(obj.param);
        const TransposeTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << inputShape << "_" << testValues.transposeConstValues << "_"
               << testValues.actual.precisionBeforeDequantization << "_" << testValues.actual.dequantization << "_"
               << testValues.expected.dequantizationBefore;
        return result.str();
    }
};

TEST_P(TransposeTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes4D = {{1, 3, 16, 16}, {-1, -1, -1, -1}};

const std::vector<TransposeTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {{0, 1, 3, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {{128}, ov::element::f32, {}, true, 1, ov::element::u8, true}, {0.1f}}},
     {ov::element::u8,
      {{}, {}, {}},
      ov::element::u8,
      {{ov::element::f32}, {{128}, ov::element::f32, {}, true, 1, ov::element::u8, true}, {0.1f}}}},
    // U8: per-tensor quantization
    {{0, 1, 3, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}},
     {ov::element::u8, {{}, {}, {}}, ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}}},
    // U8: per-channel quantization
    {{0, 1, 3, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{128, 64, 32}, ov::element::f32, {1, 3, 1, 1}},
       {{0.3f, 0.2f, 0.1f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {{}, {}, {}},
      ov::element::u8,
      {{ov::element::f32},
       {{128, 64, 32}, ov::element::f32, {1, 3, 1, 1}},
       {{0.3f, 0.2f, 0.1f}, ov::element::f32, {1, 3, 1, 1}}}}},
    // U8: per-channel quantization with the same values,
    // subtraction with Convert from u8 to fp32, transpose channel dimension
    {{0, 3, 1, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
       {{0.1f}, ov::element::f32, {1, 3, 1, 1}}}},
     {ov::element::u8,
      {{}, {}, {}},
      ov::element::u8,
      {{ov::element::f32},
       {{128.f}, element::dynamic, {1, 1, 3, 1}, false, 1ul, element::u8, true},
       {{0.1f}, ov::element::f32, {1, 1, 3, 1}}}}},
    // U8: per-tensor quantization, transpose channel dimension
    {{0, 3, 1, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}},
     {ov::element::u8, {{}, {}, {}}, ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}}},
    // U8: per-channel quantization, transpose channel dimension
    {{0, 2, 1, 3},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32},
       {{128, 64, 32}, ov::element::f32, {1, 3, 1, 1}},
       {{0.3f, 0.2f, 0.1f}, ov::element::f32, {1, 3, 1, 1}}}},
     {
         ov::element::u8,
         {{ov::element::f32},
          {{128, 64, 32}, ov::element::f32, {1, 3, 1, 1}},
          {{0.3f, 0.2f, 0.1f}, ov::element::f32, {1, 3, 1, 1}}},
         ov::element::f32,
         {{}, {}, {}},
     }},
    // empty
    {{0, 1, 3, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {}},
     {ov::element::u8, {}, ov::element::u8, {}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         TransposeTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D), ::testing::ValuesIn(testValues)),
                         TransposeTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapes3D = {{1, 16, 512}, {-1, -1, -1}};

const std::vector<TransposeTransformationTestValues> testValues = {
    {{0, 2, 1},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}},
     {ov::element::u8, {{}, {}, {}}, ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         TransposeTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D), ::testing::ValuesIn(testValues)),
                         TransposeTransformation::getTestCaseName);
}  // namespace testValues2

namespace testValues3 {
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {PartialShape::dynamic()};

const std::vector<TransposeTransformationTestValues> testValues = {
    {{0, 1, 3, 2},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8,
      {{ov::element::f32}, {{128}, ov::element::f32, {}, true, 1, ov::element::u8, true}, {0.1f}}},
     {ov::element::u8,
      {{}, {}, {}},
      ov::element::u8,
      {{ov::element::f32}, {{128}, ov::element::f32, {}, true, 1, ov::element::u8, true}, {0.1f}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         TransposeTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapesWithDynamicRank),
                                            ::testing::ValuesIn(testValues)),
                         TransposeTransformation::getTestCaseName);
}  // namespace testValues3

namespace testValues4 {
const std::vector<ov::PartialShape> inputShapes6D = {{-1, -1, -1, -1, -1, -1}};

const std::vector<TransposeTransformationTestValues> testValues = {
    {{0, 1, 2, 3, 4, 5},
     LayerTransformation::createParamsU8I8(),
     {ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}},
     {ov::element::u8, {{}, {}, {}}, ov::element::u8, {{ov::element::f32}, {128}, {0.1f}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         TransposeTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes6D), ::testing::ValuesIn(testValues)),
                         TransposeTransformation::getTestCaseName);
}  // namespace testValues4
}  // namespace
