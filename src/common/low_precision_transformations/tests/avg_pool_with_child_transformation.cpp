// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/avg_pool.hpp"
#include "low_precision/convolution.hpp"
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

class AvgPoolWithChildTransformationTestValues {
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
        ov::builder::subgraph::DequantizationOperations dequantizationEnd;
    };

    TestTransformationParams params;
    std::vector<std::string> additionalOperations;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ov::element::Type, ov::PartialShape, AvgPoolWithChildTransformationTestValues>
    AvgPoolWithChildTransformationParams;

class AvgPoolWithChildTransformation : public LayerTransformation,
                                       public testing::WithParamInterface<AvgPoolWithChildTransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision;
        ov::PartialShape shape;
        std::string additionalLayer;
        AvgPoolWithChildTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = GetParam();
        actualFunction = ov::builder::subgraph::AvgPoolFunction::getOriginal(precision,
                                                                                 testValues.actual.inputPrecision,
                                                                                 shape,
                                                                                 false,
                                                                                 testValues.additionalOperations,
                                                                                 testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::AvgPoolTransformation, ov::op::v1::AvgPool>(testValues.params);
        transform.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(
            testValues.params);
        transform.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::AvgPoolFunction::getReference(precision,
                                                                     testValues.expected.inputPrecision,
                                                                     shape,
                                                                     false,
                                                                     testValues.additionalOperations,
                                                                     testValues.expected.dequantizationBefore,
                                                                     testValues.expected.preicsionAfterOperation,
                                                                     testValues.expected.dequantizationAfter,
                                                                     testValues.expected.dequantizationEnd);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolWithChildTransformationParams> obj) {
        ov::element::Type precision;
        ov::PartialShape shape;
        std::string additionalLayer;
        AvgPoolWithChildTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_"
               << LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision,
                                                               shape,
                                                               testValues.params)
               << "_" << testValues.actual.dequantization << "_" << testValues.expected.dequantizationBefore << "_"
               << testValues.expected.preicsionAfterOperation << "_" << testValues.expected.dequantizationAfter
               << "_additional_operations_";
        for (const auto& elem : testValues.additionalOperations) {
            result << elem << "_";
        }

        return result.str();
    }
};

TEST_P(AvgPoolWithChildTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {ov::element::f32};

const std::vector<ov::PartialShape> shapes = {{1, 3, 72, 48}, {4, 3, 72, 48}};

const std::vector<AvgPoolWithChildTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {"convolution"},
     {ov::element::u8, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::u8, {}, ov::element::u8, {}, {{}, {}, {std::vector<float>{0.0002f}, element::f32, {}}}}},
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {"softmax", "convolution"},
     {ov::element::u8, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::u8, {}, ov::element::f32, {{}, {}, {0.02f}}, {}}},
    {LayerTransformation::createParamsU8I8(),
     {"unsupported_convolution"},
     {ov::element::u8, {{ov::element::f32}, {}, {0.02f}}},
     {ov::element::u8, {}, ov::element::f32, {{}, {}, {0.02f}}, {}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         AvgPoolWithChildTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(testValues)),
                         AvgPoolWithChildTransformation::getTestCaseName);
