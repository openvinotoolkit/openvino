// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/avg_pool.hpp>
#include <low_precision/convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/avg_pool_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;

class AvgPoolWithChildTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        ngraph::builder::subgraph::DequantizationOperations dequantizationEnd;
    };

    TestTransformationParams params;
    std::vector<std::string> additionalOperations;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    AvgPoolWithChildTransformationTestValues> AvgPoolWithChildTransformationParams;

class AvgPoolWithChildTransformation : public LayerTransformation, public testing::WithParamInterface<AvgPoolWithChildTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::PartialShape shape;
        std::string additionalLayer;
        AvgPoolWithChildTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = GetParam();
        actualFunction = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
            precision,
            testValues.actual.inputPrecision,
            shape,
            false,
            testValues.additionalOperations,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::AvgPoolFunction::getReference(
            precision,
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
        ngraph::element::Type precision;
        ngraph::PartialShape shape;
        std::string additionalLayer;
        AvgPoolWithChildTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, shape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_additional_operations_";
        for (const auto& elem : testValues.additionalOperations) {
            result << elem << "_";
        }

        return result.str();
    }
};

TEST_P(AvgPoolWithChildTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32
};

const std::vector<ngraph::PartialShape> shapes = {
    { 1, 3, 72, 48 },
    { 4, 3, 72, 48 }
};

const std::vector<AvgPoolWithChildTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        { "convolution" },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {},
            {{}, {}, {std::vector<float>{0.0002f}, element::f32, {1, 6, 1, 1}}}
        }
    },
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        { "softmax", "convolution" },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {0.02f}},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        { "unsupported_convolution" },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {0.02f}},
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    AvgPoolWithChildTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    AvgPoolWithChildTransformation::getTestCaseName);
