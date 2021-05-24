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

class AvgPoolWithHandledChildTransformationTestValues {
public:
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

    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::vector<std::string> additionalOperations;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    AvgPoolWithHandledChildTransformationTestValues> AvgPoolWithHandledChildTransformationParams;

class AvgPoolWithHandledChildTransformation : public LayerTransformation, public testing::WithParamInterface<AvgPoolWithHandledChildTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolWithHandledChildTransformationTestValues testValues;
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

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolWithHandledChildTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolWithHandledChildTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, shape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter;
        return result.str();
    }
};

TEST_P(AvgPoolWithHandledChildTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 72, 48 }
};

const std::vector<AvgPoolWithHandledChildTransformationTestValues> testValues = {
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
            {{}, {}, {std::vector<float>{0.0002f}, element::f32, {1,6,1,1}}}
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
    }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    AvgPoolWithHandledChildTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    AvgPoolWithHandledChildTransformation::getTestCaseName);
