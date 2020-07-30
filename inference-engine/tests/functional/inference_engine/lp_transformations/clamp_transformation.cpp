// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/clamp.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/clamp_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ngraph::pass;

class ClampTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class ClampTransformation : public LayerTransformation, public testing::WithParamInterface<ClampTransformationTestValues> {
public:
    void SetUp() override {
        const ClampTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::ClampFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ClampTransformation, ngraph::opset1::Clamp>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ClampFunction::getReference(
            testValues.inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ClampTransformationTestValues> obj) {
        const ClampTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }
};

TEST_P(ClampTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ClampTransformationTestValues> testValues = {
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {128.f}, {3.f}}
        }
    },
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {-5.f}}
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {128.f}, {-5.f}}
        }
    },
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {3.f}}
        }
    },
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f}},
                {{3.f, 1.f, 2.f}}
            }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f}},
                {{3.f, 1.f, 2.f}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {},
                {{3.f, 1.f, 2.f}}
            }
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {},
                {{3.f, 1.f, 2.f}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
};
INSTANTIATE_TEST_CASE_P(
    LPT,
    ClampTransformation,
    ::testing::ValuesIn(testValues),
    ClampTransformation::getTestCaseName);
} // namespace
