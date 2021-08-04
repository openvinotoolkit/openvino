// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/max_pool_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"


using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class MaxPoolTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    MaxPoolTransformationTestValues> MaxPoolTransformationParams;

class MaxPoolTransformation : public LayerTransformation, public testing::WithParamInterface<MaxPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape shape = std::get<0>(GetParam());
        const MaxPoolTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::MaxPoolFunction::get(
            shape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization1,
            testValues.actual.preicsionAfterOperation,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MaxPoolFunction::get(
            shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantization1,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolTransformationParams> obj) {
        const ngraph::PartialShape shape = std::get<0>(obj.param);
        const MaxPoolTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << testValues.actual.precisionBeforeDequantization << "_"<<
            shape << "_" << toString(testValues.params) << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.dequantization2 << "_" <<
            testValues.expected.dequantization1 << "_" <<
            testValues.expected.dequantization2 << "_";
        return result.str();
    }
};

TEST_P(MaxPoolTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> shapes = {
    { 1, 3, 72, 48 },
    { 4, 3, 72, 48 },
    { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() },
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    // Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {}, {}, { {0.02f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 }},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { {0.02f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 }}
        }
    },
    // Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {},
                { {128.f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 },
                { {0.02f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 }
            },
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                ngraph::element::f32,
                { {128.f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 },
                { {0.02f}, ngraph::element::f32, {}, true, 1, ngraph::element::f32 }
            }
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, { 128 }, { 0.02f }},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { ngraph::element::f32, { 128 }, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f }},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            { ngraph::element::f32, { 128 }, { 0.02f }},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            { {}, { 128 }, { 0.02f }}
        }
    },
    // Convert + Subtract + Multiply
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            { ngraph::element::f32, {}, { 0.02f }},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            { {}, {}, { 0.02f }}
        }
    },
    // per-channel dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {{128.f, 64.f, 32.f}}, {{0.02f, 0.01f, 0.03f}}},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, { {0.02f, 0.01f, 0.03f} }}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> shapesWithDynamicChannels = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
    PartialShape::dynamic()
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    // per-tensor dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {128.f}, {0.01f}},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            { ngraph::element::f32, {128.f}, {0.01f}}
        }
    },
    // per-channel dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::f32, {{128.f, 64.f, 32.f}}, {{0.02f, 0.01f, 0.03f}}},
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, { {0.02f, 0.01f, 0.03f} }},
            ngraph::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
} // namespace testValues2
