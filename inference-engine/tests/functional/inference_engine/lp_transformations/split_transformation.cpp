// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/split.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/split_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ngraph::pass;

class SplitTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precision;
        std::vector<ngraph::builder::subgraph::DequantizationOperations> dequantizationAfter;
    };

    ngraph::Shape inputShape;
    std::int64_t splitedAxis;
    size_t numSplits;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator<<(std::ostream& os,
    const std::vector<ngraph::builder::subgraph::DequantizationOperations>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class SplitTransformation : public LayerTransformation, public testing::WithParamInterface<SplitTransformationTestValues> {
public:
    void SetUp() override {
        const SplitTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::SplitFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.splitedAxis,
            testValues.numSplits);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::SplitTransformation, ngraph::opset1::Split>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::SplitFunction::getReference(
            testValues.inputShape,
            testValues.expected.precision,
            testValues.expected.dequantizationAfter,
            testValues.splitedAxis,
            testValues.numSplits);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SplitTransformationTestValues> obj) {
        const SplitTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationAfter <<
            "_axis=" << testValues.splitedAxis <<
            "_num_splits=" << testValues.numSplits;
        return result.str();
    }
};

TEST_P(SplitTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<SplitTransformationTestValues> testValues = {
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {128.f}, {3.f}},
                {{ngraph::element::f32}, {128.f}, {3.f}},
            }
        }
    },
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{1}, size_t{3},
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            {
                {
                    {ngraph::element::f32},
                    {{1.f}, ngraph::element::f32, {1, 1, 1, 1}},
                    {{11.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{2.f}, ngraph::element::f32, {1, 1, 1, 1}},
                    {{22.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{3.f}, ngraph::element::f32, {1, 1, 1, 1}},
                    {{33.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
            }
        }
    },
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-1}, size_t{2},
        LayerTransformation::createParamsU8I8(),
        // Actualvalues
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        // Expectedvalues
        {
            ngraph::element::u8,
            {
                {
                    {ngraph::element::f32},
                    {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
                    {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
                    {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
                    {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}
                }
            }
        }
    },
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-3}, size_t{3},
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            {
                {
                    {ngraph::element::f32},
                    {},
                    {{11.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {},
                    {{22.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {},
                    {{33.f}, ngraph::element::f32, {1, 1, 1, 1}}
                },
            }
        }
    },
    {
        ngraph::Shape({ 1, 3, 4, 4 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {1, 1, 4, 1}},
            {{11.f, 22.f, 33.f, 44.f}, ngraph::element::f32, {1, 1, 4, 1}}}
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            {
                {
                    {ngraph::element::f32},
                    {{1.f, 2.f}, ngraph::element::f32, {1, 1, 2, 1}},
                    {{11.f, 22.f}, ngraph::element::f32, {1, 1, 2, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{3.f, 4.f}, ngraph::element::f32, {1, 1, 2, 1}},
                    {{33.f, 44.f}, ngraph::element::f32, {1, 1, 2, 1}}
                }
            }
        }
    },
    // no Convert
    {
        ngraph::Shape({ 1, 3, 4, 4 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::f32,
            {{},
            {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {1, 1, 4, 1}},
            {{11.f, 22.f, 33.f, 44.f}, ngraph::element::f32, {1, 1, 4, 1}}}
        },
        // ExpectedValues
        {
            ngraph::element::f32,
            {
                {
                    {},
                    {{1.f, 2.f}, ngraph::element::f32, {1, 1, 2, 1}},
                    {{11.f, 22.f}, ngraph::element::f32, {1, 1, 2, 1}}
                },
                {
                    {},
                    {{3.f, 4.f}, ngraph::element::f32, {1, 1, 2, 1}},
                    {{33.f, 44.f}, ngraph::element::f32, {1, 1, 2, 1}}
                }
            }
        }
    },
    // empty
    {
        ngraph::Shape({ 1, 3, 4, 4 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        { },
        // ExpectedValues
        { }
    },
};
INSTANTIATE_TEST_CASE_P(
    LPT,
    SplitTransformation,
    ::testing::ValuesIn(testValues),
    SplitTransformation::getTestCaseName);
} // namespace
