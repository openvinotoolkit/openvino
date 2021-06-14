// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <transformations/init_node_info.hpp>
#include <low_precision/split.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/split_function.hpp"
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
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        std::vector<ngraph::builder::subgraph::DequantizationOperations> dequantizationAfter;
    };

    ngraph::Shape inputShape;
    std::int64_t splitedAxis;
    size_t numSplits;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
    bool addUnsupportedConcat;
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

typedef std::tuple <
    ngraph::element::Type,
    SplitTransformationTestValues
> SplitTransformationParams;

class SplitTransformation : public LayerTransformation, public testing::WithParamInterface<SplitTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision = std::get<0>(GetParam());
        SplitTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::SplitFunction::getOriginal(
            precision,
            testValues.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.splitedAxis,
            testValues.numSplits,
            testValues.addUnsupportedConcat);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::SplitTransformation, ngraph::opset1::Split>(testValues.params.setSupportAsymmetricQuantization(true));
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::SplitFunction::getReference(
            precision,
            testValues.inputShape,
            testValues.expected.inputPrecision,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            testValues.splitedAxis,
            testValues.numSplits,
            testValues.addUnsupportedConcat);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SplitTransformationParams> obj) {
        ngraph::element::Type precision = std::get<0>(obj.param);
        SplitTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << precision << "_" <<
            toString(testValues.params) << "_" <<
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

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<SplitTransformationTestValues> testValues = {
    // U8 per tensor quantization
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
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {128.f}, {3.f}},
                {{ngraph::element::f32}, {128.f}, {3.f}},
            }
        }
    },
    // I8 per tensor quantization
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {128.f}, {3.f}},
                {{ngraph::element::f32}, {128.f}, {3.f}},
            }
        }
    },
    // U8 per channel quantization with different values
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{1}, size_t{3},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {2.f}, {22.f}},
                {{ngraph::element::f32}, {3.f}, {33.f}},
            }
        }
    },
    // U8 per channel quantization with different values (constants without batch)
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-3}, size_t{3},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {2.f}, {22.f}},
                {{ngraph::element::f32}, {3.f}, {33.f}},
            }
        }
    },
    // I8 per channel quantization with different values
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{1}, size_t{3},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {2.f}, {22.f}},
                {{ngraph::element::f32}, {3.f}, {33.f}},
            }
        }
    },
    // U8 per channel quantization with the same values
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{1}, size_t{3},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {{1.f, 1.f, 1.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 11.f, 11.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {1.f}, {11.f}},
            }
        }
    },
    // I8 per channel quantization with the same values
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{1}, size_t{3},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 1.f, 1.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 11.f, 11.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {1.f}, {11.f}},
                {{ngraph::element::f32}, {1.f}, {11.f}}
            }
        }
    },
    // U8 split second dimension
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-1}, size_t{2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
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
    // I8 split second dimension
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-1}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f}, ngraph::element::f32, {1, 3, 1, 1}},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
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
    // U8 without subtract
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-3}, size_t{3},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32},
            {},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {{ngraph::element::f32}, {}, {11.f}},
                {{ngraph::element::f32}, {}, {22.f}},
                {{ngraph::element::f32}, {}, {33.f}},
            }
        }
    },
    // I8 without subtract
    {
        ngraph::Shape({ 1, 3, 16, 16 }), std::int64_t{-3}, size_t{3},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {},
            {{11.f, 22.f, 33.f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {{ngraph::element::f32}, {}, {11.f}},
                {{ngraph::element::f32}, {}, {22.f}},
                {{ngraph::element::f32}, {}, {33.f}},
            }
        }
    },
    // I8 dequantization in second dimension
    {
        ngraph::Shape({ 1, 4, 3, 3 }), std::int64_t{1}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32},
            {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {1, 4, 1, 1}},
            {{11.f, 22.f, 33.f, 44.f}, ngraph::element::f32, {1, 4, 1, 1}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {
                    {ngraph::element::f32},
                    {{1.f, 2.f}, ngraph::element::f32, {1, 2, 1, 1}},
                    {{11.f, 22.f}, ngraph::element::f32, {1, 2, 1, 1}}
                },
                {
                    {ngraph::element::f32},
                    {{3.f, 4.f}, ngraph::element::f32, {1, 2, 1, 1}},
                    {{33.f, 44.f}, ngraph::element::f32, {1, 2, 1, 1}}
                }
            }
        }
    },
    // without Convert
    {
        ngraph::Shape({ 1, 4, 3, 3 }), std::int64_t{1}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::f32,
            {{},
            {{1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {1, 4, 1, 1}},
            {{11.f, 22.f, 33.f, 44.f}, ngraph::element::f32, {1, 4, 1, 1}}}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {
                {
                    {},
                    {{1.f, 2.f}, ngraph::element::f32, {1, 2, 1, 1}},
                    {{11.f, 22.f}, ngraph::element::f32, {1, 2, 1, 1}}
                },
                {
                    {},
                    {{3.f, 4.f}, ngraph::element::f32, {1, 2, 1, 1}},
                    {{33.f, 44.f}, ngraph::element::f32, {1, 2, 1, 1}}
                }
            }
        }
    },
    // issue #56781: unsupported Concat after Split
    {
        ngraph::Shape({ 1, 4, 3, 3 }), std::int64_t{2}, size_t{3},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}},
            ngraph::element::f32,
            {}
        },
        true
    },
    // no dequantization
    {
        ngraph::Shape({ 1, 3, 4, 4 }), std::int64_t{2}, size_t{2},
        LayerTransformation::createParamsI8I8(),
        { },
        { }
    },
};
INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    SplitTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(testValues)),
    SplitTransformation::getTestCaseName);
} // namespace
