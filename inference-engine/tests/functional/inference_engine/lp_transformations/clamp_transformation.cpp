// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/clamp.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/clamp_function.hpp"
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
    bool nonDequantizationMultiply;
};

class ClampTransformation : public LayerTransformation, public testing::WithParamInterface<ClampTransformationTestValues> {
public:
    void SetUp() override {
        const ClampTransformationTestValues testValues = GetParam();

        actualFunction = testValues.nonDequantizationMultiply ?
            ngraph::builder::subgraph::ClampFunction::getWithNonDequantizationMultiply(
                testValues.inputShape,
                testValues.actual.precisionBeforeDequantization) :
            ngraph::builder::subgraph::ClampFunction::getOriginal(
                testValues.inputShape,
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ClampTransformation, ngraph::opset1::Clamp>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = testValues.nonDequantizationMultiply ?
            ngraph::builder::subgraph::ClampFunction::getWithNonDequantizationMultiply(
                testValues.inputShape,
                testValues.actual.precisionBeforeDequantization) :
            ngraph::builder::subgraph::ClampFunction::getReference(
                testValues.inputShape,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationBefore,
                testValues.expected.precisionAfterOperation,
                testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ClampTransformationTestValues> obj) {
        const ClampTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << toString(testValues.params) << "_" <<
            testValues.inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore <<
            (testValues.nonDequantizationMultiply ? "non_deq_mul" : "");
        return result.str();
    }
};

TEST_P(ClampTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ClampTransformationTestValues> testValues = {
    // U8 per tensor quantization
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
    // U8 per tensor quantization
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true},
                {3.f}
            }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{128.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true},
                {3.f}
            }
        }
    },
    // I8 per tensor quantization
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {-5.f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {128.f}, {-5.f}}
        }
    },
    // U8 without convert
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {128.f}, {3.f}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {128.f}, {3.f}}
        }
    },
    // I8 without convert
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::f32,
            {{}, {128.f}, {3.f}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {128.f}, {3.f}}
        }
},
    // U8 without subtract
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {3.f}}
        }
    },
    // I8 without subtract
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {3.f}}
        }
    },
    // U8 per channel quantization with different values
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // I8 per channel quantization with different values
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            }
        },
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // U8 per channel quantization with the same values
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            },
        }
    },
    // I8 per channel quantization with the same values
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            }
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            },
        }
    },
    // U8 dequantization in second dimension
    {
        ngraph::Shape({ 1, 3, 4, 4 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f, 128.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 3.f, 3.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f, 128.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 3.f, 3.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // I8 dequantization in second dimension
    {
        ngraph::Shape({ 1, 3, 4, 4 }),
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f, 128.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 3.f, 3.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            }
        },
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{128.f, 128.f, 128.f, 128.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 3.f, 3.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // U8 asymmetric quantization
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }},
                {{ 3.f, 3.f, 3.f }}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }},
                {{ 3.f, 3.f, 3.f }}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // U8 without asymmetric quantization
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }},
                {{ 3.f, 3.f, 3.f }}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{ 128.f, 0.f, 128.f }},
                {{ 3.f, 3.f, 3.f }}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // per channel quantization with small values
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                    {ngraph::element::f32},
                    {{1e-14, 1e-12, 1e-15}},
                    {{1e-14, 1e-12, 1e-15}}
            }
        },
        {
            ngraph::element::u8,
            {
                    {ngraph::element::f32},
                    {{1e-14, 1e-12, 1e-15}},
                    {{1e-14, 1e-12, 1e-15}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // without dequantization
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {}}
        },
    },
    // with non dequantization multiply (issue #49965)
    {
        ngraph::Shape({ 1, 3, 224, 224 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {}}
        },
        true // non dequantization multiply
    },
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ClampTransformation,
    ::testing::ValuesIn(testValues),
    ClampTransformation::getTestCaseName);
} // namespace
