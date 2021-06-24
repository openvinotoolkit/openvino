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
#include <low_precision/max_pool.hpp>
#include <low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/avg_pool_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;

class AvgPoolTransformationTestValues {
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
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool, // additional FakeQuantize After
    std::string, // additional layer before FQ
    AvgPoolTransformationTestValues> AvgPoolTransformationParams;

class AvgPoolTransformation : public LayerTransformation, public testing::WithParamInterface<AvgPoolTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();
        actualFunction = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
            precision,
            testValues.actual.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::AvgPoolFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        AvgPoolTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, shape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(AvgPoolTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<std::string> additionalLayer = {
    "",
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    true,
    false
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 72, 48 }
};

const std::vector<AvgPoolTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
    // U8 without subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::f32,
            {{}, {}, {0.02f}}
        }
    },
    // U8 per channel quantization with different values
    {
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
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            },
        }
    },
    // U8 per channel quantization with the same values
    {
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
    // U8 without dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {}
        }
    },
    // U8 not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
    // I8 per tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
    // I8 without subtract
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {0.02f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {}, {0.02f}}
        }
    },
    // I8 per channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{64.f, 0.f, 32.f}},
                {{3.f, 1.f, 2.f}}
            }
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{64.f, 0.f, 32.f}},
                {{3.f, 1.f, 2.f}}
            },
        }
    },
    // I8 per channel quantization with the same values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{64.f, 64.f, 64.f}},
                {{3.f, 3.f, 3.f}}
            }
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::f32,
            {
                {},
                {{64.f, 64.f, 64.f}},
                {{3.f, 3.f, 3.f}}
            },
        }
    },
    // I8 without dequantization
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {}
        }
    },
    // I8 not update precisions
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    AvgPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(addFQ),
        ::testing::ValuesIn(additionalLayer),
        ::testing::ValuesIn(testValues)),
    AvgPoolTransformation::getTestCaseName);
