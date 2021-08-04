// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <low_precision/shuffle_channels.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/shuffle_channels_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ShuffleChannelsTransformationTestValues {
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

    TestTransformationParams params;
    std::int64_t axis;
    std::int64_t group;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    ShuffleChannelsTransformationTestValues> ShuffleChannelsTransformationParams;

class ShuffleChannelsTransformation : public LayerTransformation, public testing::WithParamInterface<ShuffleChannelsTransformationParams> {
public:
    void SetUp() override {
        ngraph::PartialShape inputShape = std::get<0>(GetParam());
        ShuffleChannelsTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::ShuffleChannelsFunction::getOriginal(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.axis,
            testValues.group);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ShuffleChannelsTransformation, ngraph::opset1::ShuffleChannels>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ShuffleChannelsFunction::getReference(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.axis,
            testValues.group,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsTransformationParams> obj) {
        ngraph::PartialShape inputShape = std::get<0>(obj.param);
        ShuffleChannelsTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, inputShape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_axis_" <<
            testValues.axis << "_group_" << testValues.group;

        return result.str();
    }
};

TEST_P(ShuffleChannelsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    { 1, 3, 8, 10 },
    { 4, 3, 8, 10 },
    { Dimension::dynamic(), 3, 8, Dimension::dynamic() }
};

const std::vector<ShuffleChannelsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        1, // axis
        1, // group
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        }
    },
    // U8 per channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        1,
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // U8 quantization by spatial dimension, shuffling by the same dimension
    {
        LayerTransformation::createParamsU8I8(),
        2,
        4,
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}}
            }
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{121.f, 123.f, 125.f, 127.f, 122.f, 124.f, 126.f, 128.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}},
                {{1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f}, ngraph::element::f32, ngraph::Shape {1, 1, 8, 1}},
            }
        }
    },
    // U8 per channel quantization, shuffling by spatial dimension
    {
        LayerTransformation::createParamsU8I8(),
        -2,
        4,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // I8 per tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        1,
        1,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        }
    },
    // I8 per channel quantization
    {
        LayerTransformation::createParamsI8I8(),
        1,
        1,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // I8 quantization by spatial dimension, shuffling by the same dimension
    {
        LayerTransformation::createParamsI8I8(),
        2,
        4,
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{121.f, 123.f, 125.f, 127.f, 122.f, 124.f, 126.f, 128.f}, ngraph::element::f32, ngraph::Shape{1, 1, 8, 1}},
                {{1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f}, ngraph::element::f32, ngraph::Shape {1, 1, 8, 1}},
            }
        }
    },
    // I8 per channel quantization, shuffling by spatial dimension
    {
        LayerTransformation::createParamsI8I8(),
        -2,
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // U8 per tensor quantization, not update precision
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3,
        5,
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
    // U8 without dequantization operations
    {
        LayerTransformation::createParamsU8I8(),
        2,
        4,
        {
            ngraph::element::u8,
            {{}, {}, {}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ShuffleChannelsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    ShuffleChannelsTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicChannels = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
};

const std::vector<ShuffleChannelsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        1, // axis
        1, // group
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        }
    },
    // U8 per channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        1,
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ShuffleChannelsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    ShuffleChannelsTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicRank = {
    ngraph::PartialShape::dynamic()
};

const std::vector<ShuffleChannelsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        1, // axis
        1, // group
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.02f}},
            ngraph::element::f32,
            {},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ShuffleChannelsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(testValues)),
    ShuffleChannelsTransformation::getTestCaseName);
} // namespace testValues3
} // namespace
