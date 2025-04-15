// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "low_precision/shuffle_channels.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/shuffle_channels.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

class ShuffleChannelsTransformationTestValues {
public:
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
    };

    TestTransformationParams params;
    std::int64_t axis;
    std::int64_t group;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::PartialShape,
    ShuffleChannelsTransformationTestValues> ShuffleChannelsTransformationParams;

class ShuffleChannelsTransformation : public LayerTransformation, public testing::WithParamInterface<ShuffleChannelsTransformationParams> {
public:
    void SetUp() override {
        ov::PartialShape inputShape = std::get<0>(GetParam());
        ShuffleChannelsTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::ShuffleChannelsFunction::getOriginal(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.axis,
            testValues.group);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ShuffleChannelsTransformation, ov::op::v0::ShuffleChannels>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ShuffleChannelsFunction::getReference(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.axis,
            testValues.group,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsTransformationParams> obj) {
        ov::PartialShape inputShape = std::get<0>(obj.param);
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 3, 8, 10 },
    { 4, 3, 8, 10 },
    { -1, -1, -1, -1 }
};

const std::vector<ShuffleChannelsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        1, // axis
        1, // group
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}}
        }
    },
    // U8 per channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        1,
        1,
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // subtraction with Convert from u8 to fp32
    {
        LayerTransformation::createParamsU8I8(),
        1,
        1,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
                {3.f}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
                {3.f}
            }
        }
    },
    // U8 quantization by spatial dimension, shuffling by the same dimension
    {
        LayerTransformation::createParamsU8I8(),
        2,
        4,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{121.f, 123.f, 125.f, 127.f, 122.f, 124.f, 126.f, 128.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}},
                {{1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f}, ov::element::f32, ov::Shape {1, 1, 8, 1}},
            }
        }
    },
    // U8 per channel quantization, shuffling by spatial dimension
    {
        LayerTransformation::createParamsU8I8(),
        -2,
        4,
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // I8 per tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        1,
        1,
        {
            ov::element::i8,
            {{ov::element::f32}, {128.f}, {0.02f}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {128.f}, {0.02f}}
        }
    },
    // I8 per channel quantization
    {
        LayerTransformation::createParamsI8I8(),
        1,
        1,
        {
            ov::element::i8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // I8 quantization by spatial dimension, shuffling by the same dimension
    {
        LayerTransformation::createParamsI8I8(),
        2,
        4,
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{121.f, 123.f, 125.f, 127.f, 122.f, 124.f, 126.f, 128.f}, ov::element::f32, ov::Shape{1, 1, 8, 1}},
                {{1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f}, ov::element::f32, ov::Shape {1, 1, 8, 1}},
            }
        }
    },
    // I8 per channel quantization, shuffling by spatial dimension
    {
        LayerTransformation::createParamsI8I8(),
        -2,
        4,
        {
            ov::element::i8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{0.01f, 0.02f, 0.03f}}}
        }
    },
    // U8 per tensor quantization, not update precision
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3,
        5,
        {
            ov::element::f32,
            {{}, {128.f}, {0.02f}}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {{}, {128.f}, {0.02f}}
        }
    },
    // U8 without dequantization operations
    {
        LayerTransformation::createParamsU8I8(),
        2,
        4,
        {
            ov::element::u8,
            {{}, {}, {}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
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
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {
    ov::PartialShape::dynamic()
};

const std::vector<ShuffleChannelsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        1, // axis
        1, // group
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            ov::element::f32,
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
} // namespace testValues2
} // namespace
