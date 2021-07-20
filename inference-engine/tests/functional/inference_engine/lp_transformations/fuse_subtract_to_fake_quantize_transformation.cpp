// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/fuse_subtract_to_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class FuseSubtractToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        DequantizationOperations dequantization;
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData2;
        DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        DequantizationOperations dequantization;
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData2;
        DequantizationOperations dequantization2;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    size_t,
    ngraph::PartialShape,
    FuseSubtractToFakeQuantizeTransformationTestValues> FuseSubtractToFakeQuantizeTransformationTestParams;

class FuseSubtractToFakeQuantizeTransformation : public LayerTransformation,
    public testing::WithParamInterface<FuseSubtractToFakeQuantizeTransformationTestParams> {
public:
    void SetUp() override {
        const size_t quantizationLevel = std::get<0>(GetParam());
        const ngraph::PartialShape inputShape = std::get<1>(GetParam());
        FuseSubtractToFakeQuantizeTransformationTestValues testValues = std::get<2>(GetParam());

        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.actual.fakeQuantizeOnData2.empty()) {
            testValues.actual.fakeQuantizeOnData2.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData2.empty()) {
            testValues.expected.fakeQuantizeOnData2.quantizationLevel = quantizationLevel;
        }

        actualFunction = testValues.actual.fakeQuantizeOnData2.empty() ?
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                inputShape,
                testValues.actual.fakeQuantizeOnData,
                testValues.actual.dequantization) :
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                inputShape,
                testValues.actual.fakeQuantizeOnData,
                testValues.actual.dequantization,
                testValues.actual.fakeQuantizeOnData2,
                testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation, ngraph::opset1::Subtract>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = testValues.expected.fakeQuantizeOnData2.empty() ?
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                inputShape,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.dequantization) :
            ngraph::builder::subgraph::FuseSubtractToFakeQuantizeFunction::get(
                inputShape,
                testValues.expected.fakeQuantizeOnData,
                testValues.expected.dequantization,
                testValues.expected.fakeQuantizeOnData2,
                testValues.expected.dequantization2);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseSubtractToFakeQuantizeTransformationTestParams> obj) {
        const size_t quantizationLevel = std::get<0>(obj.param);
        const ngraph::PartialShape inputShape = std::get<1>(obj.param);
        FuseSubtractToFakeQuantizeTransformationTestValues testValues = std::get<2>(obj.param);

        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.actual.fakeQuantizeOnData2.empty()) {
            testValues.actual.fakeQuantizeOnData2.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData2.empty()) {
            testValues.expected.fakeQuantizeOnData2.quantizationLevel = quantizationLevel;
        }

        std::ostringstream result;
        result << inputShape << "_" <<
            testValues.params.updatePrecisions << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData2 << "_" <<
            testValues.expected.dequantization2;
        return result.str();
    }
};

TEST_P(FuseSubtractToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<size_t> quantizationLevels = { 256ul, 128ul };

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    {1, 4, 16, 16},
    {Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic()}
};

const std::vector<FuseSubtractToFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::i8 },
            { {}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, { 128.f }, {} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {{128.f, 64.f, 32.f, 16.f}}, {} },
            {},
            {}
        },
        {
            { 256ul,
                {{}, {}, {1, 4, 1, 1}, {1, 4, 1, 1}},
                { 0.f }, { 2.55f },
                { -128.f, -64.f, -32.f, -16.f }, { 127.f, 191.f, 223.f, 239.f }
            },
            { {}, {}, {} },
            {},
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseSubtractToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    FuseSubtractToFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicChannels = {
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
};

const std::vector<FuseSubtractToFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
            { {}, {}, {} },
            {},
            {}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {{128.f, 64.f, 32.f, 16.f}}, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {{128.f, 64.f, 32.f, 16.f}}, {} },
            {},
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseSubtractToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(inputShapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    FuseSubtractToFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicRank = {
    PartialShape::dynamic()
};

const std::vector<FuseSubtractToFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, { 128.f }, {} },
            {},
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseSubtractToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(testValues)),
    FuseSubtractToFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues3
} // namespace
