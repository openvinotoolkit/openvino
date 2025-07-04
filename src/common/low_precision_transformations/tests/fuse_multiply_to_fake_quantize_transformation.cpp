// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/fuse_multiply_to_fake_quantize.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class FuseMultiplyToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        DequantizationOperations dequantization;
    };

    class Expected {
    public:
        FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
        DequantizationOperations dequantization;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    size_t,
    ov::PartialShape,
    FuseMultiplyToFakeQuantizeTransformationTestValues> FuseMultiplyToFakeQuantizeTransformationTestParams;

class FuseMultiplyToFakeQuantizeTransformation : public LayerTransformation,
    public testing::WithParamInterface<FuseMultiplyToFakeQuantizeTransformationTestParams> {
public:
    void SetUp() override {
        const size_t quantizationLevel = std::get<0>(GetParam());
        const ov::PartialShape inputShape = std::get<1>(GetParam());
        FuseMultiplyToFakeQuantizeTransformationTestValues testValues = std::get<2>(GetParam());

        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }

        actualFunction = FuseMultiplyToFakeQuantizeFunction::get(
            inputShape,
            testValues.actual.fakeQuantizeOnData,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation, ov::op::v1::Multiply>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = FuseMultiplyToFakeQuantizeFunction::get(
            inputShape,
            testValues.expected.fakeQuantizeOnData,
            testValues.expected.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseMultiplyToFakeQuantizeTransformationTestParams> obj) {
        const size_t quantizationLevel = std::get<0>(obj.param);
        const ov::PartialShape inputShape = std::get<1>(obj.param);
        FuseMultiplyToFakeQuantizeTransformationTestValues testValues = std::get<2>(obj.param);

        if (!testValues.actual.fakeQuantizeOnData.empty()) {
            testValues.actual.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }
        if (!testValues.expected.fakeQuantizeOnData.empty()) {
            testValues.expected.fakeQuantizeOnData.quantizationLevel = quantizationLevel;
        }

        std::ostringstream result;
        result << inputShape << "_" <<
            testValues.params.deqPrecision << "_" <<
            testValues.params.updatePrecisions << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization;
        return result.str();
    }
};

TEST_P(FuseMultiplyToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

std::vector<size_t> quantizationLevels = { 256ul, 128ul };

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    {1, 3, 16, 16},
    {Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()}
};

const std::vector<FuseMultiplyToFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f }, element::i8 },
            { {element::f32}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { -1.28f }, { 1.27f }, { -0.64f }, { 0.635f } },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, {}, {{0.5f, 0.4f, 0.3f}} },
        },
        {
            { 256ul,
                {{}, {}, { 1, 3, 1, 1 }, { 1, 3, 1, 1 }},
                { 0.f }, { 2.55f },
                { 0.f }, { 127.5f, 102.f, 76.5f }
            },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8().setDeqPrecision(ov::element::f16),
        {
            FakeQuantizeOnDataWithConstant(256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8).setConstantPrecision(ov::element::f16),
            { {}, {}, { 0.5f } },
        },
        {
            FakeQuantizeOnDataWithConstant(256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f }).setConstantPrecision(ov::element::f16),
            { {}, {}, {} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseMultiplyToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    FuseMultiplyToFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapesWithDynamicChannels = {
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
    ov::PartialShape::dynamic()
};

const std::vector<FuseMultiplyToFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {}, {}, { 0.5f } },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 127.5f } },
            { {}, {}, {} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {}, {{0.5f, 0.4f, 0.3f}} },
        },
        {
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, element::u8 },
            { {element::f32}, {}, {{0.5f, 0.4f, 0.3f}} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseMultiplyToFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(quantizationLevels),
        ::testing::ValuesIn(inputShapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    FuseMultiplyToFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues2
} // namespace
