// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <low_precision/avg_pool.hpp>
#include <low_precision/common/precisions_restriction.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/low_precision.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include "lpt_ngraph_functions/fake_quantize_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeTransformationTestValues {
public:
    FakeQuantizeTransformationTestValues() = default;

    FakeQuantizeTransformationTestValues(
        const TestTransformationParams& params,
        const builder::subgraph::FakeQuantizeOnDataWithConstant& actual,
        const builder::subgraph::FakeQuantizeOnDataWithConstant& expected,
        const ngraph::element::Type expectedFakeQuantizeOnDataPrecision,
        const std::map<ngraph::element::Type, ngraph::builder::subgraph::DequantizationOperations>& expectedValues,
        const bool addNotPrecisionPreservedOperation = false) :
            params(params),
            actual(actual),
            expected(expected),
            expectedFakeQuantizeOnDataPrecision(expectedFakeQuantizeOnDataPrecision),
            expectedValues(expectedValues),
            addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation) {}

    TestTransformationParams params;
    builder::subgraph::FakeQuantizeOnDataWithConstant actual;
    builder::subgraph::FakeQuantizeOnDataWithConstant expected;
    ngraph::element::Type expectedFakeQuantizeOnDataPrecision;
    std::map<ngraph::element::Type, ngraph::builder::subgraph::DequantizationOperations> expectedValues;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeTransformationTestValues& testValue) {
    return out << "_" << testValue.actual << "_" << testValue.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    bool,
    FakeQuantizeTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::PartialShape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizeTransformationTestValues fakeQuantizeOnData = std::get<3>(GetParam());
        std::vector<element::Type> defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_support;

        if (fakeQuantizeOnData.actual.quantizationLevel != 256) {
            defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_int16_int32_support;
        }

        const auto params = TestTransformationParams(fakeQuantizeOnData.params)
            .setUpdatePrecisions(updatePrecision)
            .setDefaultPrecisions(defaultPrecisions);

        actualFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(
            TestTransformationParams::toParams(fakeQuantizeOnData.params),
            precision,
            shape,
            fakeQuantizeOnData.actual,
            fakeQuantizeOnData.addNotPrecisionPreservedOperation);

        auto supportedPrecisions = std::vector<ngraph::pass::low_precision::PrecisionsRestriction>({
           ngraph::pass::low_precision::PrecisionsRestriction::create<ngraph::opset1::AvgPool>({{0, params.precisionsOnActivations}})
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisions, {}, { ngraph::element::f32, defaultPrecisions });
        transform.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(params);
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getReference(
            TestTransformationParams::toParams(fakeQuantizeOnData.params),
            precision,
            shape,
            params.updatePrecisions,
            fakeQuantizeOnData.expected,
            fakeQuantizeOnData.expectedFakeQuantizeOnDataPrecision,
            fakeQuantizeOnData.expectedValues.find(element::f32)->second,
            fakeQuantizeOnData.addNotPrecisionPreservedOperation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::PartialShape shape;
        bool updatePrecision;
        FakeQuantizeTransformationTestValues fakeQuantizeOnData;
        std::tie(precision, shape, updatePrecision, fakeQuantizeOnData) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shape << "_" << toString(fakeQuantizeOnData.params) <<
            (updatePrecision ? "" : "_notUpdatePrecision_") <<
            fakeQuantizeOnData;
        return result.str();
    }
};

TEST_P(FakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::i32,
    ngraph::element::f16
};

const std::vector<bool> updatePrecisions = { true, false };

const std::vector<ngraph::PartialShape> shapes = {
    { 1, 3, 72, 48 },
    { -1, -1, -1, -1 },
    PartialShape::dynamic(),
    // TODO: 3D tensor
};

const std::vector<FakeQuantizeTransformationTestValues> fakeQuantizeTransformationTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.01f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.01f }} }
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { -1.23f }, { 2.55f }, { -1.23f }, { 2.55f } },
        { 256ul, {}, { -1.23f }, { 2.55f }, { 0.f }, { 255.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, {{}, { 82.97619048f }, { 0.014823529f }} },
            { ngraph::element::f16, {{}, { 83.f }, { 0.014823529f }} }
        },
        true
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
        { 256ul, {}, { -1.28f} , { 1.27f }, { 0.f }, { 255.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, {{}, { 128.f }, { 0.01f }} },
            { ngraph::element::f16, {{}, { 128.f }, { 0.01f }} }
        },
        true
    },

    // I8
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { -1.28f}, { 1.27f }, { -1.28f}, { 1.27f } },
        { 256ul, {}, { -1.28f}, { 1.27f }, { -128.f}, { 127.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, { }, { 0.01f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, { }, { 0.01f }} }
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { -0.12f}, { 1.27f }, { -0.12f}, { 1.27f } },
        { 256ul, {}, { -0.12f}, { 1.27f }, { -128.f}, { 127.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{}, { -105.9856115f }, { 0.00545098f }} },
            { ngraph::element::f16, {{}, { -105.9856115f }, { 0.00545098f }} }
        },
        true
    },
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { -128.f }, { 127.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{}, { -128.f }, { 0.01f }} },
            { ngraph::element::f16, {{}, { -128.f }, { 0.01f }} }
        },
        true
    },
    // dot interval
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 2.55f }, { 2.55f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { 1.f }, { 1.f } },
        ngraph::element::Type_t::i8,
        {
            { ngraph::element::f32, {{}, {}, { 2.55f }} }
        },
        true
    },

    // efficientnet-b0: efficientnet-b0/model/blocks_2/depthwise_conv2d/depthwise/fq_input_0, interval: -0.504395 - +0.5
    // I8 symmetric: max ratio = 0.000907078
    {
        LayerTransformation::createParamsU8I8AndI8(),
        { 256ul, {}, { -0.504395f }, { 0.5f }, { -0.504395f }, { 0.5 } },
        { 256ul, {}, { -0.504395f }, { 0.5f }, { -128.f }, { 127.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, { }, { -0.504395f / -128.0f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, { }, { -0.504395f / -128.0f }} }
        }
    },

    // denormal values
    {
        LayerTransformation::createParamsU8I8AndI8(),
        { 256ul, {}, { 0.f }, { 25.5f }, { -1.0686283872061019e-38 }, { 1.0686283872061019e-38 } },
        { 256ul, {}, { 0.f }, { 25.5f }, { -128.f }, { 127.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, {}, { 1e-32f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, {}, { 1e-32f }} }
        }
    },
    // denormal values
    {
        LayerTransformation::createParamsU8I8AndI8(),
        { 256ul, {}, { 0.f }, { 25.5f }, { 0.0 }, { 1.0686283872061019e-38 } },
        { 256ul, {}, { 0.f }, { 25.5f }, { 0.0 }, { 255 } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, {}, { 1e-32f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, {}, { 1e-32f }} }
        }
    },
    // U16
    {
        LayerTransformation::createParamsU8I8(),
        { 65536ul, {}, { 0.f }, { 65.535f }, { 0.f }, { 65.535f } },
        { 65536ul, {}, { 0.f }, { 65.535f }, { 0.f }, { 65535.f } },
        ngraph::element::u16,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.001f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.001f }} }
        }
    },
    // I16
    {
        LayerTransformation::createParamsU8I8(),
        { 65536ul, {}, { -32.768f }, { 32.767f }, { -32.768f }, { 32.767f } },
        { 65536ul, {}, { -32.768f }, { 32.767f }, { -32768.f }, { 32767.f } },
        ngraph::element::i16,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.001f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.001f }} }
        }
    },
    // U32
    {
        LayerTransformation::createParamsU8I8(),
        { static_cast<size_t>(4294967296), {}, { 0.f }, { 4.294967295f }, { 0.f }, { 4.294967295f } },
        { static_cast<size_t>(4294967296), {}, { 0.f }, { 4.294967295f }, { 0.f }, { 4294967295.f } },
        ngraph::element::u32,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.000000001f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.000000001f }} }
        }
    },
    // I32
    {
        LayerTransformation::createParamsU8I8(),
        { static_cast<size_t>(4294967296), {}, { -2.147483648f }, { 2.147483647f }, { -2.147483648f }, { 2.147483647f } },
        { static_cast<size_t>(4294967296), {}, { -2.147483648f }, { 2.147483647f }, { -2147483648.f }, { 2147483647.f } },
        ngraph::element::i32,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.000000001f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.000000001f }} }
        }
    },
    // Failed when updatePrecisions = false, U8 per-channel
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        256ul,
    //        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    //        { 0.f, 0.f, 0.f }, { 2.55f, 2.55f, 2.55f },
    //        { 0.f, 0.f, 0.f }, { 2.55f, 25.5f, 255.f }
    //    },
    //    { 256ul, {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
    //    ngraph::element::u8,
    //    {
    //        { ngraph::element::f32, { {ngraph::element::f32}, {}, { {0.01f, 0.1f, 1.f} }} },
    //        { ngraph::element::f16, { {ngraph::element::f16}, {}, { {0.01f, 0.1f, 1.f} }} }
    //    }
    //},
    // u4 through u8
    {
        LayerTransformation::createParamsU8I8(),
        { 16ul, {}, { 0.f }, { 1.5f }, { 0.f }, { 1.5f } },
        { 16ul, {}, { 0.f }, { 1.5f }, { 0.f }, { 15.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.1f }} },
            { ngraph::element::f16, { {ngraph::element::f16}, {}, { 0.1f }} }
        }
    },
    // i4 through i8
    {
        LayerTransformation::createParamsI8I8(),
        { 16ul, {}, { -0.8f }, { 0.7f }, { -0.8f }, { 0.7f } },
        { 16ul, {}, { -0.8f }, { 0.7f }, { -8.f }, { 7.f } },
        ngraph::element::i8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, { }, { 0.1f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, { }, { 0.1f }} }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeTransformation::getTestCaseName);
} // namespace testValues1
