// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <low_precision/avg_pool.hpp>
#include <low_precision/common/operation_precision_restriction.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
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

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
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

        const auto params = TestTransformationParams(fakeQuantizeOnData.params).setUpdatePrecisions(updatePrecision);

        actualFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(
            TestTransformationParams::toParams(fakeQuantizeOnData.params),
            precision,
            shape,
            fakeQuantizeOnData.actual,
            fakeQuantizeOnData.addNotPrecisionPreservedOperation);

        auto supportedPrecisions = std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>({
           ngraph::pass::low_precision::OperationPrecisionRestriction::create<ngraph::opset1::AvgPool>({{0, params.precisionsOnActivations}})
        });

        SimpleLowPrecisionTransformer transform(supportedPrecisions);
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
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
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
    { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() },
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
        { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 255.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, {{ngraph::element::f32}, { }, { 1e-32f }} },
            { ngraph::element::f16, {{ngraph::element::f16}, { }, { 1e-32f }} }
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

namespace testValues2 {
const std::vector<ngraph::PartialShape> shapesWithDynamicChannel = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
    PartialShape::dynamic(),
};

const std::vector<FakeQuantizeTransformationTestValues> fakeQuantizeTransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 255.f } },
        ngraph::element::u8,
        {
            { ngraph::element::f32, { {ngraph::element::f32}, {}, { 0.01f }} },
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            256ul,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            { 0.f, 0.f, 0.f }, { 2.55f, 2.55f, 2.55f },
            { 0.f, 0.f, 0.f }, { 2.55f, 25.5f, 255.f }
        },
        {
            256ul,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            { 0.f, 0.f, 0.f }, { 2.55f, 2.55f, 2.55f },
            { 0.f, 0.f, 0.f }, { 2.55f, 25.5f, 255.f }
        },
        ngraph::element::f32,
        {
            { ngraph::element::f32, {} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f32),
        ::testing::ValuesIn(shapesWithDynamicChannel),
        ::testing::Values(true),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeTransformation::getTestCaseName);
} // namespace testValues2
