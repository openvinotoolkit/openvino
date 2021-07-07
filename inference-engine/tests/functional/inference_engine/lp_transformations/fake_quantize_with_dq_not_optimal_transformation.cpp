// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeWithNotOptimalTransformationTestValues {
public:
    class Values {
    public:
        builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;
        builder::subgraph::DequantizationOperations::Convert convertOnData;
        builder::subgraph::DequantizationOperations dequantizationOnData;
        builder::subgraph::Constant constantOnWeights;
        builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        builder::subgraph::DequantizationOperations dequantizationOnWeights;
        builder::subgraph::DequantizationOperations dequantizationAfter;
    };
    low_precision::LayerTransformation::Params params;
    Values actual;
    Values expected;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeWithNotOptimalTransformationTestValues& testValue) {
    return out << "_" <<
        testValue.actual.fqOnData << "_" << testValue.actual.fqOnWeights <<
        testValue.expected.fqOnData << "_" << testValue.expected.fqOnWeights;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    FakeQuantizeWithNotOptimalTransformationTestValues> FakeQuantizeWithNotOptimalTransformationParams;

class FakeQuantizeWithNotOptimalTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeWithNotOptimalTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizeWithNotOptimalTransformationTestValues testValues = std::get<3>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testValues.params).
            setUpdatePrecisions(updatePrecision);

        actualFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            precision,
            shape,
            testValues.actual.fqOnData,
            testValues.actual.convertOnData,
            testValues.actual.dequantizationOnData,
            testValues.actual.constantOnWeights,
            testValues.actual.fqOnWeights,
            {},
            testValues.actual.dequantizationOnWeights,
            testValues.actual.dequantizationAfter);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(
            low_precision::LayerTransformation::Params(params).setPrecisionsOnActivations({ element::u8 }));
        transformer.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
            precision,
            shape,
            testValues.expected.fqOnData,
            {},
            testValues.expected.dequantizationOnData,
            testValues.expected.constantOnWeights,
            testValues.expected.fqOnWeights,
            {},
            testValues.expected.dequantizationOnWeights,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeWithNotOptimalTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool updatePrecision;
        FakeQuantizeWithNotOptimalTransformationTestValues fakeQuantizeOnData;
        std::tie(precision, shape, updatePrecision, fakeQuantizeOnData) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, fakeQuantizeOnData.params) <<
            (updatePrecision ? "" : "_notUpdatePrecision_") <<
            fakeQuantizeOnData;
        return result.str();
    }
};

TEST_P(FakeQuantizeWithNotOptimalTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::i32,
    //ngraph::element::f16
};

const std::vector<bool> updatePrecisions = { true/*, false*/ };

const std::vector<FakeQuantizeWithNotOptimalTransformationTestValues> fakeQuantizeTransformationTestValues = {
    // Actual:
    //
    // FakeQuantize
    //  |FP32
    //  |
    // Convert   Constant
    //  |I8         |I8
    //  |           |
    // Convert    Convert
    //   \FP32    /FP32
    //    \      /
    //    Subtract  Constant  Constant
    //      \FP32   /FP32      |FP32   Constant Constant Constant Constant
    //       \     /           |       /FP32    /FP32    /FP32    /FP32
    //       Multiply         FakeQuantize
    //         \FP32         /FP32
    //          \           /
    //           Convolution
    //
    // Transformed:
    //
    // FakeQuantize  Constant
    //   \U8        /U8
    //    \        /
    //     Subtract   Constant
    //      \FP32    /I8
    //       \      /
    //       Convolution  Constant
    //         \FP32      /FP32
    //          \        /
    //           Multiply
    {
        LayerTransformation::createParamsU8I8AndI8(),
        {
            { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 2.55f }, { -128.f }, { 127.f }, ngraph::element::i8 },
            { ngraph::element::i8, false },
            {
                { ngraph::element::f32, false },
                { {-128.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
                { {0.01f}, ngraph::element::f32, {}, false }
            },
            {{5.f}, ngraph::element::i8},
            {},
            {
                { ngraph::element::f32, false },
                { {127.f}, ngraph::element::f32, {}, false, 1ul, ngraph::element::i8, true },
                { {0.03f}, ngraph::element::f32, {}, false }
            },
            {}
        },
        {
            { 256ul, {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }, {}, {}}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, ngraph::element::u8 },
            { ngraph::element::u8, false },
            {},
            {{5.f}, ngraph::element::i8},
            {},
            {
                {},
                { std::vector<float>(64, 127.f), ngraph::element::f32, {64, 1, 1, 1}, false, 1ul, ngraph::element::i8, false, {"DISABLED_CONSTANT_FOLDING"}},
                {}
            },
            {
                { },
                { },
                { {0.0003f}, ngraph::element::f32, {1, 1, 1, 1}}
            }
        },
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeWithNotOptimalTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeWithNotOptimalTransformation::getTestCaseName);
