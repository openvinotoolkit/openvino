// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "low_precision/convolution.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include "ov_lpt_models/fake_quantize_and_convolution.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class FakeQuantizeWithNotOptimalTransformationTestValues {
public:
    class Values {
    public:
        ov::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;
        ov::builder::subgraph::DequantizationOperations::Convert convertOnData;
        ov::builder::subgraph::DequantizationOperations dequantizationOnData;
        ov::builder::subgraph::Constant constantOnWeights;
        ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };
    TestTransformationParams params;
    Values actual;
    Values expected;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeWithNotOptimalTransformationTestValues& testValue) {
    return out << "_" <<
        testValue.actual.fqOnData << "_" << testValue.actual.fqOnWeights <<
        testValue.expected.fqOnData << "_" << testValue.expected.fqOnWeights;
}

typedef std::tuple<
    ov::element::Type,
    ov::Shape,
    bool,
    FakeQuantizeWithNotOptimalTransformationTestValues> FakeQuantizeWithNotOptimalTransformationParams;

class FakeQuantizeWithNotOptimalTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<FakeQuantizeWithNotOptimalTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::Shape shape = std::get<1>(GetParam());
        const bool updatePrecision = std::get<2>(GetParam());
        const FakeQuantizeWithNotOptimalTransformationTestValues testValues = std::get<3>(GetParam());

        const auto params = TestTransformationParams(testValues.params).setUpdatePrecisions(updatePrecision);

        actualFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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

        auto precisionsRestrictions = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>({
                {{0}, {ov::element::u8}},
                {{1}, {ov::element::i8}}
            })
        });

        auto quantizationRestrictions = std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
            ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::Convolution>()
        });

        SimpleLowPrecisionTransformer transformer(precisionsRestrictions, quantizationRestrictions);
        transformer.add<ov::pass::low_precision::ConvolutionTransformation, ov::op::v1::Convolution>(
            TestTransformationParams(params).setPrecisionsOnActivations({ element::u8 }));
        transformer.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::FakeQuantizeAndConvolutionFunction::get(
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
        ov::element::Type precision;
        ov::Shape shape;
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    //ov::element::i32,
    //ov::element::f16
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
            { 256ul, {{ 1, 1, 1, 1 }}, { 0.f }, { 2.55f }, { -128.f }, { 127.f }, ov::element::i8 },
            { ov::element::i8, false },
            {
                { ov::element::f32, false },
                { {-128.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
                { {0.01f}, ov::element::f32, {}, false }
            },
            {{5.f}, ov::element::i8},
            {},
            {
                { ov::element::f32, false },
                { {127.f}, ov::element::f32, {}, false, 1ul, ov::element::i8, true },
                { {0.03f}, ov::element::f32, {}, false }
            },
            {}
        },
        {
            { 256ul, {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }, {}, {}}, { 0.f }, { 2.55f }, { 0.f }, { 255.f }, ov::element::u8 },
            { ov::element::u8, false },
            {},
            {{5.f}, ov::element::i8},
            {},
            {
                {},
                { std::vector<float>(64, 127.f), ov::element::f32,
                 {64, 1, 1, 1}, false, 1ul, ov::element::i8, false,
                 {{ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()}}},
                {}
            },
            {
                { },
                { },
                { {0.0003f}, ov::element::f32, {}}
            }
        },
    }
};

const std::vector<ov::Shape> shapes = {
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
