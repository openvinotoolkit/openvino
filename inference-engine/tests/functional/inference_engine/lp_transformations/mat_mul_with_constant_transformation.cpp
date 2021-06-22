// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/mat_mul.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class MatMullTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::Shape inputShape;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnData;

        ngraph::builder::subgraph::Constant weights;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    class Expected {
    public:
        ngraph::Shape inputShape;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnData;
        ngraph::builder::subgraph::Constant weights;

        ngraph::element::Type precisionBeforeOperation;
        ngraph::builder::subgraph::DequantizationOperations resultDequantization;

        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
        ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" <<
        actual.inputShape << "_" <<
        actual.precisionBeforeDequantization << "_" <<
        actual.dequantizationOnData << "_" <<
        actual.weights.shape << "_" <<
        actual.fqOnWeights << "_" <<
        actual.dequantizationOnWeights;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" <<
        expected.weights.shape <<"_" <<
        expected.dequantizationOnData << "_" <<
        expected.precisionBeforeOperation << "_" <<
        expected.resultDequantization << "_" <<
        expected.fqOnWeights << "_" <<
        expected.dequantizationOnWeights;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" <<
        values.params.support3DTensorOnActivations << "_" <<
        values.actual << "_" <<
        values.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    size_t,
    MatMullTransformationTestValues> MatMulTransformationParams;

class MatMulWithConstantTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const size_t batch = std::get<1>(GetParam());

        MatMullTransformationTestValues testValues = std::get<2>(GetParam());
        testValues.actual.inputShape[0] = batch;
        testValues.expected.inputShape[0] = batch;


        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(
            precision,
            testValues.actual.inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantizationOnData,
            testValues.actual.weights,
            testValues.actual.fqOnWeights,
            testValues.actual.dequantizationOnWeights);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = (testValues.expected.fqOnWeights.empty() && testValues.expected.dequantizationOnWeights.empty()) ?
            ngraph::builder::subgraph::MatMulFunction::getReference(
                precision,
                testValues.expected.inputShape,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationOnData,
                testValues.expected.weights,
                testValues.expected.resultDequantization) :
            ngraph::builder::subgraph::MatMulFunction::getOriginal(
                precision,
                testValues.expected.inputShape,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationOnData,
                testValues.expected.weights,
                testValues.expected.fqOnWeights,
                testValues.expected.dequantizationOnWeights);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        size_t batch;
        MatMullTransformationTestValues testValues;
        std::tie(precision, batch, testValues) = obj.param;

        std::stringstream ss;
        ss << precision << "_" << batch << "_" << testValues;
        return ss.str();
    }
};

TEST_P(MatMulWithConstantTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<size_t> batches = { 1, 4 };

std::vector<MatMullTransformationTestValues> testValues = {
    // supported 3D: U8 & I8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(1024 * 1024, 1.f), ngraph::element::f32, ngraph::Shape{ 1024, 1024 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            {},
            { std::vector<float>(1024 * 1024, -126.f), ngraph::element::i8, ngraph::Shape{ 1024, 1024 } },
            ngraph::element::u8,
            { {}, {}, { 0.02f * 0.1f } },
            {},
            {}
        }
    },

    // supported 3D: U8 & I8 with Dq on weights
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(1024 * 1024, 1.f), ngraph::element::i8, ngraph::Shape{ 1024, 1024 } },
            {},
            { ngraph::element::f32, {}, { 0.1f } },
        },
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            {},
            { std::vector<float>(1024 * 1024, 1.f), ngraph::element::i8, ngraph::Shape{ 1024, 1024 } },
            ngraph::element::u8,
            { {}, {}, { 0.02f * 0.1f } },
            {},
            {}
        }
    },

    // support3DTensorOnActivations = false for 3D tensor
    {
        LayerTransformation::createParamsU8I8().setSupport3DTensorOnActivations(false),
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(1024 * 1024, 1.f), ngraph::element::f32, { 1024, 1024 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} }
        },
        {
            { 1, 384, 1024 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(1024 * 1024, 1.f), ngraph::element::f32, { 1024, 1024 } },
            ngraph::element::i8,
            {},
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        }
    },

    // 3D: U8 & I8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f} } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            {},
            { std::vector<float>(4 * 4, -126.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            ngraph::element::u8,
            { {}, {}, { {0.001f, 0.002f, 0.003f} } },
            {},
            {}
        }
    },

    // 3D: U8 & I8 with Dq on weights
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f} } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            {},
            { ngraph::element::f32, {}, { 0.1f } }
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { {}, {}, {} },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            ngraph::element::u8,
            { {}, {}, { {0.001f, 0.002f, 0.003f} } },
            {},
            {}
        }
    },

    // 3D: U8 & I8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            {
                255,
                { 1, 4 },
                {0.f, 0.f, 0.f, 0.f},
                {254.f, 254.f, 254.f, 254.f},
                {-127.f, -12.7f, -1.27f , -0.127f},
                {127.f, 12.7f, 1.27f , 0.127f},
            },
            {}
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { {}, {}, {} },
            { std::vector<float>(4 * 4, -126.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            ngraph::element::u8,
            { {}, {}, {{ 0.02f, 0.002f, 0.0002f, 0.00002f }, ngraph::element::f32, ngraph::Shape{ 1, 1, 4 }}},
            {},
            {}
        }
    },

    // 3D: U8 & I8 with Dq on weights with different values
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            {},
            { ngraph::element::f32, {}, { {1.f, 0.1f, 0.01f, 0.001f} } }
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { {}, {}, {} },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::i8, ngraph::Shape{ 4, 4 } },
            ngraph::element::u8,
            { {}, {}, {{ 0.02f, 0.002f, 0.0002f, 0.00002f }, ngraph::element::f32, ngraph::Shape{ 1, 1, 4 }}},
            {},
            {}
        }
    },

    // 3D: U8 & I8: dequantization by columns in first input: can't be transformed
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f, 0.01f}, ngraph::element::f32, ngraph::Shape{1, 1, 4} } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f, 0.01f}, ngraph::element::f32, ngraph::Shape{1, 1, 4} } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            ngraph::element::f32,
            {},
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        }
    },

    // U8 & I8: dequantization by rows in second input: can't be transformed
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            {
                255,
                { 4, 1 },
                {0.f, 0.f, 0.f, 0.f},
                {254.f, 254.f, 254.f, 254.f},
                {-127.f, -12.7f, -1.27f , -0.127f},
                {127.f, 12.7f, 1.27f , 0.127f},
            },
            {}
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            ngraph::element::f32,
            {},
            {
                255,
                { 4, 1 },
                {0.f, 0.f, 0.f, 0.f},
                {254.f, 254.f, 254.f, 254.f},
                {-127.f, -12.7f, -1.27f , -0.127f},
                {127.f, 12.7f, 1.27f , 0.127f},
            },
            {}
        },
    },

    // U8 & I8: dequantization by rows in second input: can't be transformed (Dq on weights)
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            {},
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f, 0.01f}, ngraph::element::f32, ngraph::Shape{4, 1} } },
        },
        {
            { 1, 3, 4 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(4 * 4, 1.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
            ngraph::element::u8,
            {},
            {},
            { ngraph::element::f32, {}, { {0.01f, 0.02f, 0.03f, 0.01f}, ngraph::element::f32, ngraph::Shape{4, 1} } },
        },
    },

    // 2D: U8 & I8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 2048 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 2048 },
            ngraph::element::u8,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, -126.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::u8,
            { {}, {}, { 0.02f * 0.1f } },
            {},
            {}
        }
    },

    // 2D: U8 & I8 with Dq on weights
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 1, 2048 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.2f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            {},
            { ngraph::element::f32, {}, { 0.2f } }
        },
        {
            { 1, 2048 },
            ngraph::element::u8,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::u8,
            { {}, {}, { 0.2f * 0.2f } },
            {},
            {}
        }
    },

    // support3DTensorOnActivations = false, but 2D tensor is used
    {
        LayerTransformation::createParamsU8I8().setSupport3DTensorOnActivations(false),
        {
            { 1, 2048 },
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, { 2048, 1000 }},
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 2048 },
            ngraph::element::u8,
            {},
            { std::vector<float>(2048 * 1000, -126), ngraph::element::i8, { 2048, 1000 }},
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.1f } },
            {}
        }
    },

    // 2D: I8 & I8
    {
        LayerTransformation::createParamsI8I8(),
        {
            { 1, 2048 },
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 2048 },
            ngraph::element::i8,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, -126.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.1f } },
            {},
            {}
        }
    },

    // 2D: I8 & I8 with Dq on weights with small subtract values
    {
        LayerTransformation::createParamsI8I8(),
        {
            { 1, 2048 },
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            {},
            { ngraph::element::f32, { 1e-7f }, { 0.02f } }
        },
        {
            { 1, 2048 },
            ngraph::element::i8,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.02f } },
            {},
            {}
        }
    },

    // 2D: I8 & I8 with Dq on weights with zero subtract values
    {
        LayerTransformation::createParamsI8I8(),
        {
            { 1, 2048 },
            ngraph::element::i8,
            { ngraph::element::f32, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            {},
            { ngraph::element::f32, { 0.f }, { 0.02f } }
        },
        {
            { 1, 2048 },
            ngraph::element::i8,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::i8, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::i8,
            { {}, {}, { 0.02f * 0.02f } },
            {},
            {}
        }
    },

    // 2D: FP32 & FP32
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            { 1, 2048 },
            ngraph::element::f32,
            { {}, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
            {}
        },
        {
            { 1, 2048 },
            ngraph::element::f32,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, -126.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::f32,
            { {}, {}, { 0.02f * 0.1f } },
            {},
            {}
        }
    },

    // 2D: FP32 & FP32 with Dq on weights
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            { 1, 2048 },
            ngraph::element::f32,
            { {}, {}, { 0.02f } },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            {},
            { ngraph::element::f32, {}, { 0.02f } }
        },
        {
            { 1, 2048 },
            ngraph::element::f32,
            { {}, {}, {} },
            { std::vector<float>(2048 * 1000, 1.f), ngraph::element::f32, ngraph::Shape{ 2048, 1000 } },
            ngraph::element::f32,
            { {}, {}, { 0.02f * 0.02f } },
            {},
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(batches),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);

} // namespace
