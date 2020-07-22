// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/mat_mul.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class MatMullTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::Shape weightsConstShape;
        std::vector<float> weightsConstValues;
        ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type weightsConstPrecision;
        ngraph::Shape weightsConstShape;
        std::vector<float> weightsConstValues;
        ngraph::element::Type precisionBeforeOperation1;
        ngraph::element::Type precisionBeforeOperation2;
        ngraph::builder::subgraph::DequantizationOperations resultDequantization;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.dequantization << "_" << actual.fqOnWeights;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" << expected.dequantization << "_" << expected.resultDequantization;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params,
    bool,
    MatMullTransformationTestValues> MatMulTransformationParams;

class MatMulWithConstantTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const bool updatePrecisions = std::get<3>(GetParam());
        const MatMullTransformationTestValues testValues = std::get<4>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(
            precision,
            shape,
            testValues.actual.precisionBeforeDequantization,
            {
                testValues.actual.dequantization.convert.outPrecision == ngraph::element::undefined ?
                    precision :
                    testValues.actual.dequantization.convert.outPrecision,
                testValues.actual.dequantization.subtractValues,
                testValues.actual.dequantization.multiplyValues
            },
            testValues.actual.weightsConstShape,
            testValues.actual.weightsConstValues,
            testValues.actual.fqOnWeights);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getReference(
            precision,
            shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantization,
            testValues.expected.weightsConstPrecision,
            testValues.expected.weightsConstShape,
            testValues.expected.weightsConstValues,
            testValues.expected.resultDequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shapes;
        low_precision::LayerTransformation::Params params;
        bool updatePrecisions;
        MatMullTransformationTestValues testValues;
        std::tie(precision, shapes, params, updatePrecisions, testValues) = obj.param;

        std::stringstream ss;
        ss << LayerTransformation::getTestCaseNameByParams(precision, shapes, params) << "_" <<
            (updatePrecisions ? "updatePrecisions_" : "notUpdatePrecisions_") <<
            testValues;
        return ss.str();
    }
};

TEST_P(MatMulWithConstantTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = { { 1, 2048 } };

const std::vector<low_precision::LayerTransformation::Params> params = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

const std::vector<bool> updatePrecisions = { true, false };

std::vector<MatMullTransformationTestValues> testValues = {
    // U8 & I8
    //{
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        { ngraph::element::undefined, {}, { 0.02f } },
    //        { 2048, 1000 },
    //        std::vector<float>(2048 * 1000, 1.f),
    //        { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
    //    },
    //    {
    //        ngraph::element::u8,
    //        { ngraph::element::undefined, {}, {} },
    //        ngraph::element::i8,
    //        {2048, 1000},
    //        std::vector<float>(2048 * 1000, 1.f),
    //        ngraph::element::u8,
    //        ngraph::element::i8,
    //        { ngraph::element::undefined, {}, { 0.02f * 0.1f } },
    //    }
    //},
    // I8 & I8
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { ngraph::element::undefined, {}, { 0.02f } },
            { 2048, 1000 },
            std::vector<float>(2048 * 1000, 1.f),
            { 255, { 1, 1 },  {0.f}, {254.f}, {-12.7f}, {12.7} },
        },
        {
            ngraph::element::i8,
            { ngraph::element::undefined, {}, {} },
            ngraph::element::i8,
            {2048, 1000},
            std::vector<float>(2048 * 1000, -126),
            ngraph::element::i8,
            ngraph::element::i8,
            { ngraph::element::undefined, {}, { 0.02f * 0.1f } },
        }
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);

} // namespace
