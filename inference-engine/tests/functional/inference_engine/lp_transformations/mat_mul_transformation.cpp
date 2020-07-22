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
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
        ngraph::element::Type precisionBeforeOperation;
        ngraph::builder::subgraph::DequantizationOperations result;
    };

    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Actual& actual) {
    return out << "_" << actual.dequantization1 << "_" << actual.dequantization2;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues::Expected& expected) {
    return out << "_" << expected.dequantization1 << "_" << expected.dequantization2 << "_" << expected.result;
}

inline std::ostream& operator << (std::ostream& out, const MatMullTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<
    ngraph::element::Type,
    std::pair<ngraph::Shape, ngraph::Shape>,
    ngraph::pass::low_precision::LayerTransformation::Params,
    bool,
    MatMullTransformationTestValues> MatMulTransformationParams;

class MatMulTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const std::pair<ngraph::Shape, ngraph::Shape> shapes = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const bool updatePrecisions = std::get<3>(GetParam());
        const MatMullTransformationTestValues testValues = std::get<4>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(
            precision,
            shapes.first,
            testValues.actual.dequantization1,
            shapes.second,
            testValues.actual.dequantization2);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getReference(
            precision,
            shapes.first,
            testValues.expected.dequantization1,
            shapes.second,
            testValues.expected.dequantization2,
            testValues.expected.result);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        std::pair<ngraph::Shape, ngraph::Shape> shapes;
        low_precision::LayerTransformation::Params params;
        bool updatePrecisions;
        MatMullTransformationTestValues testValues;
        std::tie(precision, shapes, params, updatePrecisions, testValues) = obj.param;

        std::stringstream ss;
        ss << LayerTransformation::getTestCaseNameByParams(precision, shapes.first, params) << "_" <<
            (updatePrecisions ? "updatePrecisions_" : "notUpdatePrecisions_") <<
            testValues;
        return ss.str();
    }
};

TEST_P(MatMulTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape>> shapes = {
    { { 1, 16, 384, 64 }, { 1, 16, 64, 384 } }
};

const std::vector<low_precision::LayerTransformation::Params> params = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

const std::vector<bool> updatePrecisions = { true, false };

std::vector<MatMullTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { ngraph::element::undefined, {}, { 0.02f } },
            { ngraph::element::undefined, {}, { 0.03f } },
        },
        {
            ngraph::element::u8,
            { ngraph::element::undefined, {}, {} },
            { ngraph::element::undefined, {}, {} },
            ngraph::element::u8,
            { ngraph::element::undefined, {}, { 0.02f * 0.03f } },
        }
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(testValues)),
    MatMulTransformation::getTestCaseName);

} // namespace
