// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include "simple_low_precision_transformer.hpp"
#include <transformations/low_precision/normalize_l2.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class NormalizeL2TransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;

    NormalizeL2ActualValues actual;
    NormalizeL2ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::op::EpsMode,
    NormalizeL2TransformationTestValues> NormalizeL2TransformationParams;

class NormalizeL2Transformation : public LayerTransformation, public testing::WithParamInterface<NormalizeL2TransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const ngraph::op::EpsMode epsMode = std::get<2>(GetParam());
        const NormalizeL2TransformationTestValues params = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
            precision,
            shape,
            epsMode,
            params.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::NormalizeL2Transformation, ngraph::opset1::NormalizeL2>(
            low_precision::LayerTransformation::Params(params.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = (!params.transformationParams.supportAsymmetricQuantization) && (!params.expected.subtractValues.empty()) ?
            ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
                precision,
                shape,
                epsMode,
                params.actual) :
            ngraph::builder::subgraph::NormalizeL2Function::getReference(
                precision,
                shape,
                epsMode,
                params.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        ngraph::Shape axes;
        ngraph::op::EpsMode epsMode;
        NormalizeL2TransformationTestValues params;
        std::tie(precision, shape, epsMode, params) = obj.param;

        std::ostringstream result;
        result << toString(params.transformationParams) << precision << "_" << shape << "_" <<
            axes << epsMode << params.actual << params.expected;
        return result.str();
    }
};

TEST_P(NormalizeL2Transformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 4, 16, 16 }
};

std::vector<ngraph::op::EpsMode> epsMode = {
    ngraph::op::EpsMode::ADD,
    ngraph::op::EpsMode::MAX
};

const std::vector<NormalizeL2TransformationTestValues> normalizeL2TransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        { ngraph::element::u8, { 1 }, { 2.f }, { -12.3f, -12.3f, -12.3f, -12.3f }},
        { ngraph::element::u8, { 1 }, { 2.f }, { -1.f,   -1.f,   -1.f, -1.f}}
    },

    // U8
    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 1 }, { 2.f }, { -12.3f, -12.3f, -12.3f, -12.3f }},
        { ngraph::element::u8, { 1 }, { 2.f }, { -1.f,   -1.f,   -1.f, -1.f}}
    },

    {
        LayerTransformation::createParamsU8I8(),
        { ngraph::element::u8, { 1, 2, 3 }, { }, { 12.3f }},
        { ngraph::element::u8, { 1, 2, 3 }, { }, { 1.f }}
    },

    // I8
    {
        LayerTransformation::createParamsI8I8(),
        { ngraph::element::i8, { 1 }, { 2.f }, { -12.3f, -12.3f, -12.3f, -12.3f }},
        { ngraph::element::i8, { 1 }, { 2.f }, { -1.f,   -1.f,   -1.f, -1.f}}
    },

    {
        LayerTransformation::createParamsI8I8(),
        { ngraph::element::i8, { 1, 2, 3 }, { }, { 12.3f }},
        { ngraph::element::i8, { 1, 2, 3 }, { }, { 1.f }}
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(epsMode),
        ::testing::ValuesIn(normalizeL2TransformationTestValues)),
    NormalizeL2Transformation::getTestCaseName);
