// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/clamp.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/clamp_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class ClampTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    ngraph::builder::subgraph::ClampFunction::ActualValues actual;
    ngraph::builder::subgraph::ClampFunction::ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    ClampTransformationTestValues> ClampTransformationParams;

class ClampTransformation : public LayerTransformation, public testing::WithParamInterface<ClampTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecisions = std::get<2>(GetParam());
        const ClampTransformationTestValues testValues = std::get<3>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testValues.transformationParams).
            setUpdatePrecisions(updatePrecisions);

        actualFunction = ngraph::builder::subgraph::ClampFunction::getOriginal(
            precision,
            shape,
            updatePrecisions,
            testValues.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ClampTransformation, ngraph::opset1::Clamp>(testValues.transformationParams);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ClampFunction::getReference(
            precision,
            shape,
            updatePrecisions,
            testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const bool updatePrecisions = std::get<2>(obj.param);
        ClampTransformationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.transformationParams.setUpdatePrecisions(updatePrecisions)) <<
            testValues.actual << testValues.expected;
        return result.str();
    }
};

TEST_P(ClampTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 224, 224 }
};

const std::vector<bool> updatePrecision = {
    true,
    false
};

const std::vector<ClampTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { 128.f },
            { 3.f }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128.f },
            { 3.f }
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { 128.f },
            { -5.f }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128.f },
            { -5.f }
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { },
            { 3.f }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { },
            { 3.f }
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        // ActualValues
        {
            ngraph::element::i8,
            { 0.f },
            { 5.f }
        },
        // ExpectedValues
        {
            ngraph::element::i8,
            { 0.f },
            { 5.f }
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { 0.f, 0.f, -255.f },
            { 0.01f, 0.01f, 0.005f }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 0.f, 0.f, -255.f },
            { 0.01f, 0.01f, 0.005f }
        }
    }
};
INSTANTIATE_TEST_CASE_P(
    LPT,
    ClampTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecision),
        ::testing::ValuesIn(testValues)),
    ClampTransformation::getTestCaseName);
