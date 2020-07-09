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
#include <transformations/low_precision/max_pool.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class MaxPoolTransformationTestValues {
public:
    ngraph::builder::subgraph::MaxPoolFunction::ActualValues actual;
    ngraph::builder::subgraph::MaxPoolFunction::ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    low_precision::LayerTransformation::Params,
    MaxPoolTransformationTestValues> MaxPoolTransformationParams;

class MaxPoolTransformation : public LayerTransformation, public testing::WithParamInterface<MaxPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const MaxPoolTransformationTestValues testValues = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::MaxPoolFunction::getOriginal(precision, shape, params, testValues.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MaxPoolFunction::getReference(precision, shape, params, testValues.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const low_precision::LayerTransformation::Params params = std::get<2>(obj.param);
        const MaxPoolTransformationTestValues testValues = std::get<3>(obj.param);

        return LayerTransformation::getTestCaseNameByParams(precision, shape, params);
    }
};

TEST_P(MaxPoolTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<low_precision::LayerTransformation::Params> params = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    {
        // ActualValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.02f }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.00f }
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    DISABLED_LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
