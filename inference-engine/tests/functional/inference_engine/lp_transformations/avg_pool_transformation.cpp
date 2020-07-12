// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/avg_pool.hpp>
#include <transformations/low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/avg_pool_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class AvgPoolTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    AvgPoolTransformationTestValues> AvgPoolTransformationParams;

class AvgPoolTransformation : public LayerTransformation, public testing::WithParamInterface<AvgPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const AvgPoolTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::AvgPoolFunction::getReference(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });
    }

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const AvgPoolTransformationTestValues testValues = std::get<2>(obj.param);
        return LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params);
    }
};

TEST_P(AvgPoolTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<AvgPoolTransformationTestValues> testValues = {
    { LayerTransformation::createParamsU8I8(), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsU8I8().setUpdatePrecisions(false), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsI8I8(), { 128 }, { 0.02f } },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    AvgPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    AvgPoolTransformation::getTestCaseName);
