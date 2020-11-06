// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class MaxPoolTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    MaxPoolTransformationTestValues> MaxPoolTransformationParams;

class MaxPoolTransformation : public LayerTransformation, public testing::WithParamInterface<MaxPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const MaxPoolTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::MaxPoolFunction::getOriginal(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MaxPoolFunction::getReference(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });
    }

    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const MaxPoolTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            testValues.subtractValues.size() << "_" <<
            testValues.mutliplyValues.size() << "_";
        return result.str();
    }
};

TEST_P(MaxPoolTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
    { LayerTransformation::createParamsU8I8(), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsU8I8(), {}, { 0.02f } },
    { LayerTransformation::createParamsU8I8().setUpdatePrecisions(false), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsU8I8().setUpdatePrecisions(false), {}, { 0.02f } },
    { LayerTransformation::createParamsI8I8(), { 128 }, { 0.02f } },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
