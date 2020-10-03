// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "transformations/low_precision/multiply.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/multiply_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class MultiplyTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    MultiplyValues actual;
    MultiplyValues expected;

    MultiplyTransformationTestValues() = default;

    MultiplyTransformationTestValues(
        low_precision::LayerTransformation::Params transformationParams,
        MultiplyValues actual,
        MultiplyValues expected):
        transformationParams(std::move(transformationParams)),
        actual(std::move(actual)),
        expected(std::move(expected)) {}
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    MultiplyTransformationTestValues> MultiplyTransformationParams;

class MultiplyTransformation : public LayerTransformation, public testing::WithParamInterface<MultiplyTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool broadcast = std::get<2>(GetParam());
        const MultiplyTransformationTestValues testParams = std::get<3>(GetParam());

        actualFunction = MultiplyFunction::get(shape, testParams.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::MultiplyTransformation, ngraph::opset1::Multiply>(
            low_precision::LayerTransformation::Params(testParams.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = MultiplyFunction::get(shape, testParams.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool broadcast;
        MultiplyTransformationTestValues params;
        std::tie(precision, shape, broadcast, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, params.transformationParams) <<
            (broadcast ? "_broadcast_" : "") <<
            params.actual <<
            params.expected;
        return result.str();
    }
};

TEST_P(MultiplyTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<bool> broadcastValues = {
    true,
    false
};

const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(broadcastValues),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyTransformation::getTestCaseName);
