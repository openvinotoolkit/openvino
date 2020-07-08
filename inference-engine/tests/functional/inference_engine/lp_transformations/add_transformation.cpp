// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/low_precision/add.hpp>
#include "ngraph_functions/low_precision_transformations/add_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class AddTransformation : public LayerTransformation, public testing::WithParamInterface<LayerTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::AddFunction::getOriginal(precision, shape);
        // transform(actualFunction);
        referenceFunction = ngraph::builder::subgraph::AddFunction::getReference(precision, shape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<LayerTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        std::tie(precision, shape, params) = obj.param;

        return LayerTransformation::getTestCaseNameByParams(precision, shape, params);
    }
};

TEST_P(AddTransformation, CompareFunctions) {
    // InitNodeInfo().run_on_function(actualFunction);
    // ConvFusion().run_on_function(actualFunction);

    // actualFunction->validate_nodes_and_infer_types();

    // auto res = compare_functions(referenceFunction, actualFunction);
    // ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues)),
    AddTransformation::getTestCaseName);
