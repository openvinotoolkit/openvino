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
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/relu_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class ReluTransformation : public LayerTransformation, public testing::WithParamInterface<LayerTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::ReluFunction::getOriginal(precision, shape);
        // transform(actualFunction);
        referenceFunction = ngraph::builder::subgraph::ReluFunction::getReference(precision, shape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<LayerTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        std::tie(precision, shape, params) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shape << "_" << toString(params);
        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};

TEST_P(ReluTransformation, CompareFunctions) {
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
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues)),
    ReluTransformation::getTestCaseName);
