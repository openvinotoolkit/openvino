// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "../transformations/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"
#include "layer_transformation.hpp"

using namespace testing;
using namespace ngraph::pass;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    low_precision::LayerTransformation::Params> DepthToSpaceTestsParams;

class DepthToSpaceTransformation : public LayerTransformation, public testing::WithParamInterface<DepthToSpaceTestsParams> {
public:
    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;

    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::DepthToSpaceFunction::getOriginal(precision, shape);
        // std::vector<std::shared_ptr<ngraph::Function>> module{ actualFunction };
        // VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(module);

        transform(actualFunction);
        // std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ actualFunction };
        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

        referenceFunction = ngraph::builder::subgraph::DepthToSpaceFunction::getReference(precision, shape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceTestsParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        std::tie(precision, shape, params) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shape << "_" << toString(params);
        return result.str();
    }
};

TEST_P(DepthToSpaceTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    ConvFusion().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

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
    DepthToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues)),
    DepthToSpaceTransformation::getTestCaseName);
