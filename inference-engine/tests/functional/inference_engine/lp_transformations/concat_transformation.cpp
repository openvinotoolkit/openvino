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
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/concat.hpp>
#include <transformations/low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params> ConcatTransformationParams;

class ConcatTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginal(
            precision,
            shape,
            params);

        // TODO: do we really need transformer here to run single transformation?
        ngraph::pass::low_precision::LowPrecisionTransformations transformations(
            {
                { "Concat", ngraph::pass::low_precision::LayerTransformationPtr(new ngraph::pass::low_precision::ConcatTransformation(params)) }
            },
            {
                { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(new ngraph::pass::low_precision::FakeQuantizeTransformation(params)) }
            },
            {});
        ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
        transformer.transform(actualFunction);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReference(
            precision,
            shape,
            params);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        std::tie(precision, shape, params) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, params);
        return result.str();
    }
};

TEST_P(ConcatTransformation, CompareFunctions) {
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

const std::vector<low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

INSTANTIATE_TEST_CASE_P(
    DISABLED_LPT,
    ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues)),
    ConcatTransformation::getTestCaseName);
