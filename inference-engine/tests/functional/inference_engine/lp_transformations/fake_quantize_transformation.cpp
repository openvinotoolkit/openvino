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

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

#include "../transformations/ngraph_test_utils.hpp"
#include <transformations/low_precision/transformer.hpp>
#include "transformations/low_precision/fake_quantize.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

// typedef std::pair<ngraph::Shape, std::vector<builder::subgraph::FakeQuantizeOnData>> FakeQuantizeShapes;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params,
    builder::subgraph::FakeQuantizeOnData> FakeQuantizeTransformationParams;

class FakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const ngraph::pass::low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(precision, shape, params, fakeQuantizeOnData);
        // transform(actualFunction);

        ngraph::pass::low_precision::LowPrecisionTransformations transformations(
            {},
            { { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(new ngraph::pass::low_precision::FakeQuantizeTransformation(params)) } },
            {});
        ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
        transformer.transform(actualFunction);
        //{
        //    std::vector<std::shared_ptr<ngraph::Function>> module{ actualFunction };
        //    VisualizeTree("C:\\Projects\\temp\\test.actual").run_on_module(module);
        //}

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getReference(precision, shape, params, fakeQuantizeOnData);
        //{
        //    std::vector<std::shared_ptr<ngraph::Function>> module{ referenceFunction };
        //    VisualizeTree("C:\\Projects\\temp\\test.reference").run_on_module(module);
        //}
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        std::tie(precision, shape, params, fakeQuantizeOnData) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, params) << fakeQuantizeOnData;
        return result.str();
    }
};

TEST_P(FakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f } },
    { 256ul, {}, { -1.28f} , { 1.27f } },
    // { 256ul, { 1ul }, { 0.f }, { 2.55f } },
    // { 256ul, { 1ul }, { -1.28f} , { 1.27f } }
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
    FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FakeQuantizeTransformation::getTestCaseName);
