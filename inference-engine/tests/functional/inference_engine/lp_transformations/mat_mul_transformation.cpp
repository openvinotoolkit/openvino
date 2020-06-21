// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>
#include <map>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include "transformations/low_precision/transformer.hpp"
#include "transformations/low_precision/mat_mul.hpp"
#include "transformations/low_precision/fake_quantize.hpp"

#include "../transformations/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

using namespace testing;
using namespace ngraph::pass;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params,
    std::vector<std::shared_ptr<ngraph::Node>>> MatMulTransformationParams;

class MatMulTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const std::vector<std::shared_ptr<ngraph::Node>> nodes = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(precision, shape, nodes);

        // transform(actualFunction);

        low_precision::LowPrecisionTransformations transformations(
            std::map<std::string, low_precision::LayerTransformationPtr>({}),
            std::map<std::string, low_precision::LayerTransformationPtr>({
                { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(new low_precision::FakeQuantizeTransformation(params)) },
                { "MatMul", ngraph::pass::low_precision::LayerTransformationPtr(new low_precision::MatMulTransformation(params)) },
            }),
            std::map<std::string, low_precision::LayerTransformationPtr>({}));;
        low_precision::LowPrecisionTransformer transformer(transformations);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getReference(precision, shape, nodes);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        std::vector<std::shared_ptr<ngraph::Node>> nodes;
        std::tie(precision, shape, params, nodes) = obj.param;

        return LayerTransformation::getTestCaseNameByParams(precision, shape, params);
    }
};

TEST_P(MatMulTransformation, CompareFunctions) {
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

static const size_t levels = 256ul;
static const float k = 2.f;
static const float lowU8 = 0.f;
static const float highU8 = 255.f / k;
static const float lowI8 = -128.f / k;
static const float highI8 = 127.f / k;
static const ngraph::element::Type precision = ngraph::element::f32;

std::vector<std::vector<std::shared_ptr<ngraph::Node>>> branches = {
    // GEMM 4D : 1x16x384x64 & 1x16x64x384 = > 1x16x384x384(BERT MLPerf)
    {
        std::make_shared<ngraph::opset1::Multiply>(
            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 384, 64 })),
            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true)),

        std::make_shared<ngraph::opset1::Multiply>(
            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 64, 384 })),
            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true))
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(branches)),
    MatMulTransformation::getTestCaseName);
