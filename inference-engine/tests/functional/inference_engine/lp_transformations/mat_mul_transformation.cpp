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
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/mat_mul.hpp>
#include <transformations/low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph::pass;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ngraph::builder::subgraph::MatMulFunctionBranches> MatMulTransformationParams;

class MatMulTransformation : public LayerTransformation, public testing::WithParamInterface<MatMulTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const ngraph::builder::subgraph::MatMulFunctionBranches branches = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::MatMulFunction::getOriginal(precision, shape, branches);

        // VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        ngraph::pass::low_precision::LowPrecisionTransformations transformations;
        transformations.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(params);
        transformations.add<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>(params);

        low_precision::LowPrecisionTransformer transformer(transformations);
        transformer.transform(actualFunction);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        referenceFunction = ngraph::builder::subgraph::MatMulFunction::getReference(precision, shape, branches);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        ngraph::builder::subgraph::MatMulFunctionBranches branches;
        std::tie(precision, shape, params, branches) = obj.param;

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
    // ngraph::element::f16
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

std::vector<ngraph::builder::subgraph::MatMulFunctionBranches> branches = {
    // {
    //    {{ 1, 16, 384, 64 }, {}, {}, {}, {}},
    //    {{ 1, 16, 64, 384 }, {}, {}, {}, {}}
    // },
    {
        {{ 1, 16, 384, 64 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 16, 1, 1}, {}}},
        {{ 1, 16, 64, 384 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 16, 1, 1}, {}}}
    },
    // {
    //    {{ 1, 16, 384, 64 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1}, {2.f}}},
    //    {{ 1, 16, 64, 384 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1}, {3.f}}}
    // }
};

// std::vector<std::vector<std::shared_ptr<ngraph::Node>>> branches = {
//    // GEMM 4D : 1x16x384x64 & 1x16x64x384 = > 1x16x384x384(BERT MLPerf)
//    {
//        std::make_shared<ngraph::opset1::Multiply>(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 384, 64 })),
//            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true)),
//
//        std::make_shared<ngraph::opset1::Multiply>(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 64, 384 })),
//            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true))
//    },
//
//    // GEMM 4D : 1x16x384x64 & 1x16x64x384 = > 1x16x384x384(BERT MLPerf)
//    // {
//    //    std::make_shared<ngraph::opset1::Multiply>(
//    //        std::make_shared<ngraph::opset1::Convert>(
//    //            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 384, 64 })),
//    //            ngraph::element::u8),
//    //        ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true)),
//
//    //    std::make_shared<ngraph::opset1::Multiply>(
//    //        std::make_shared<ngraph::opset1::Convert>(
//    //            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 16, 64, 384 })),
//    //            ngraph::element::u8),
//    //        ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true))
//    // }
// };

INSTANTIATE_TEST_CASE_P(
    LPT,
    MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(branches)),
    MatMulTransformation::getTestCaseName);
