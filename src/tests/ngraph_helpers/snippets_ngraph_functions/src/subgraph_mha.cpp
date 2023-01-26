// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mha.hpp"

#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> MHAFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    std::shared_ptr<ov::Node> matmul_parent1 = transpose1;
    if (with_mul) {
        std::vector<float> mulConstData(ngraph::shape_size(constantShapes[2]));
        auto mulConst = ngraph::builder::makeConstant(precisions[1], constantShapes[2], mulConstData, true);
        matmul_parent1 = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, matmul_parent1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAFunction::initReference() const {
    auto data0 = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto data2 = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto data3 = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);
    ngraph::ParameterVector ngraphParams = {data0, data1, data2, data3};
    NodeVector subgraph_inputs = {data0, data1, data2, data3};

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    ngraph::ParameterVector subgraph_params = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    std::shared_ptr<ov::Node> matmul_parent1 = transpose1;
    if (with_mul) {
        std::vector<float> mulConstData(ngraph::shape_size(constantShapes[2]));
        auto mulConst = ngraph::builder::makeConstant(precisions[1], constantShapes[2], mulConstData, true);
        auto mulParam = std::make_shared<ngraph::opset1::Parameter>(precisions[1], mulConst->get_shape());
        matmul_parent1 = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulParam);
        subgraph_params = {transpose0Param, transpose1Param, mulParam, addParam, transpose2Param};
        subgraph_inputs = {data0, data1, mulConst, data2, data3};
    }
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, matmul_parent1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_inputs,
            std::make_shared<ov::Model>(NodeVector{transpose3}, subgraph_params));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ngraphParams);
}

std::shared_ptr<ov::Model> MHAMatMul0TransposeFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    const auto order = std::vector<int64_t>{0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], order);
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], order);
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], order);
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], order);

    std::vector<float> mulConstData(1);
    auto mulConst = ngraph::builder::makeConstant(precisions[1], ov::Shape{1}, mulConstData, true);

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul, transA, true);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAMatMul0TransposeFunction::initReference() const {
    auto data0 = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto data2 = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto data3 = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);
    ngraph::ParameterVector ngraphParams = {data0, data1, data2, data3};

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1], std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5], std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6], std::vector<int64_t>{0, 2, 1, 3});

    std::vector<float> mulConstData(1);
    auto mulConst = ngraph::builder::makeConstant(precisions[1], ov::Shape{1}, mulConstData, true);
    ngraph::ParameterVector subgraphParams = {transpose0Param, transpose1Param, addParam, transpose2Param};

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(
            NodeVector{data0, data1, data2, data3},
            std::make_shared<ov::Model>(NodeVector{transpose3}, subgraphParams));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ngraphParams);
}

std::shared_ptr<ov::Model> MHASelectFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[0], input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[1], input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precisions[2], input_shapes[2]);
    auto less0Param = std::make_shared<ngraph::opset1::Parameter>(precisions[3], input_shapes[3]);
    auto less1Param = std::make_shared<ngraph::opset1::Parameter>(precisions[4], input_shapes[4]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precisions[5], input_shapes[5]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, less0Param, less1Param, transpose2Param};

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, input_shapes[1].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({input_shapes[0].get_shape().size()}));

    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[0],
                                                         std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[1],
                                                         std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5],
                                                         std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[6],
                                                         std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] *
                                                                   input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[4], reshape1ConstData);
    // Value is equal to '1' - to avoid situation e^(-1000) / (sum(e^(-1000)) = 0/0 = NAN
    auto selectConst = ngraph::builder::makeConstant(precisions[2], ov::Shape{1}, std::vector<float>{1});

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto less = std::make_shared<ngraph::opset3::Less>(less0Param, less1Param);
    std::shared_ptr<ov::Node> selectCond = less;
    if (add->get_output_partial_shape(0) != input_shapes[3]) {
        const auto broadcast_shape = ngraph::builder::makeConstant(ngraph::element::i64, constantShapes[5],
                                                                   add->get_output_shape(0));
        const auto broadcast = ngraph::builder::makeBroadcast(selectCond, broadcast_shape,
                                                              ngraph::op::BroadcastType::NUMPY);
        selectCond = broadcast;
    }
    const auto select = std::make_shared<ngraph::opset1::Select>(selectCond, selectConst, add,
                                                                 ngraph::op::AutoBroadcastType::NUMPY);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(select, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    // to generate valid values
    less0Param->set_friendly_name("less0");
    less0Param->set_friendly_name("less1");

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}

std::shared_ptr<ov::Model> MHAWOTransposeOnInputsFunction::initOriginal() const {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    ngraph::ParameterVector ngraphParam = {param0, param1, param2};

    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 1, 3});

    float transA = false;
    float transB = false;
    const auto mulConst = ngraph::builder::makeConstant(precision, ov::Shape({1}), std::vector<float>{1}, true);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(param1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(param0, mul, transA, transB);
    const auto softmax = std::make_shared<ngraph::opset1::Softmax>(matMul0, 3);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, param2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}

std::shared_ptr<ov::Model> MHAFQAfterMatMulFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    const auto shape_rank = input_shapes[0].get_shape().size();
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape0ConstData.size()}, reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape1ConstData.size()}, reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    auto fq0 = ngraph::builder::makeFakeQuantize(matMul0, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    const auto add = std::make_shared<ngraph::opset3::Add>(fq0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    auto fq1 = ngraph::builder::makeFakeQuantize(matMul1, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fq1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAINT8MatMulFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    const auto shape_rank = input_shapes[0].get_shape().size();
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape0ConstData.size()}, reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape1ConstData.size()}, reshape1ConstData);

    auto fq0 = ngraph::builder::makeFakeQuantize(transpose0Param, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    auto fq1 = ngraph::builder::makeFakeQuantize(transpose1Param, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    auto fq2 = ngraph::builder::makeFakeQuantize(transpose2Param, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fq0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(fq1, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    auto fq3 = ngraph::builder::makeFakeQuantize(matMul0, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    const auto add = std::make_shared<ngraph::opset3::Add>(fq3, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    auto fq4 = ngraph::builder::makeFakeQuantize(reshape1, ov::element::f32, 256, {1},
                                                 {0}, {0.820726}, {0}, {0.820726});
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(fq2, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(fq4, transpose2, transA, transB);
    auto fq5 = ngraph::builder::makeFakeQuantize(matMul1, ov::element::f32, 256, {1},
                                                 {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fq5, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAFQFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    const auto shape_rank = input_shapes[0].get_shape().size();
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});

    const auto fq0 = ngraph::builder::makeFakeQuantize(transpose0Param, ov::element::f32, 256, {1},
                                                       {-5.217694}, {6.661877}, {-5.217694}, {6.661877});
    const auto fq1 = ngraph::builder::makeFakeQuantize(transpose1Param, ov::element::f32, 256, {1},
                                                       {-6.40245}, {6.45286}, {-6.40245}, {6.45286});
    const auto fq_add = ngraph::builder::makeFakeQuantize(addParam, ov::element::f32, 256, {1},
                                                          {-1000}, {0}, {-1000}, {0});

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fq0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(fq1, transpose1Const);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto mul_const = ngraph::builder::makeConstant(ov::element::i8, ov::Shape{1}, std::vector<int8_t>{127});
    const auto convert = std::make_shared<ngraph::opset1::Convert>(mul_const, ov::element::f32);
    const auto mul_deq_const = ngraph::builder::makeConstant(ov::element::f32, ov::Shape{1}, std::vector<float>{0.00098425});
    const auto mul_deq = std::make_shared<ngraph::opset1::Multiply>(convert, mul_deq_const);
    const auto mul = std::make_shared<ngraph::opset1::Multiply>(transpose1, mul_deq);
    auto fq1_1 = ngraph::builder::makeFakeQuantize(mul, ov::element::f32, 256, {1},
                                                   {-0.8003067}, {0.8066083}, {-0.8003067}, {0.8066083});
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, fq1_1, transA, transB);
    auto fq2 = ngraph::builder::makeFakeQuantize(matMul0, ov::element::f32, 256, {1},
                                                   {-14.50351}, {17.65645}, {-14.50351}, {17.65645});
    const auto add = std::make_shared<ngraph::opset1::Add>(fq2, fq_add);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(add, 3);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softMax, transpose2, transA, transB);
    auto fq3 = ngraph::builder::makeFakeQuantize(matMul1, ov::element::f32, 256, {1},
                                                 {-1.895786}, {2.0028071}, {-1.895786}, {2.0028071});
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fq3, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAINT8MatMulTypeRelaxedFunction::initOriginal() const {
    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParam = {transpose0Param, transpose1Param, addParam, transpose2Param};

    const auto shape_rank = input_shapes[0].get_shape().size();
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape0ConstData.size()}, reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape1ConstData.size()}, reshape1ConstData);

    const auto fq_signed_params = ngraph::builder::subgraph::FakeQuantizeOnData(256, {1}, {-36912.66015625}, {36624.28125}, {-128}, {127}, ov::element::i8);
    const auto fq0 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(transpose0Param, ov::element::i8, fq_signed_params);
    const auto fq1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(transpose1Param, ov::element::i8, fq_signed_params);
    const auto fq2 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(transpose2Param, ov::element::i8, fq_signed_params);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fq0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(fq1, transpose1Const);
    const auto matMul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(transpose0, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(transpose1, element::f32).get(), transA, transB);

    const auto fq3 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(matMul0, ov::element::i8, fq_signed_params);
    const auto add = std::make_shared<op::TypeRelaxed<ngraph::opset3::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(fq3, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(addParam, element::f32).get());
    const auto deq = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.1122});
    const auto deq_mul = std::make_shared<op::TypeRelaxed<ngraph::opset3::Multiply>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(add, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(deq, element::f32).get());

    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);

    const auto fq_unsigned_params = ngraph::builder::subgraph::FakeQuantizeOnData(256, {1}, {0}, {0.245}, {0}, {255}, ov::element::u8);
    const auto fq4 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(reshape1, ov::element::u8, fq_unsigned_params);

    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(fq2, transpose2Const);
    const auto matMul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(fq4, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(transpose2, element::f32).get(), transA, transB);
    const auto fq5 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(matMul1, ov::element::i8, fq_signed_params);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fq5, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ov::Model>(results, ngraphParam, "mha");
}
std::shared_ptr<ov::Model> MHAINT8MatMulTypeRelaxedFunction::initReference() const {
    auto data0 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ngraph::ParameterVector ngraphParams = {data0, data1, data2, data3};

    const auto fq_signed_params = ngraph::builder::subgraph::FakeQuantizeOnData(256, {1}, {-36912.66015625}, {36624.28125}, {-128}, {127}, ov::element::i8);
    const auto fq0 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(data0, ov::element::i8, fq_signed_params);
    const auto fq1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(data1, ov::element::i8, fq_signed_params);
    const auto fq2 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(data3, ov::element::i8, fq_signed_params);
    NodeVector subgraph_inputs = {fq0, fq1, data2, fq2};

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[1]);
    auto addParam = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[2]);
    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[3]);
    ov::ParameterVector subgraph_params = {transpose0Param, transpose1Param, addParam, transpose2Param};

    const auto shape_rank = input_shapes[0].get_shape().size();
    auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose1Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 3, 1});
    auto transpose2Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose3Const = ngraph::builder::makeConstant(ngraph::element::i64, {shape_rank}, std::vector<int64_t>{0, 2, 1, 3});

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0] *
                                                                   input_shapes[0].get_shape()[1] * input_shapes[0].get_shape()[2]),
                                              -1};
    auto reshape0Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape0ConstData.size()}, reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(input_shapes[0].get_shape()[0]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[2]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1]),
                                              static_cast<int64_t>(input_shapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ngraph::element::i64, {reshape1ConstData.size()}, reshape1ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(transpose0, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(transpose1, element::f32).get(), transA, transB);

    const auto fq3 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(matMul0, ov::element::i8, fq_signed_params);
    const auto add = std::make_shared<op::TypeRelaxed<ngraph::opset3::Add>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(fq3, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(addParam, element::f32).get());
    const auto deq = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.1122});
    const auto deq_mul = std::make_shared<op::TypeRelaxed<ngraph::opset3::Multiply>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(add, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(deq, element::f32).get());

    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);

    const auto fq_unsigned_params = ngraph::builder::subgraph::FakeQuantizeOnData(256, {1}, {0}, {0.245}, {0}, {255}, ov::element::u8);
    const auto fq4 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(reshape1, ov::element::u8, fq_unsigned_params);

    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ov::op::TemporaryReplaceOutputType(fq4, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(transpose2, element::f32).get(), transA, transB);
    const auto fq5 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(matMul1, ov::element::i8, fq_signed_params);

    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_inputs,
                                                                     std::make_shared<ov::Model>(NodeVector{fq5}, subgraph_params));
    // TODO: At the moment Snippets don't support explicitly Transpose.
    //       So we cannot collapse Transpose into Subgraph if there are ops between MatMul2 and Transpose3
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(subgraph, transpose3Const);

    return std::make_shared<ov::Model>(NodeVector{transpose3}, ngraphParams);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
