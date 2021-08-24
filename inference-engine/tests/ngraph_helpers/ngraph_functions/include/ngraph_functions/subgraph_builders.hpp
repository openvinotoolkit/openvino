// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
inline std::shared_ptr<ngraph::Function> makeConvPoolRelu(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                          ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    std::vector<size_t> constShape = {inputShape[0], inputShape[2], inputShape[1], inputShape[3]};
    auto const1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    auto conv1 = ngraph::builder::makeConvolution(reshape1, ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ngraph::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                               ngraph::op::RoundingType::FLOOR,
                                                               ngraph::op::PadType::EXPLICIT);
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    ngraph::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ngraph::shape_size(reluShape)};
    auto const2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, constShape2);
    const2->set_friendly_name("Const_2");
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(relu1, const2, false);
    reshape2->set_friendly_name("Reshape_2");
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                            ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("SplitConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeKSOFunction(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                         ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto shapeOf = std::make_shared<ngraph::opset4::ShapeOf>(params[0]);
    auto convert = std::make_shared<ngraph::opset4::Convert>(shapeOf, ngPrc);
    auto newShape = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {4}, {1, 4, 1, 1});
    auto reshape = std::make_shared<ngraph::opset4::Reshape>(convert, newShape, false);
    auto conv1 = ngraph::builder::makeConvolution(params[0], ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 4);
    auto relu1 = std::make_shared<ngraph::opset4::Relu>(conv1);
    auto add = std::make_shared<ngraph::opset4::Add>(relu1, reshape);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("KSOFunction");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeSplitMultiConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20}) {
    auto ngPrc = ngraph::element::Type_t::f32;
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1_0 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_0 = std::make_shared<ngraph::opset1::Relu>(conv1_0);
    auto conv1_1 = ngraph::builder::makeConvolution(relu1_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_1 = std::make_shared<ngraph::opset1::Relu>(conv1_1);
    auto conv1_2 = ngraph::builder::makeConvolution(relu1_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_2 = std::make_shared<ngraph::opset1::Relu>(conv1_2);
    auto conv1_3 = ngraph::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_3 = std::make_shared<ngraph::opset1::Relu>(conv1_3);
    auto conv1_4 = ngraph::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_4 = std::make_shared<ngraph::opset1::Relu>(conv1_4);

    auto conv2_0 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_0 = std::make_shared<ngraph::opset1::Relu>(conv2_0);
    auto conv2_1 = ngraph::builder::makeConvolution(relu2_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_1 = std::make_shared<ngraph::opset1::Relu>(conv2_1);
    auto conv2_2 = ngraph::builder::makeConvolution(relu2_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_2 = std::make_shared<ngraph::opset1::Relu>(conv2_2);
    auto conv2_3 = ngraph::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_3 = std::make_shared<ngraph::opset1::Relu>(conv2_3);
    auto conv2_4 = ngraph::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_4 = std::make_shared<ngraph::opset1::Relu>(conv2_4);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1_4->output(0), relu2_4->output(0)}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("SplitMultiConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeTIwithLSTMcell(
        ngraph::element::Type_t ngPRC = ngraph::element::Type_t::f32,
        size_t N = 32,   // Batch size
        size_t L = 10,   // Sequence length
        size_t I = 8,    // Input size
        size_t H = 32) { // Hidden size
    auto SENT = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, L, I});

    auto H_init = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});
    auto C_init = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});

    auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});
    auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});

    // Body
    auto X = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, I});
    std::vector<uint64_t> dataW(4 * H * I, 0);
    auto W_body = std::make_shared<ngraph::opset1::Constant>(ngPRC, ngraph::Shape{4 * H, I}, dataW);
    std::vector<uint64_t> dataR(4 * H * H, 0);
    auto R_body = std::make_shared<ngraph::opset1::Constant>(ngPRC, ngraph::Shape{4 * H, H}, dataR);
    std::vector<uint64_t> inShape = {N, H};
    auto constantH = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, inShape);
    inShape = {N, I};
    auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, inShape);
    auto LSTM_cell =
            std::make_shared<ngraph::opset4::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
                                                   std::make_shared<ngraph::opset1::Reshape>(H_t, constantH, false),
                                                   std::make_shared<ngraph::opset1::Reshape>(C_t, constantH, false),
                                                   W_body,
                                                   R_body,
                                                   H);
    inShape = {N, 1, H};
    auto constantHo = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{3}, inShape);
    auto H_o = std::make_shared<ngraph::opset1::Reshape>(LSTM_cell->output(0), constantHo, false);
    auto C_o = std::make_shared<ngraph::opset1::Reshape>(LSTM_cell->output(1), constantHo, false);
    auto body = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{H_o, C_o}, ngraph::ParameterVector{X, H_t, C_t});

    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
    tensor_iterator->set_body(body);
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, 1);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(out0),
                                        std::make_shared<ngraph::opset1::Result>(out1)};
    auto fn_ptr = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{SENT, H_init, C_init});
    fn_ptr->set_friendly_name("TIwithLSTMcell");
    return fn_ptr;
}

inline std::shared_ptr<ngraph::Function> makeSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                        ngraph::element::Type_t type = ngraph::element::Type_t::f32) {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));

    auto conv1 = ngraph::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 4);
    auto result = std::make_shared<ngraph::opset1::Result>(conv1);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
    fn_ptr->set_friendly_name("SingleConv");
    return fn_ptr;
}

inline std::shared_ptr<ngraph::Function> makeMultiSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24},
    ngraph::element::Type type = ngraph::element::Type_t::f32) {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto conv1 = ngraph::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv2 = ngraph::builder::makeConvolution(conv1, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv3 = ngraph::builder::makeConvolution(conv2, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv4 = ngraph::builder::makeConvolution(conv3, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv5 = ngraph::builder::makeConvolution(conv4, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv6 = ngraph::builder::makeConvolution(conv5, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv7 = ngraph::builder::makeConvolution(conv6, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv8 = ngraph::builder::makeConvolution(conv7, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                 ngraph::op::PadType::EXPLICIT, 5);
    auto conv9 = ngraph::builder::makeConvolution(conv8, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv10 = ngraph::builder::makeConvolution(conv9, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto result = std::make_shared<ngraph::opset1::Result>(conv10);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
    fn_ptr->set_friendly_name("MultiSingleConv");
    return fn_ptr;
}

inline std::shared_ptr<ngraph::Function> make2InputSubtract(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                            ngraph::element::Type_t type = ngraph::element::Type_t::f32) {
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto subtract = std::make_shared<ngraph::opset1::Subtract>(param0, param1);
    auto result = std::make_shared<ngraph::opset1::Result>(subtract);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0, param1});
    fn_ptr->set_friendly_name("TwoInputSubtract");
    return fn_ptr;
}

inline std::shared_ptr<ngraph::Function> makeNestedSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                   ngraph::element::Type ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 10);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);

    auto split2 = ngraph::builder::makeSplit(relu2, ngPrc, 2, 1);

    auto conv3 = ngraph::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3);

    auto conv4 = ngraph::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu4 = std::make_shared<ngraph::opset1::Relu>(conv4);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu3->output(0), relu4->output(0)}, 1);

    auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), concat}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat1)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("NestedSplitConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeSplitConvConcatInputInBranch(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                          ngraph::element::Type ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);

    auto conv4 = ngraph::builder::makeConvolution(params[1]->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu4 = std::make_shared<ngraph::opset1::Relu>(conv4);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu4->output(0), relu2->output(0)}, 1);

    auto conv3 = ngraph::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3);

    auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu3->output(0)}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat1)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatInputInBranch");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeSplitConvConcatNestedInBranch(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                           ngraph::element::Type ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});
    int localId = 0;
    #define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1); SET_NAME(split);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);  SET_NAME(conv1);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1); SET_NAME(relu1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv2);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2); SET_NAME(relu2);

    auto nestedSubgraph = [&] {
        auto split = ngraph::builder::makeSplit(params[1], ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ngraph::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ngraph::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ngraph::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ngraph::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ngraph::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ngraph::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{nestedSubgraph->output(0), relu2->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ngraph::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3); SET_NAME(relu3);

    auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu3->output(0)}, 1); SET_NAME(concat1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat1)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatNestedInBranch");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeSplitConvConcatNestedInBranchNestedOut(
        std::vector<size_t> inputShape = {1, 4, 20, 20},
        ngraph::element::Type ngPrc = ngraph::element::Type_t::f32) {
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});
    int localId = 0;
    #define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1); SET_NAME(split);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);  SET_NAME(conv1);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1); SET_NAME(relu1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2); SET_NAME(relu2);

    auto split3 = ngraph::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split3);

    auto conv32 = ngraph::builder::makeConvolution(split3->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 10); SET_NAME(conv32);
    auto relu32 = std::make_shared<ngraph::opset1::Relu>(conv32); SET_NAME(relu32);

    auto nestedSubgraph = [&] {
        auto split = ngraph::builder::makeSplit(params[1], ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ngraph::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ngraph::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ngraph::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ngraph::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ngraph::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ngraph::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();

    auto nestedSubgraph1 = [&] {
        auto split = ngraph::builder::makeSplit(relu32, ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ngraph::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ngraph::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ngraph::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ngraph::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ngraph::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ngraph::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{nestedSubgraph->output(0), split3->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ngraph::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(conv3); SET_NAME(relu3);

    auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu3->output(0)}, 1); SET_NAME(concat1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat1), std::make_shared<ngraph::opset1::Result>(nestedSubgraph1)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatNestedInBranchNestedOut");
    return fnPtr;
}

inline std::shared_ptr<ngraph::Function> makeConvBias(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                      ngraph::element::Type type = ngraph::element::Type_t::f32) {
    auto parameter =  ngraph::builder::makeParams(type, {inputShape});
    parameter[0]->set_friendly_name("parameter");
    auto weights = ngraph::opset1::Constant::create(type, ngraph::Shape{6, 3, 1, 1}, {1});
    auto biases = ngraph::opset1::Constant::create(type, ngraph::Shape{6, 1, 1}, {1});
    auto conv = std::make_shared<ngraph::opset1::Convolution>(parameter[0], weights, ngraph::Strides{1, 1},
            ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{1, 1});
    conv->set_friendly_name("conv");
    auto add = std::make_shared<ngraph::opset1::Add>(conv, biases);
    add->set_friendly_name("add");
    auto result = std::make_shared<ngraph::opset1::Result>(add);
    result->set_friendly_name("result");
    std::shared_ptr<ngraph::Function> fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter});
    fn_ptr->set_friendly_name("ConvBias");
    return fn_ptr;
}

inline std::shared_ptr<ngraph::Function> makeReadConcatSplitAssign(std::vector<size_t> inputShape = {1, 1, 2, 4},
                                                                   ngraph::element::Type type = ngraph::element::Type_t::f32) {
    auto parameter =  ngraph::builder::makeParams(type, {inputShape});
    parameter[0]->set_friendly_name("parameter");
    auto init_const = ngraph::op::Constant::create(element::f32, Shape{1, 1, 2, 2}, {0, 0, 0, 0});
    auto read = std::make_shared<ngraph::opset5::ReadValue>(init_const, "v0");
    read->set_friendly_name("read");
    std::vector<std::shared_ptr<ngraph::Node>> args = {parameter[0], read};
    auto conc = std::make_shared<ngraph::op::Concat>(args, 3);
    conc->set_friendly_name("concat");
    auto res = std::make_shared<ngraph::op::Result>(conc);
    res->set_friendly_name("result");
    const auto axis = ngraph::op::Constant::create(element::i64, Shape{}, {3});
    axis->set_friendly_name("axis");
    auto crop = std::make_shared<ngraph::op::v1::Split>(conc, axis, 3);
    crop->set_friendly_name("crop");
    auto assign = std::make_shared<ngraph::opset5::Assign>(crop, "v0");
    assign->set_friendly_name("assign");

    std::shared_ptr<ngraph::Function> fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector({res}),
                                                                                  ngraph::SinkVector({assign}),
                                                                                  ngraph::ParameterVector{parameter});
    fn_ptr->set_friendly_name("ReadConcatSplitAssign");
    return fn_ptr;
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
