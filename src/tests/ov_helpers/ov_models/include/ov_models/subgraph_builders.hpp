// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {
inline std::shared_ptr<ov::Model> makeConvPoolRelu(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                          ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    std::vector<size_t> constShape = {inputShape[0], inputShape[2], inputShape[1], inputShape[3]};
    auto const1 = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    auto reshape1 = std::make_shared<ov::opset1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});
    auto conv1 = ov::builder::makeConvolution(reshape1, ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                               ov::op::RoundingType::FLOOR,
                                                               ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ov::opset1::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});
    ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
    auto const2 = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
    const2->output(0).get_tensor().set_names({"const2"});
    const2->set_friendly_name("Const_2");
    auto reshape2 = std::make_shared<ov::opset1::Reshape>(relu1, const2, false);
    reshape2->output(0).get_tensor().set_names({"reshape2"});
    reshape2->set_friendly_name("Reshape_2");
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(reshape2)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeConvPoolReluNoReshapes(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                                    ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    auto conv1 = ov::builder::makeConvolution(params.front(), ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                           ov::op::RoundingType::FLOOR,
                                                           ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ov::opset1::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});
    ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(relu1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeConvPool2Relu2(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                            ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    std::vector<size_t> constShape = {inputShape[0], inputShape[2], inputShape[1], inputShape[3]};
    auto const1 = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    auto reshape1 = std::make_shared<ov::opset1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});
    auto conv1 = ov::builder::makeConvolution(reshape1, ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};

    ov::ResultVector results;
    {
        auto pool1 = std::make_shared<ov::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                                ov::op::RoundingType::FLOOR,
                                                                ov::op::PadType::EXPLICIT);
        pool1->output(0).get_tensor().set_names({"pool_0"});
        pool1->set_friendly_name("Pool_1_0");
        auto relu1 = std::make_shared<ov::opset1::Relu>(pool1);
        relu1->set_friendly_name("Relu_1_0");
        relu1->output(0).get_tensor().set_names({"relu_0"});
        ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
        std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
        auto const2 = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
        const2->output(0).get_tensor().set_names({"const2_0"});
        const2->set_friendly_name("Const_2_0");
        auto reshape2 = std::make_shared<ov::opset1::Reshape>(relu1, const2, false);
        reshape2->output(0).get_tensor().set_names({"reshape2_0"});
        reshape2->set_friendly_name("Reshape_2_0");
        results.push_back(std::make_shared<ov::opset1::Result>(reshape2));
    }
    {
        auto pool1 = std::make_shared<ov::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                                ov::op::RoundingType::FLOOR,
                                                                ov::op::PadType::EXPLICIT);
        pool1->output(0).get_tensor().set_names({"pool_1"});
        pool1->set_friendly_name("Pool_1_1");
        auto relu1 = std::make_shared<ov::opset1::Relu>(pool1);
        relu1->set_friendly_name("Relu_1_1");
        relu1->output(0).get_tensor().set_names({"relu_1"});
        ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
        std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
        auto const2 = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
        const2->output(0).get_tensor().set_names({"const2_1"});
        const2->set_friendly_name("Const_2_1");
        auto reshape2 = std::make_shared<ov::opset1::Reshape>(relu1, const2, false);
        reshape2->output(0).get_tensor().set_names({"reshape2_1"});
        reshape2->set_friendly_name("Reshape_2_1");
        results.push_back(std::make_shared<ov::opset1::Result>(reshape2));
    }
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeConvPoolReluNonZero(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                                 ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    auto conv1 = ov::builder::makeConvolution(params.front(), ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                               ov::op::RoundingType::FLOOR,
                                                               ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ov::opset1::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});
    auto nonZero = std::make_shared<ov::op::v3::NonZero>(relu1);
    nonZero->set_friendly_name("nonZero_1");
    nonZero->output(0).get_tensor().set_names({"nonZero"});
    auto gatherIndices = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                ov::Shape{1},
                                                                std::vector<int64_t>{0});
    gatherIndices->set_friendly_name("gatherIndices_1");
    gatherIndices->output(0).get_tensor().set_names({"gatherIndices"});
    auto gatherAxis = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                             ov::Shape{1},
                                                             std::vector<int64_t>{1});
    gatherAxis->set_friendly_name("gatherAxis_1");
    gatherAxis->output(0).get_tensor().set_names({"gatherAxis"});
    auto gather = std::make_shared<ov::opset1::Gather>(nonZero->output(0), gatherIndices, gatherAxis);
    gather->set_friendly_name("gather_1");
    gather->output(0).get_tensor().set_names({"gather"});
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(gather)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                             ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1);

    auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 1);
    concat->get_output_tensor(0).set_names({"concat_tensor"});
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeKSOFunction(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                         ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto shapeOf = std::make_shared<ov::opset4::ShapeOf>(params[0]);
    auto convert = std::make_shared<ov::opset4::Convert>(shapeOf, ngPrc);
    auto newShape = ov::builder::makeConstant<int64_t>(ov::element::i64, {4}, {1, 4, 1, 1});
    auto reshape = std::make_shared<ov::opset4::Reshape>(convert, newShape, false);
    auto conv1 = ov::builder::makeConvolution(params[1], ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    auto relu1 = std::make_shared<ov::opset4::Relu>(conv1);
    auto add = std::make_shared<ov::opset4::Add>(relu1, reshape);

    ov::ResultVector results{std::make_shared<ov::opset1::Result>(add)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("KSOFunction");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeSplitMultiConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                  ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({ "input_tensor" });
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1_0 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1_0 = std::make_shared<ov::opset1::Relu>(conv1_0);
    auto conv1_1 = ov::builder::makeConvolution(relu1_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1_1 = std::make_shared<ov::opset1::Relu>(conv1_1);
    auto conv1_2 = ov::builder::makeConvolution(relu1_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu1_2 = std::make_shared<ov::opset1::Relu>(conv1_2);
    auto conv1_3 = ov::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu1_3 = std::make_shared<ov::opset1::Relu>(conv1_3);
    auto conv1_4 = ov::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu1_4 = std::make_shared<ov::opset1::Relu>(conv1_4);

    auto conv2_0 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu2_0 = std::make_shared<ov::opset1::Relu>(conv2_0);
    auto conv2_1 = ov::builder::makeConvolution(relu2_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu2_1 = std::make_shared<ov::opset1::Relu>(conv2_1);
    auto conv2_2 = ov::builder::makeConvolution(relu2_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu2_2 = std::make_shared<ov::opset1::Relu>(conv2_2);
    auto conv2_3 = ov::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu2_3 = std::make_shared<ov::opset1::Relu>(conv2_3);
    auto conv2_4 = ov::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);
    auto relu2_4 = std::make_shared<ov::opset1::Relu>(conv2_4);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1_4->output(0), relu2_4->output(0)}, 1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitMultiConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeTIwithLSTMcell(
        ov::element::Type_t ngPRC = ov::element::Type_t::f32,
        size_t N = 32,   // Batch size
        size_t L = 10,   // Sequence length
        size_t I = 8,    // Input size
        size_t H = 32) { // Hidden size
    auto SENT = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, L, I});

    auto H_init = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, 1, H});
    auto C_init = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, 1, H});

    auto H_t = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, 1, H});
    auto C_t = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, 1, H});

    // Body
    auto X = std::make_shared<ov::opset1::Parameter>(ngPRC, ov::Shape{N, 1, I});
    std::vector<uint64_t> dataW(4 * H * I, 0);
    auto W_body = std::make_shared<ov::opset1::Constant>(ngPRC, ov::Shape{4 * H, I}, dataW);
    std::vector<uint64_t> dataR(4 * H * H, 0);
    auto R_body = std::make_shared<ov::opset1::Constant>(ngPRC, ov::Shape{4 * H, H}, dataR);
    std::vector<uint64_t> inShape = {N, H};
    auto constantH = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    inShape = {N, I};
    auto constantX = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    auto LSTM_cell =
            std::make_shared<ov::opset4::LSTMCell>(std::make_shared<ov::opset1::Reshape>(X, constantX, false),
                                                   std::make_shared<ov::opset1::Reshape>(H_t, constantH, false),
                                                   std::make_shared<ov::opset1::Reshape>(C_t, constantH, false),
                                                   W_body,
                                                   R_body,
                                                   H);
    inShape = {N, 1, H};
    auto constantHo = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, inShape);
    auto H_o = std::make_shared<ov::opset1::Reshape>(LSTM_cell->output(0), constantHo, false);
    auto C_o = std::make_shared<ov::opset1::Reshape>(LSTM_cell->output(1), constantHo, false);
    auto body = std::make_shared<ov::Model>(
            ov::OutputVector{H_o, C_o}, ov::ParameterVector{X, H_t, C_t});

    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
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

    auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(out0),
                                        std::make_shared<ov::opset1::Result>(out1)};
    auto fn_ptr = std::make_shared<ov::Model>(results, ov::ParameterVector{SENT, H_init, C_init});
    fn_ptr->set_friendly_name("TIwithLSTMcell");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                        ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param0 = std::make_shared<ov::opset1::Parameter>(type, ov::Shape(inputShape));

    auto conv1 = ov::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 4);
    auto result = std::make_shared<ov::opset1::Result>(conv1);
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0});
    fn_ptr->set_friendly_name("SingleConv");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeDetectionOutput(ov::element::Type_t type = ov::element::Type_t::f32) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(type, ov::Shape{1, 4, 10, 10});

    const auto& constant_0 = std::make_shared<ov::opset1::Constant>(type, ov::Shape{1, 1, 1, 1});
    const auto& mul_0 = std::make_shared<ov::opset1::Multiply>(data, constant_0);

    const auto& filters = std::make_shared<ov::opset1::Constant>(type, ov::Shape{1, 4, 1, 1});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(
            mul_0, filters, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});

    const auto& box_logits_reshape = std::make_shared<ov::opset1::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, -1});
    const auto& box_logits = std::make_shared<ov::opset1::Reshape>(conv, box_logits_reshape, true);

    const auto& four_times = std::make_shared<ov::opset1::Tile>(box_logits, std::make_shared<ov::opset1::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 4}));

    const auto& third_input_reshape = std::make_shared<ov::opset1::Constant>(
            ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, -1});
    const auto& third_input = std::make_shared<ov::opset1::Reshape>(four_times, third_input_reshape, true);

    ov::op::v0::DetectionOutput::Attributes attr;
    attr.num_classes = 4;
    attr.background_label_id = 0;
    attr.top_k = 75;
    attr.variance_encoded_in_target = true;
    attr.keep_top_k = {50};
    attr.code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
    attr.share_location = true;
    attr.nms_threshold = 0.5f;
    attr.confidence_threshold = 0.5f;
    attr.clip_after_nms = false;
    attr.clip_before_nms = false;
    attr.decrease_label_id = false;
    attr.normalized = true;
    attr.input_height = 1;
    attr.input_width = 1;
    attr.objectness_score = 0.4f;
    const auto& detection = std::make_shared<ov::opset1::DetectionOutput>(four_times, four_times, third_input, attr);
    const auto& convert = std::make_shared<ov::opset1::Convert>(detection, type);

    return std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{data}, "SplitableDetectionOutput");
}

inline std::shared_ptr<ov::Model> makeMultiSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24},
    ov::element::Type type = ov::element::Type_t::f32) {
    auto param0 = std::make_shared<ov::opset1::Parameter>(type, ov::Shape(inputShape));
    auto conv1 = ov::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv2 = ov::builder::makeConvolution(conv1, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv3 = ov::builder::makeConvolution(conv2, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv4 = ov::builder::makeConvolution(conv3, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv5 = ov::builder::makeConvolution(conv4, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv6 = ov::builder::makeConvolution(conv5, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv7 = ov::builder::makeConvolution(conv6, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv8 = ov::builder::makeConvolution(conv7, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                 ov::op::PadType::EXPLICIT, 5);
    auto conv9 = ov::builder::makeConvolution(conv8, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto conv10 = ov::builder::makeConvolution(conv9, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto result = std::make_shared<ov::opset1::Result>(conv10);
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0});
    fn_ptr->set_friendly_name("MultiSingleConv");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> make2InputSubtract(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                            ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param0 = std::make_shared<ov::opset1::Parameter>(type, ov::Shape(inputShape));
    auto param1 = std::make_shared<ov::opset1::Parameter>(type, ov::Shape(inputShape));
    auto subtract = std::make_shared<ov::opset1::Subtract>(param0, param1);
    auto result = std::make_shared<ov::opset1::Result>(subtract);
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
    fn_ptr->set_friendly_name("TwoInputSubtract");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeNestedBranchConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                   ov::element::Type ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto relu0 = std::make_shared<ov::opset1::Relu>(params[0]);

    auto conv1 = ov::builder::makeConvolution(relu0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1);

    auto conv2 = ov::builder::makeConvolution(relu0, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 10);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2);

    auto conv3 = ov::builder::makeConvolution(relu2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu3 = std::make_shared<ov::opset1::Relu>(conv3);

    auto conv4 = ov::builder::makeConvolution(relu2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu4 = std::make_shared<ov::opset1::Relu>(conv4);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);

    auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("NestedBranchConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeNestedSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                   ov::element::Type ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1);

    auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 10);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2);

    auto split2 = ov::builder::makeSplit(relu2, ngPrc, 2, 1);

    auto conv3 = ov::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu3 = std::make_shared<ov::opset1::Relu>(conv3);

    auto conv4 = ov::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu4 = std::make_shared<ov::opset1::Relu>(conv4);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);

    auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("NestedSplitConvConcat");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeSplitConvConcatInputInBranch(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                          ov::element::Type ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1);

    auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2);

    auto conv4 = ov::builder::makeConvolution(params[1]->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu4 = std::make_shared<ov::opset1::Relu>(conv4);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu4->output(0), relu2->output(0)}, 1);

    auto conv3 = ov::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5);
    auto relu3 = std::make_shared<ov::opset1::Relu>(conv3);

    auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatInputInBranch");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeSplitConvConcatNestedInBranch(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                                           ov::element::Type ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    int localId = 0;
    #define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1); SET_NAME(split);

    auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);  SET_NAME(conv1);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1); SET_NAME(relu1);

    auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv2);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2); SET_NAME(relu2);

    auto nestedSubgraph = [&] {
        auto split = ov::builder::makeSplit(params[1], ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ov::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ov::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ov::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ov::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ov::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ov::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ov::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ov::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ov::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();
    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{nestedSubgraph->output(0), relu2->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ov::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
    auto relu3 = std::make_shared<ov::opset1::Relu>(conv3); SET_NAME(relu3);

    auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1); SET_NAME(concat1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatNestedInBranch");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeSplitConvConcatNestedInBranchNestedOut(
        std::vector<size_t> inputShape = {1, 4, 20, 20},
        ov::element::Type ngPrc = ov::element::Type_t::f32) {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    int localId = 0;
    #define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));
    auto split = ov::builder::makeSplit(params[0], ngPrc, 2, 1); SET_NAME(split);

    auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5);  SET_NAME(conv1);
    auto relu1 = std::make_shared<ov::opset1::Relu>(conv1); SET_NAME(relu1);

    auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
    auto relu2 = std::make_shared<ov::opset1::Relu>(conv2); SET_NAME(relu2);

    auto split3 = ov::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split3);

    auto conv32 = ov::builder::makeConvolution(split3->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 10); SET_NAME(conv32);
    auto relu32 = std::make_shared<ov::opset1::Relu>(conv32); SET_NAME(relu32);

    auto nestedSubgraph = [&] {
        auto split = ov::builder::makeSplit(params[1], ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ov::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ov::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ov::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ov::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ov::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ov::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ov::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ov::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ov::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();

    auto nestedSubgraph1 = [&] {
        auto split = ov::builder::makeSplit(relu32, ngPrc, 2, 1); SET_NAME(split);

        auto conv1 = ov::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv1);
        auto relu1 = std::make_shared<ov::opset1::Relu>(conv1); SET_NAME(relu1);

        auto conv2 = ov::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 10); SET_NAME(conv2);
        auto relu2 = std::make_shared<ov::opset1::Relu>(conv2); SET_NAME(relu2);

        auto split2 = ov::builder::makeSplit(relu2, ngPrc, 2, 1); SET_NAME(split2);

        auto conv3 = ov::builder::makeConvolution(split2->output(0), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
        auto relu3 = std::make_shared<ov::opset1::Relu>(conv3); SET_NAME(relu3);

        auto conv4 = ov::builder::makeConvolution(split2->output(1), ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv4);
        auto relu4 = std::make_shared<ov::opset1::Relu>(conv4); SET_NAME(relu4);

        auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), concat}, 1); SET_NAME(concat1);

        auto conv5 = ov::builder::makeConvolution(concat1, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                    ov::op::PadType::EXPLICIT, 5); SET_NAME(conv5);
        auto relu5 = std::make_shared<ov::opset1::Relu>(conv5); SET_NAME(relu5);

        return relu5;
    }();

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{nestedSubgraph->output(0), split3->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ov::builder::makeConvolution(concat, ngPrc, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 5); SET_NAME(conv3);
    auto relu3 = std::make_shared<ov::opset1::Relu>(conv3); SET_NAME(relu3);

    auto concat1 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1); SET_NAME(concat1);
    ov::ResultVector results{std::make_shared<ov::opset1::Result>(concat1), std::make_shared<ov::opset1::Result>(nestedSubgraph1)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitConvConcatNestedInBranchNestedOut");
    return fnPtr;
}

inline std::shared_ptr<ov::Model> makeConvBias(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                      ov::element::Type type = ov::element::Type_t::f32) {
    ov::ParameterVector parameter {std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    parameter[0]->set_friendly_name("parameter");
    auto weights = ov::opset1::Constant::create(type, ov::Shape{6, 3, 1, 1}, {1});
    auto biases = ov::opset1::Constant::create(type, ov::Shape{6, 1, 1}, {1});
    auto conv = std::make_shared<ov::opset1::Convolution>(parameter[0], weights, ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
    conv->set_friendly_name("conv");
    auto add = std::make_shared<ov::opset1::Add>(conv, biases);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset1::Result>(add);
    result->set_friendly_name("result");
    std::shared_ptr<ov::Model> fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    fn_ptr->set_friendly_name("ConvBias");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeReadConcatSplitAssign(std::vector<size_t> inputShape = {1, 1, 2, 4},
                                                                   ov::element::Type type = ov::element::Type_t::f32) {
    ov::ParameterVector parameter {std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    parameter[0]->set_friendly_name("parameter");
    auto init_const = ov::op::v0::Constant::create(type, inputShape, {0});
    auto read = std::make_shared<ov::opset5::ReadValue>(init_const, "v0");
    read->set_friendly_name("read");
    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], read};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");
    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");
    const auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    axis->set_friendly_name("axis");
    auto crop = std::make_shared<ov::op::v1::Split>(conc, axis, 2);
    crop->set_friendly_name("split");
    auto assign = std::make_shared<ov::opset5::Assign>(crop, "v0");
    assign->set_friendly_name("assign");
    std::shared_ptr<ov::Model> fn_ptr = std::make_shared<ov::Model>(ov::ResultVector({res}),
                                                                                  ov::SinkVector({assign}),
                                                                                  ov::ParameterVector{parameter});
    fn_ptr->set_friendly_name("ReadConcatSplitAssign");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeMatMulBias(std::vector<size_t> inputShape = { 1, 3, 24, 24 },
                                                        ov::element::Type type = ov::element::Type_t::f32) {
    ov::ParameterVector parameter {std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    parameter[0]->set_friendly_name("parameter");
    auto weights = ov::opset1::Constant::create(type, ov::Shape{ 24, 24 }, { 1 });
    auto biases = ov::opset1::Constant::create(type, ov::Shape{ 1, 24 }, { 1 });
    auto matmul = std::make_shared<opset1::MatMul>(parameter[0], weights);
    matmul->set_friendly_name("matmul");
    auto add = std::make_shared<opset1::Add>(matmul, biases);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset1::Result>(add);
    result->set_friendly_name("result");
    std::shared_ptr<ov::Model> fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{ result }, ov::ParameterVector{ parameter });
    fn_ptr->set_friendly_name("MatMulBias");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeConvertTranspose(std::vector<size_t> inputShape = { 1, 3, 24, 24 },
                                                        std::vector<size_t> inputOrder = { 0, 1, 2, 3 },
                                                        ov::element::Type type = ov::element::Type_t::f32) {
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    const auto order = ov::op::v0::Constant::create(element::i32, {inputOrder.size()}, inputOrder);

    auto convert = std::make_shared<opset1::Convert>(params.front(), type);
    convert->set_friendly_name("convert");
    auto transpose = std::make_shared<opset1::Transpose>(convert, order);
    transpose->set_friendly_name("transpose");
    auto result = std::make_shared<ov::opset1::Result>(transpose);
    result->set_friendly_name("result");

    std::shared_ptr<ov::Model> fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{ result }, ov::ParameterVector{ params });
    fn_ptr->set_friendly_name("ConvertTranspose");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeMultipleInputOutputReLU(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                                    ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param1 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape(inputShape));
    param1->set_friendly_name("param1");
    param1->output(0).get_tensor().set_names({"data1"});
    auto param2 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape(inputShape));
    param2->set_friendly_name("param2");
    param2->output(0).get_tensor().set_names({"data2"});
    auto relu = std::make_shared<ov::opset8::Relu>(param1);
    relu->set_friendly_name("relu_op");
    relu->output(0).get_tensor().set_names({"relu"});
    auto result1 = std::make_shared<ov::opset8::Result>(relu);
    result1->set_friendly_name("result1");
    auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
    concat->set_friendly_name("concat_op");
    concat->output(0).get_tensor().set_names({"concat"});
    auto result2 = std::make_shared<ov::opset8::Result>(concat);
    result2->set_friendly_name("result2");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result1, result2},
                                                     ov::ParameterVector{param1, param2});
    fn_ptr->set_friendly_name("MultipleInputOutputReLU");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeMultipleInputOutputDoubleConcat(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                                            ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param1 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape{inputShape});
    param1->set_friendly_name("param1");
    param1->output(0).get_tensor().set_names({"data1"});
    auto param2 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape(inputShape));
    param2->set_friendly_name("param2");
    param2->output(0).get_tensor().set_names({"data2"});
    auto concat1 = std::make_shared<ov::opset8::Concat>(OutputVector{param1, param2}, 1);
    concat1->set_friendly_name("concat_op1");
    concat1->output(0).get_tensor().set_names({"concat1"});
    auto result1 = std::make_shared<ov::opset8::Result>(concat1);
    result1->set_friendly_name("result1");
    auto concat2 = std::make_shared<ov::opset8::Concat>(OutputVector{concat1, param2}, 1);
    concat2->set_friendly_name("concat_op2");
    concat2->output(0).get_tensor().set_names({"concat2"});
    auto result2 = std::make_shared<ov::opset8::Result>(concat2);
    result2->set_friendly_name("result2");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{ result1, result2 },
                                                     ov::ParameterVector{ param1, param2 });
    fn_ptr->set_friendly_name("makeMultipleInputOutputDoubleConcat");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeSingleConcatWithConstant(std::vector<size_t> inputShape = {1, 1, 2, 4},
                                                    ov::element::Type type = ov::element::Type_t::f32) {
    ov::ParameterVector parameter {std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    parameter[0]->set_friendly_name("Param_1");
    parameter[0]->output(0).get_tensor().set_names({"data"});
    auto init_const = ov::op::v0::Constant::create(type, inputShape, {0});

    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], init_const};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");
    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");

    std::shared_ptr<ov::Model> fn_ptr = std::make_shared<ov::Model>(ov::ResultVector({res}),
                                                                                  ov::ParameterVector{parameter});
    fn_ptr->set_friendly_name("SingleConcatWithConstant");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeConcatWithParams(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                    ov::element::Type_t type = ov::element::Type_t::f32) {
    auto parameter1 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape{inputShape});
    parameter1->set_friendly_name("param1");
    parameter1->output(0).get_tensor().set_names({"data1"});
    auto parameter2 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape{inputShape});
    parameter2->set_friendly_name("param2");
    parameter2->output(0).get_tensor().set_names({"data2"});
    auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{parameter1, parameter2}, 1);
    concat->set_friendly_name("concat_op");
    concat->output(0).get_tensor().set_names({"concat"});
    auto result = std::make_shared<ov::opset8::Result>(concat);
    result->set_friendly_name("result");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                     ov::ParameterVector{parameter1, parameter2});
    fn_ptr->set_friendly_name("SingleConcatWithParams");
    return fn_ptr;
}


inline std::shared_ptr<ov::Model> makeSingleSplit(std::vector<size_t> inputShape = {1, 4, 32, 32},
                                                         ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param1 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape{inputShape});
    param1->set_friendly_name("param1");
    param1->output(0).get_tensor().set_names({"data1"});
    auto axis_node = ov::opset8::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<ov::opset8::Split>(param1, axis_node, 2);
    split->set_friendly_name("split");
    split->output(0).get_tensor().set_names({"tensor_split_1"});
    split->output(1).get_tensor().set_names({"tensor_split_2"});
    auto result1 = std::make_shared<ov::opset8::Result>(split->output(0));
    result1->set_friendly_name("result1");
    auto result2 = std::make_shared<ov::opset8::Result>(split->output(1));
    result2->set_friendly_name("result2");
    auto fn_ptr =
        std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1});
    fn_ptr->set_friendly_name("SingleSplit");
    return fn_ptr;
}

inline std::shared_ptr<ov::Model> makeSplitConcat(std::vector<size_t> inputShape = {1, 4, 24, 24},
                                                         ov::element::Type_t type = ov::element::Type_t::f32) {
    auto param1 = std::make_shared<ov::opset8::Parameter>(type, ov::Shape{inputShape});
    param1->set_friendly_name("Param1");
    param1->output(0).get_tensor().set_names({"data1"});
    auto axis_node = ov::opset8::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<ov::opset8::Split>(param1, axis_node, 2);
    split->set_friendly_name("Split");
    split->output(0).get_tensor().set_names({"tensor_split_1"});
    split->output(1).get_tensor().set_names({"tensor_split_2"});

    auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{split->output(0), split->output(1)}, 1);
    concat->set_friendly_name("Concat_op");
    concat->output(0).get_tensor().set_names({"Concat"});
    auto result = std::make_shared<ov::opset8::Result>(concat);
    result->set_friendly_name("Result");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{ result },
                                                     ov::ParameterVector{ param1 });
    fn_ptr->set_friendly_name("SplitConcat");
    return fn_ptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
