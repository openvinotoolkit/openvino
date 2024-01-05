// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/subgraph_builders.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> makeConvPoolRelu(std::vector<size_t> inputShape, ov::element::Type_t ngPrc) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    std::vector<size_t> constShape = {inputShape[0], inputShape[2], inputShape[1], inputShape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});
    auto conv1 = ngraph::builder::makeConvolution(reshape1,
                                                  ngPrc,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                       stride,
                                                       padB,
                                                       padE,
                                                       kernel,
                                                       ov::op::RoundingType::FLOOR,
                                                       ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});
    ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
    auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
    const2->output(0).get_tensor().set_names({"const2"});
    const2->set_friendly_name("Const_2");
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(relu1, const2, false);
    reshape2->output(0).get_tensor().set_names({"reshape2"});
    reshape2->set_friendly_name("Reshape_2");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape2)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

std::shared_ptr<ov::Model> makeConvPool2Relu2(std::vector<size_t> inputShape, ov::element::Type_t ngPrc) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    std::vector<size_t> constShape = {inputShape[0], inputShape[2], inputShape[1], inputShape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});
    auto conv1 = ngraph::builder::makeConvolution(reshape1,
                                                  ngPrc,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};

    ov::ResultVector results;
    {
        auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                           stride,
                                                           padB,
                                                           padE,
                                                           kernel,
                                                           ov::op::RoundingType::FLOOR,
                                                           ov::op::PadType::EXPLICIT);
        pool1->output(0).get_tensor().set_names({"pool_0"});
        pool1->set_friendly_name("Pool_1_0");
        auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
        relu1->set_friendly_name("Relu_1_0");
        relu1->output(0).get_tensor().set_names({"relu_0"});
        ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
        std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
        auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
        const2->output(0).get_tensor().set_names({"const2_0"});
        const2->set_friendly_name("Const_2_0");
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(relu1, const2, false);
        reshape2->output(0).get_tensor().set_names({"reshape2_0"});
        reshape2->set_friendly_name("Reshape_2_0");
        results.push_back(std::make_shared<ov::op::v0::Result>(reshape2));
    }
    {
        auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                           stride,
                                                           padB,
                                                           padE,
                                                           kernel,
                                                           ov::op::RoundingType::FLOOR,
                                                           ov::op::PadType::EXPLICIT);
        pool1->output(0).get_tensor().set_names({"pool_1"});
        pool1->set_friendly_name("Pool_1_1");
        auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
        relu1->set_friendly_name("Relu_1_1");
        relu1->output(0).get_tensor().set_names({"relu_1"});
        ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
        std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
        auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
        const2->output(0).get_tensor().set_names({"const2_1"});
        const2->set_friendly_name("Const_2_1");
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(relu1, const2, false);
        reshape2->output(0).get_tensor().set_names({"reshape2_1"});
        reshape2->set_friendly_name("Reshape_2_1");
        results.push_back(std::make_shared<ov::op::v0::Result>(reshape2));
    }
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

std::shared_ptr<ov::Model> makeConvPoolReluNonZero(std::vector<size_t> inputShape, ov::element::Type_t ngPrc) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});
    auto conv1 = ngraph::builder::makeConvolution(params.front(),
                                                  ngPrc,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                       stride,
                                                       padB,
                                                       padE,
                                                       kernel,
                                                       ov::op::RoundingType::FLOOR,
                                                       ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");
    auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});
    auto nonZero = std::make_shared<ov::op::v3::NonZero>(relu1);
    nonZero->set_friendly_name("nonZero_1");
    nonZero->output(0).get_tensor().set_names({"nonZero"});
    auto gatherIndices =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    gatherIndices->set_friendly_name("gatherIndices_1");
    gatherIndices->output(0).get_tensor().set_names({"gatherIndices"});
    auto gatherAxis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    gatherAxis->set_friendly_name("gatherAxis_1");
    gatherAxis->output(0).get_tensor().set_names({"gatherAxis"});
    auto gather = std::make_shared<ov::op::v1::Gather>(nonZero->output(0), gatherIndices, gatherAxis);
    gather->set_friendly_name("gather_1");
    gather->output(0).get_tensor().set_names({"gather"});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    return fnPtr;
}

std::shared_ptr<ov::Model> makeSplitConvConcat(std::vector<size_t> inputShape, ov::element::Type_t ngPrc) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto split_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0),
                                                  ngPrc,
                                                  {3, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  5);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1),
                                                  ngPrc,
                                                  {3, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  5);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 1);
    concat->get_output_tensor(0).set_names({"concat_tensor"});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("SplitConvConcat");
    return fnPtr;
}

std::shared_ptr<ov::Model> makeKSOFunction(std::vector<size_t> inputShape, ov::element::Type_t ngPrc) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
    auto convert = std::make_shared<ov::op::v0::Convert>(shapeOf, ngPrc);
    auto newShape = ov::test::utils::deprecated::make_constant<int64_t>(ov::element::i64, {4}, {1, 4, 1, 1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, newShape, false);
    auto conv1 = ngraph::builder::makeConvolution(params[1],
                                                  ngPrc,
                                                  {3, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  4);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
    auto add = std::make_shared<ov::op::v1::Add>(relu1, reshape);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    std::shared_ptr<ov::Model> fnPtr = std::make_shared<ov::Model>(results, params);
    fnPtr->set_friendly_name("KSOFunction");
    return fnPtr;
}

std::shared_ptr<ov::Model> makeSingleConcatWithConstant(std::vector<size_t> inputShape, ov::element::Type type) {
    ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(inputShape))};
    parameter[0]->set_friendly_name("Param_1");
    parameter[0]->output(0).get_tensor().set_names({"data"});
    auto init_const = ov::op::v0::Constant::create(type, inputShape, {0});

    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], init_const};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");
    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");

    std::shared_ptr<ov::Model> fn_ptr =
        std::make_shared<ov::Model>(ov::ResultVector({res}), ov::ParameterVector{parameter});
    fn_ptr->set_friendly_name("SingleConcatWithConstant");
    return fn_ptr;
}

std::shared_ptr<ov::Model> makeConcatWithParams(std::vector<size_t> inputShape, ov::element::Type_t type) {
    auto parameter1 = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{inputShape});
    parameter1->set_friendly_name("param1");
    parameter1->output(0).get_tensor().set_names({"data1"});
    auto parameter2 = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{inputShape});
    parameter2->set_friendly_name("param2");
    parameter2->output(0).get_tensor().set_names({"data2"});
    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{parameter1, parameter2}, 1);
    concat->set_friendly_name("concat_op");
    concat->output(0).get_tensor().set_names({"concat"});
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("result");
    auto fn_ptr = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter1, parameter2});
    fn_ptr->set_friendly_name("SingleConcatWithParams");
    return fn_ptr;
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
