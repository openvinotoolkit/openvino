// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_conv_pool_relu(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    ov::Shape const_shape = {input_shape[0], input_shape[2], input_shape[1], input_shape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});

    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});

    auto conv1 = ov::test::utils::make_convolution(reshape1,
                                                   type,
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
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    return model;
}

std::shared_ptr<ov::Model> make_conv_pool2_relu2(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    std::vector<size_t> constShape = {input_shape[0], input_shape[2], input_shape[1], input_shape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, constShape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});

    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});

    auto conv1 = ov::test::utils::make_convolution(reshape1,
                                                   type,
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
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov