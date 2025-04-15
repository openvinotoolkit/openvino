// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_conv_pool_relu_non_zero(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    auto conv1 = ov::test::utils::make_convolution(params.front(),
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

    auto non_zero = std::make_shared<ov::op::v3::NonZero>(relu1);
    non_zero->set_friendly_name("nonZero_1");
    non_zero->output(0).get_tensor().set_names({"nonZero"});

    auto gather_indices =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    gather_indices->set_friendly_name("gather_indices_1");
    gather_indices->output(0).get_tensor().set_names({"gather_indices"});

    auto gather_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    gather_axis->set_friendly_name("gather_axis_1");
    gather_axis->output(0).get_tensor().set_names({"gather_axis"});

    auto gather = std::make_shared<ov::op::v1::Gather>(non_zero->output(0), gather_indices, gather_axis);
    gather->set_friendly_name("gather_1");
    gather->output(0).get_tensor().set_names({"gather"});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov