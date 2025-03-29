// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/nested_split_conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_nested_split_conv_concat(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    auto split_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   10);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);

    auto split2_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split2 = std::make_shared<ov::op::v1::Split>(relu2, split2_axis_op, 2);

    auto conv3 = ov::test::utils::make_convolution(split2->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);

    auto conv4 = ov::test::utils::make_convolution(split2->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu4 = std::make_shared<ov::op::v0::Relu>(conv4);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat1)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("NestedSplitConvConcat");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov