// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_multi_single_conv(ov::Shape input_shape, ov::element::Type type) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto conv1 = ov::test::utils::make_convolution(param0,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv2 = ov::test::utils::make_convolution(conv1,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv3 = ov::test::utils::make_convolution(conv2,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv4 = ov::test::utils::make_convolution(conv3,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv5 = ov::test::utils::make_convolution(conv4,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv6 = ov::test::utils::make_convolution(conv5,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv7 = ov::test::utils::make_convolution(conv6,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv8 = ov::test::utils::make_convolution(conv7,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv9 = ov::test::utils::make_convolution(conv8,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto conv10 = ov::test::utils::make_convolution(conv9,
                                                    type,
                                                    {3, 3},
                                                    {1, 1},
                                                    {0, 0},
                                                    {0, 0},
                                                    {1, 1},
                                                    ov::op::PadType::EXPLICIT,
                                                    5);
    auto result = std::make_shared<ov::op::v0::Result>(conv10);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0});
    model->set_friendly_name("MultiSingleConv");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov