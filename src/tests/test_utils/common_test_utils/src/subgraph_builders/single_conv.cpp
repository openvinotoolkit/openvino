// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_single_conv(ov::Shape input_shape, ov::element::Type type) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(input_shape));

    auto conv1 = ov::test::utils::make_convolution(param0,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   4);
    auto result = std::make_shared<ov::op::v0::Result>(conv1);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0});
    model->set_friendly_name("SingleConv");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov