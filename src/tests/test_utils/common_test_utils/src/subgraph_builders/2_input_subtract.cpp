// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_2_input_subtract(ov::Shape input_shape, ov::element::Type type) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(param0, param1);
    auto result = std::make_shared<ov::op::v0::Result>(subtract);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
    model->set_friendly_name("TwoInputSubtract");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov