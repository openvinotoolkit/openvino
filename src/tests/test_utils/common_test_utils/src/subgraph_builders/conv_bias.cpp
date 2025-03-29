// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/conv_bias.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_conv_bias(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    parameter[0]->set_friendly_name("parameter");

    auto weights = ov::op::v0::Constant::create(type, ov::Shape{6, 3, 1, 1}, {1});
    auto biases = ov::op::v0::Constant::create(type, ov::Shape{6, 1, 1}, {1});
    auto conv = std::make_shared<ov::op::v1::Convolution>(parameter[0],
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1});
    conv->set_friendly_name("conv");

    auto add = std::make_shared<ov::op::v1::Add>(conv, biases);
    add->set_friendly_name("add");

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("result");

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    model->set_friendly_name("ConvBias");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov