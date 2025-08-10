// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/reshape.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace test {
std::string ReshapeLayerTest::getTestCaseName(const testing::TestParamInfo<reshapeParams>& obj) {
    const auto& [special_zero, model_type, input_shape, out_form_shapes, target_device] = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "OS=" << ov::test::utils::vec2str(out_form_shapes) << "_";
    result << "specialZero=" << special_zero << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ReshapeLayerTest::SetUp() {
    const auto& [special_zero, model_type, input_shape, out_form_shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{out_form_shapes.size()}, out_form_shapes);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, const_node, special_zero);
    function = std::make_shared<ov::Model>(reshape->outputs(), ov::ParameterVector{param}, "Reshape");
}
}  // namespace test
}  // namespace ov
