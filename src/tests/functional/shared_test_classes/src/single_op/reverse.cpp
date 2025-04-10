// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/reverse.hpp"

namespace ov {
namespace test {

std::string ReverseLayerTest::getTestCaseName(const testing::TestParamInfo<reverseParams>& obj) {
    std::vector<size_t> input_shape;
    std::vector<int> axes;
    std::string mode;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shape, axes, mode, model_type, target_device) = obj.param;

    std::ostringstream result;
    result << "in_shape=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "mode=" << mode << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void ReverseLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    std::vector<int> axes;
    std::string mode;
    ov::element::Type model_type;
    std::tie(input_shape, axes, mode, model_type, targetDevice) = GetParam();

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    std::shared_ptr<ov::op::v0::Constant> axes_constant;
    if (mode == "index") {
        axes_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{axes.size()}, axes);
    } else {
        std::vector<bool> axes_mask(input_shape.size(), false);
        for (auto axis : axes)
            axes_mask[axis] = true;
        axes_constant =
            std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{axes_mask.size()}, axes_mask);
    }
    auto reverse = std::make_shared<ov::op::v1::Reverse>(param, axes_constant, mode);
    function = std::make_shared<ov::Model>(reverse->outputs(), ov::ParameterVector{param}, "reverse");
}
}  // namespace test
}  // namespace ov
