// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/roll.hpp"

namespace ov {
namespace test {
std::string RollLayerTest::getTestCaseName(const testing::TestParamInfo<rollParams>& obj) {
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    std::vector<int64_t> shift;
    std::vector<int64_t> axes;
    std::string target_device;
    std::tie(input_shapes, model_type, shift, axes, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "modelType=" << model_type.to_string() << "_";
    result << "Shift=" << ov::test::utils::vec2str(shift) << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void RollLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    std::vector<int64_t> shift;
    std::vector<int64_t> axes;
    std::string target_device;
    std::tie(input_shapes, model_type, shift, axes, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.at(0));
    auto shift_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shift.size()}, shift);
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
    auto roll = std::make_shared<ov::op::v7::Roll>(param, shift_const, axes_const);
    function = std::make_shared<ov::Model>(roll->outputs(), ov::ParameterVector{param}, "Roll");
}
}  // namespace test
}  // namespace ov
