// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/one_hot.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/one_hot.hpp"

namespace ov {
namespace test {
std::string OneHotLayerTest::getTestCaseName(const testing::TestParamInfo<oneHotLayerTestParamsSet>& obj) {
    int64_t axis;
    ov::element::Type depth_type, set_type;
    int64_t depth_val;
    float on_val, off_val;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string targetDevice;

    std::tie(depth_type, depth_val, set_type, on_val, off_val, axis, model_type, shapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "depthType=" << depth_type << "_";
    result << "depth=" << depth_val << "_";
    result << "SetValueType=" << set_type << "_";
    result << "onValue=" << on_val << "_";
    result << "offValue=" << off_val << "_";
    result << "axis=" << axis << "_";

    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void OneHotLayerTest::SetUp() {
    int64_t axis;
    ov::element::Type depth_type, set_type;
    int64_t depth_val;
    float on_val, off_val;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::tie(depth_type, depth_val, set_type, on_val, off_val, axis, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto depth_const = std::make_shared<ov::op::v0::Constant>(depth_type, ov::Shape{}, depth_val);
    auto on_value_const = std::make_shared<ov::op::v0::Constant>(set_type, ov::Shape{}, on_val);
    auto off_value_const = std::make_shared<ov::op::v0::Constant>(set_type, ov::Shape{}, off_val);
    auto onehot = std::make_shared<ov::op::v1::OneHot>(param, depth_const, on_value_const, off_value_const, axis);

    auto result = std::make_shared<ov::op::v0::Result>(onehot);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "OneHot");
}
}  // namespace test
}  // namespace ov
